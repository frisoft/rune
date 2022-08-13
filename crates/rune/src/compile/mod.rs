//! The Rune compiler.
//!
//! The main entry to compiling rune source is [prepare][crate::prepare] which
//! uses this compiler. In here you'll just find compiler-specific types.

mod assembly;
pub(crate) use self::assembly::{Assembly, AssemblyAddress};

pub(crate) mod assemble;
pub(crate) use self::assemble::{Allocator, Name};

pub(crate) mod attrs;

mod compile_error;
pub use self::compile_error::{CompileError, CompileErrorKind, ImportStep};

mod compile_visitor;
pub use self::compile_visitor::CompileVisitor;
pub(crate) use self::compile_visitor::NoopCompileVisitor;

pub(crate) mod context;
pub use self::context::{Context, ContextError, ContextSignature, ContextTypeInfo};

mod prelude;
pub(crate) use self::prelude::Prelude;

pub(crate) mod ir;
pub(crate) use self::ir::{IrBudget, IrCompiler, IrEvalContext, IrEvalOutcome, IrInterpreter};
pub use self::ir::{IrError, IrErrorKind, IrEval, IrValue};

pub(crate) mod item;
pub use self::item::{Component, ComponentRef, IntoComponent, Item, ItemBuf};

mod source_loader;
pub use self::source_loader::{FileSourceLoader, SourceLoader};

mod unit_builder;
pub use self::unit_builder::LinkerError;
pub(crate) use self::unit_builder::UnitBuilder;

mod options;
pub use self::options::{Options, ParseOptionError};

mod label;
pub(crate) use self::label::Label;

mod location;
pub use self::location::Location;

mod meta;
pub(crate) use self::meta::{
    CaptureMeta, ContextMeta, ContextMetaKind, Doc, ItemMeta, PrivMeta, PrivMetaKind,
    PrivStructMeta, PrivTupleMeta, PrivVariantMeta,
};
pub use self::meta::{Meta, MetaKind, MetaRef, SourceMeta};

mod module;
pub use self::module::{
    AssocType, AsyncFunction, AsyncInstFn, Function, InstFn, InstallWith, Module, Variant,
};

mod pool;
pub(crate) use self::pool::{ItemId, ModId, ModMeta, Pool};

mod named;
pub use self::named::Named;

mod names;
pub(crate) use self::names::Names;

mod visibility;
pub(crate) use self::visibility::Visibility;

use crate::arena::Arena;
use crate::ast::{self, Span, Spanned};
use crate::hir;
use crate::macros::Storage;
use crate::parse::Resolve;
use crate::query::{Build, BuildEntry, Query};
use crate::shared::{Consts, Gen};
use crate::worker::{LoadFileKind, Task, Worker};
use crate::{Diagnostics, Sources};

/// Encode the given object into a collection of asm.
pub(crate) fn compile(
    unit: &mut UnitBuilder,
    prelude: &Prelude,
    sources: &mut Sources,
    pool: &mut Pool,
    context: &Context,
    diagnostics: &mut Diagnostics,
    options: &Options,
    visitor: &mut dyn CompileVisitor,
    source_loader: &mut dyn SourceLoader,
) -> Result<(), ()> {
    // Shared id generator.
    let gen = Gen::new();
    let mut consts = Consts::default();
    let mut storage = Storage::default();
    let mut inner = Default::default();

    // The worker queue.
    let mut worker = Worker::new(
        context,
        &mut consts,
        &mut storage,
        sources,
        pool,
        options,
        unit,
        prelude,
        diagnostics,
        visitor,
        source_loader,
        &gen,
        &mut inner,
    );

    // Queue up the initial sources to be loaded.
    for source_id in worker.q.sources.source_ids() {
        let mod_item = match worker.q.insert_root_mod(source_id, Span::empty()) {
            Ok(result) => result,
            Err(error) => {
                worker.q.diagnostics.error(source_id, error);
                return Err(());
            }
        };

        worker.queue.push_back(Task::LoadFile {
            kind: LoadFileKind::Root,
            source_id,
            mod_item,
        });
    }

    worker.run();

    if worker.q.diagnostics.has_error() {
        return Err(());
    }

    loop {
        while let Some(entry) = worker.q.next_build_entry() {
            tracing::trace!("next build entry: {}", entry.item_meta.item);
            let source_id = entry.item_meta.location.source_id;

            let task = CompileBuildEntry {
                context,
                options,
                q: worker.q.borrow(),
            };

            if let Err(error) = task.compile(entry) {
                worker.q.diagnostics.error(source_id, error);
            }
        }

        match worker.q.queue_unused_entries() {
            Ok(true) => (),
            Ok(false) => break,
            Err((source_id, error)) => {
                worker.q.diagnostics.error(source_id, error);
            }
        }
    }

    if worker.q.diagnostics.has_error() {
        return Err(());
    }

    Ok(())
}

struct CompileBuildEntry<'a> {
    context: &'a Context,
    options: &'a Options,
    q: Query<'a>,
}

impl CompileBuildEntry<'_> {
    #[tracing::instrument(skip(self, entry))]
    fn compile(mut self, entry: BuildEntry) -> Result<(), CompileError> {
        let BuildEntry {
            item_meta,
            build,
            used,
        } = entry;

        let location = item_meta.location;

        match build {
            Build::Query => {
                tracing::trace!("query: {}", self.q.pool.item(item_meta.item));

                if self
                    .q
                    .query_meta(item_meta.location.span, item_meta.item, used)?
                    .is_none()
                {
                    return Err(CompileError::new(
                        item_meta.location.span,
                        CompileErrorKind::MissingItem {
                            item: self.q.pool.item(item_meta.item).to_owned(),
                        },
                    ));
                }
            }
            Build::Function(function) => {
                tracing::trace!("function: {}", self.q.pool.item(item_meta.item));

                let arena = Arena::new();

                let (scopes, asm) = {
                    let mut asm = self.q.unit.new_assembly(location);

                    let hir = {
                        let mut cx =
                            hir::lowering::Ctxt::new(location.source_id, self.q.borrow(), &arena);
                        hir::lowering::item_fn(&mut cx, &function.ast)?
                    };

                    let mut cx = self::assemble::Ctxt::new(
                        self.context,
                        self.q.borrow(),
                        &mut asm,
                        self.options,
                        &arena,
                    );
                    self::assemble::fn_from_item_fn(&mut cx, &hir, false)?;
                    (cx.into_allocator(), asm)
                };

                if used.is_unused() {
                    self.q
                        .diagnostics
                        .not_used(location.source_id, function.ast.span(), None);
                    return Ok(());
                }

                let args = format_fn_args(
                    self.q.sources,
                    location,
                    function.ast.args.iter().map(|(a, _)| a),
                )?;

                self.q.unit.new_function(
                    location,
                    self.q.pool.item(item_meta.item),
                    function.ast.args.len(),
                    asm,
                    scopes,
                    function.call,
                    args,
                )?;
            }
            Build::InstanceFunction(instance_function) => {
                tracing::trace!("instance function: {}", self.q.pool.item(item_meta.item));

                let arena = crate::arena::Arena::new();

                let (scopes, asm) = {
                    let mut asm = self.q.unit.new_assembly(location);

                    let hir = {
                        let mut cx =
                            hir::lowering::Ctxt::new(location.source_id, self.q.borrow(), &arena);
                        hir::lowering::item_fn(&mut cx, &instance_function.function.ast)?
                    };

                    let mut cx = self::assemble::Ctxt::new(
                        self.context,
                        self.q.borrow(),
                        &mut asm,
                        self.options,
                        &arena,
                    );
                    self::assemble::fn_from_item_fn(&mut cx, &hir, true)?;
                    (cx.into_allocator(), asm)
                };

                if used.is_unused() {
                    self.q
                        .diagnostics
                        .not_used(location.source_id, location.span, None);
                    return Ok(());
                }

                let name = instance_function
                    .function
                    .ast
                    .name
                    .resolve(resolve_context!(self.q))?;

                let args = format_fn_args(
                    self.q.sources,
                    location,
                    instance_function.function.ast.args.iter().map(|(a, _)| a),
                )?;

                self.q.unit.new_instance_function(
                    location,
                    self.q.pool.item(item_meta.item),
                    self.q.pool.item_type_hash(instance_function.impl_item),
                    name,
                    instance_function.function.ast.args.len(),
                    asm,
                    scopes,
                    instance_function.function.call,
                    args,
                )?;
            }
            Build::Closure(closure) => {
                tracing::trace!("closure: {}", self.q.pool.item(item_meta.item));

                let arena = Arena::new();

                let (scopes, asm) = {
                    let mut asm = self.q.unit.new_assembly(location);

                    let hir = {
                        let mut cx =
                            hir::lowering::Ctxt::new(location.source_id, self.q.borrow(), &arena);
                        hir::lowering::expr_closure(&mut cx, &closure.ast)?
                    };

                    let mut cx = self::assemble::Ctxt::new(
                        self.context,
                        self.q.borrow(),
                        &mut asm,
                        self.options,
                        &arena,
                    );

                    self::assemble::closure_from_expr_closure(&mut cx, &hir, &closure.captures)?;

                    (cx.into_allocator(), asm)
                };

                if used.is_unused() {
                    self.q
                        .diagnostics
                        .not_used(location.source_id, location.span, None);
                    return Ok(());
                }

                let args = format_fn_args(
                    self.q.sources,
                    location,
                    closure.ast.args.as_slice().iter().map(|(a, _)| a),
                )?;

                self.q.unit.new_function(
                    location,
                    self.q.pool.item(item_meta.item),
                    closure.ast.args.len(),
                    asm,
                    scopes,
                    closure.call,
                    args,
                )?;
            }
            Build::AsyncBlock(b) => {
                tracing::trace!("async block: {}", self.q.pool.item(item_meta.item));

                let arena = Arena::new();

                let (scopes, asm) = {
                    let mut asm = self.q.unit.new_assembly(location);

                    let hir = {
                        let mut cx =
                            hir::lowering::Ctxt::new(location.source_id, self.q.borrow(), &arena);
                        hir::lowering::block(&mut cx, &b.ast)?
                    };

                    let mut cx = self::assemble::Ctxt::new(
                        self.context,
                        self.q.borrow(),
                        &mut asm,
                        self.options,
                        &arena,
                    );
                    self::assemble::closure_from_block(&mut cx, &hir, &b.captures)?;
                    (cx.into_allocator(), asm)
                };

                if used.is_unused() {
                    self.q
                        .diagnostics
                        .not_used(location.source_id, location.span, None);
                    return Ok(());
                }

                self.q.unit.new_function(
                    location,
                    self.q.pool.item(item_meta.item),
                    b.captures.len(),
                    asm,
                    scopes,
                    b.call,
                    Default::default(),
                )?;
            }
            Build::Unused => {
                tracing::trace!("unused: {}", self.q.pool.item(item_meta.item));

                if !item_meta.visibility.is_public() {
                    self.q
                        .diagnostics
                        .not_used(location.source_id, location.span, None);
                }
            }
            Build::Import(import) => {
                tracing::trace!("import: {}", self.q.pool.item(item_meta.item));

                // Issue the import to check access.
                let result =
                    self.q
                        .import(location.span, item_meta.module, item_meta.item, used)?;

                if used.is_unused() {
                    self.q
                        .diagnostics
                        .not_used(location.source_id, location.span, None);
                }

                let missing = match result {
                    Some(item_id) => {
                        let item = self.q.pool.item(item_id);

                        if self.context.contains_prefix(item) || self.q.contains_prefix(item) {
                            None
                        } else {
                            Some(item_id)
                        }
                    }
                    None => Some(import.entry.target),
                };

                if let Some(item) = missing {
                    return Err(CompileError::new(
                        location.span,
                        CompileErrorKind::MissingItem {
                            item: self.q.pool.item(item).to_owned(),
                        },
                    ));
                }
            }
            Build::ReExport => {
                tracing::trace!("re-export: {}", self.q.pool.item(item_meta.item));

                let import =
                    match self
                        .q
                        .import(location.span, item_meta.module, item_meta.item, used)?
                    {
                        Some(item) => item,
                        None => {
                            return Err(CompileError::new(
                                location.span,
                                CompileErrorKind::MissingItem {
                                    item: self.q.pool.item(item_meta.item).to_owned(),
                                },
                            ))
                        }
                    };

                self.q.unit.new_function_reexport(
                    location,
                    self.q.pool.item(item_meta.item),
                    self.q.pool.item(import),
                )?;
            }
        }

        Ok(())
    }
}

fn format_fn_args<'a, I>(
    sources: &Sources,
    location: Location,
    arguments: I,
) -> Result<Box<[Box<str>]>, CompileError>
where
    I: IntoIterator<Item = &'a ast::FnArg>,
{
    let mut args = Vec::new();

    for arg in arguments {
        match arg {
            ast::FnArg::SelfValue(..) => {
                args.push("self".into());
            }
            ast::FnArg::Pat(pat) => {
                let span = pat.span();

                if let Some(s) = sources.source(location.source_id, span) {
                    args.push(s.into());
                } else {
                    args.push("*".into());
                }
            }
        }
    }

    Ok(args.into())
}
