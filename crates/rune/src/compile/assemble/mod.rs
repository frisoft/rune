use std::fmt;
use std::mem;
use std::ops::Deref;
use std::ops::Neg as _;
use std::vec;

use num::ToPrimitive;
use rune_macros::__instrument_hir as instrument;

use crate::arena::{Arena, ArenaAllocError, ArenaWriteSliceOutOfBounds};
use crate::ast::{self, Span, Spanned};
use crate::collections::{HashMap, HashSet};
use crate::compile::{
    ir, Assembly, AssemblyAddress, CaptureMeta, CompileError, CompileErrorKind, IrBudget,
    IrCompiler, IrInterpreter, Item, ItemId, ItemMeta, Label, Location, Options, PrivMeta,
    PrivMetaKind, PrivVariantMeta, ScopeId, Scopes,
};
use crate::hash::ParametersBuilder;
use crate::hir;
use crate::parse::{ParseErrorKind, Resolve};
use crate::query::{Named, Query, QueryConstFn, Used};
use crate::runtime::{AssemblyInst as Inst, ConstValue, InstRangeLimits, InstValue, PanicReason};
use crate::{Context, Hash, SourceId};

/// Allocate a single object in the arena.
macro_rules! alloc {
    ($cx:expr; $value:expr) => {
        $cx.arena.alloc($value).map_err(|e| {
            $cx.error(crate::compile::CompileErrorKind::ArenaAllocError {
                requested: e.requested,
            })
        })?
    };
}

/// Unpacks an optional value and allocates it in the arena.
macro_rules! option {
    ($cx:expr; $value:expr, |$pat:pat_param| $closure:expr) => {
        match $value {
            Some($pat) => {
                Some(&*alloc!($cx; $closure))
            }
            None => {
                None
            }
        }
    };
}

/// Unpacks an iterator value and allocates it in the arena as a slice.
macro_rules! iter {
    ($cx:expr; $iter:expr) => {
        iter!($cx; $iter, |item| item)
    };

    ($cx:expr; $iter:expr, |$pat:pat_param| $closure:expr) => {{
        let mut it = IntoIterator::into_iter($iter);

        let mut writer = match $cx.arena.alloc_iter(ExactSizeIterator::len(&it)) {
            Ok(writer) => writer,
            Err(e) => {
                return Err($cx.error(
                    crate::compile::CompileErrorKind::ArenaAllocError {
                        requested: e.requested,
                    },
                ));
            }
        };

        while let Some($pat) = it.next() {
            if let Err(e) = writer.write($closure) {
                return Err($cx.error(
                    crate::compile::CompileErrorKind::ArenaWriteSliceOutOfBounds { index: e.index },
                ));
            }
        }

        writer.finish()
    }};
}

mod bound_pat;
use self::bound_pat::{BoundPat, BoundPatKind};

mod conditions;
use self::conditions::{ConditionBranch, Conditions};

mod expr;
use self::expr::{Expr, ExprBreakValue, ExprKind, ExprOutcome, ExprStructKind, ExprUnOp};

mod matches;
use self::matches::{MatchBranch, Matches};

mod type_match;
use self::type_match::TypeMatch;

mod pat;
use self::pat::{Pat, PatKind, PatObjectKind, PatOutcome, PatPath, PatTupleKind};

/// The needs of an expression.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Needs {
    Value,
    Type,
}

#[derive(Debug, Clone, Copy)]
enum AllocKind {
    Temporary,
    Allocated,
    Array(usize),
}

/// An allocated address that might've been allocated. It has to be freed.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub(crate) struct MaybeAlloc {
    /// The address being referenced.
    address: AssemblyAddress,
    /// Indicates if the address is temporary or not. If it is, the receiver
    /// must free it.
    kind: AllocKind,
}

impl MaybeAlloc {
    /// Construct a temporary address.
    pub(crate) fn temporary(address: AssemblyAddress) -> Self {
        Self {
            address,
            kind: AllocKind::Temporary,
        }
    }

    /// Construct an allocated address.
    pub(crate) fn allocated(address: AssemblyAddress) -> Self {
        Self {
            address,
            kind: AllocKind::Allocated,
        }
    }

    /// Construct an array address.
    pub(crate) fn array(address: AssemblyAddress, len: usize) -> Self {
        Self {
            address,
            kind: AllocKind::Array(len),
        }
    }

    /// Free the address.
    fn free(self, cx: &mut Ctxt<'_, '_>) -> Result<AssemblyAddress> {
        match self.kind {
            AllocKind::Temporary => {}
            AllocKind::Allocated => {
                cx.scopes.free(cx.span, self.address)?;
            }
            AllocKind::Array(len) => {
                cx.scopes.free_array(cx.span, len)?;
            }
        }

        Ok(self.address)
    }
}

impl Deref for MaybeAlloc {
    type Target = AssemblyAddress;

    fn deref(&self) -> &Self::Target {
        &self.address
    }
}

/// `self` variable.
const SELF: &str = "self";

type Result<T> = std::result::Result<T, CompileError>;

use Needs::*;

#[derive(Clone, Copy, Default)]
enum CtxtState {
    #[default]
    Default,
    Unreachable {
        reported: bool,
    },
}

/// Handler for dealing with an assembler.
pub(crate) struct Ctxt<'a, 'hir> {
    /// The context we are compiling for.
    context: &'a Context,
    /// Query system to compile required items.
    pub(self) q: Query<'a>,
    /// The assembly we are generating.
    asm: &'a mut Assembly,
    /// Enabled optimizations.
    // TODO: remove this
    #[allow(unused)]
    options: &'a Options,
    /// Arena to use for allocations.
    arena: &'hir Arena,
    /// Source being processed.
    pub(self) source_id: SourceId,
    /// Context for which to emit warnings.
    pub(self) span: Span,
    /// Scope associated with the context.
    scope: ScopeId,
    /// Scopes declared in context.
    scopes: Scopes,
    /// State of code generation.
    state: CtxtState,
}

impl Spanned for Ctxt<'_, '_> {
    fn span(&self) -> Span {
        self.span
    }
}

impl<'a, 'hir> Ctxt<'a, 'hir> {
    pub(crate) fn new(
        context: &'a Context,
        q: Query<'a>,
        asm: &'a mut Assembly,
        options: &'a Options,
        arena: &'hir Arena,
    ) -> Self {
        let source_id = asm.location.source_id;
        let span = asm.location.span;

        Self {
            context,
            q,
            asm,
            options,
            arena,
            source_id,
            span,
            scope: ScopeId::CONST,
            scopes: Scopes::new(),
            state: CtxtState::default(),
        }
    }

    /// Access the scopes build by the context.
    pub(crate) fn into_scopes(self) -> Scopes {
        self.scopes
    }

    /// Access the meta for the given language item.
    #[tracing::instrument(skip_all)]
    pub(crate) fn try_lookup_meta(&mut self, span: Span, item: ItemId) -> Result<Option<PrivMeta>> {
        tracing::trace!(?item, "lookup meta");

        if let Some(meta) = self.q.query_meta(span, item, Default::default())? {
            tracing::trace!(?meta, "found in query");
            self.q.visitor.visit_meta(
                Location::new(self.source_id, span),
                meta.as_meta_ref(self.q.pool),
            );
            return Ok(Some(meta));
        }

        if let Some(meta) = self.context.lookup_meta(self.q.pool.item(item)) {
            let meta = self.q.insert_context_meta(span, meta)?;
            tracing::trace!(?meta, "found in context");
            self.q.visitor.visit_meta(
                Location::new(self.source_id, span),
                meta.as_meta_ref(self.q.pool),
            );
            return Ok(Some(meta));
        }

        Ok(None)
    }

    /// Access the meta for the given language item.
    pub(crate) fn lookup_meta(&mut self, spanned: Span, item: ItemId) -> Result<PrivMeta> {
        if let Some(meta) = self.try_lookup_meta(spanned, item)? {
            return Ok(meta);
        }

        Err(CompileError::new(
            spanned,
            CompileErrorKind::MissingItem {
                item: self.q.pool.item(item).to_owned(),
            },
        ))
    }

    /// Convert an [ast::Path] into a [Named] item.
    pub(crate) fn convert_path(&mut self, path: &hir::Path<'hir>) -> Result<Named<'hir>> {
        self.q.convert_path(self.context, path)
    }

    /// Get the latest relevant warning context.
    pub(crate) fn context(&self) -> Span {
        self.span
    }

    /// Calling a constant function by id and return the resuling value.
    pub(crate) fn call_const_fn(
        &mut self,
        meta: &PrivMeta,
        from: &ItemMeta,
        query_const_fn: &QueryConstFn,
        args: &[hir::Expr<'_>],
    ) -> Result<ConstValue> {
        if query_const_fn.ir_fn.args.len() != args.len() {
            return Err(self.error(CompileErrorKind::UnsupportedArgumentCount {
                meta: meta.info(self.q.pool),
                expected: query_const_fn.ir_fn.args.len(),
                actual: args.len(),
            }));
        }

        let mut compiler = IrCompiler {
            source_id: self.source_id,
            q: self.q.borrow(),
        };

        let mut compiled = vec::Vec::new();

        // TODO: precompile these and fetch using opaque id?
        for (hir, name) in args.iter().zip(&query_const_fn.ir_fn.args) {
            compiled.push((ir::compile::expr(hir, &mut compiler)?, name));
        }

        let mut interpreter = IrInterpreter {
            budget: IrBudget::new(1_000_000),
            scopes: Default::default(),
            module: from.module,
            item: from.item,
            q: self.q.borrow(),
        };

        for (ir, name) in compiled {
            let value = interpreter.eval_value(&ir, Used::Used)?;
            interpreter.scopes.decl(name, value, self.span)?;
        }

        interpreter.module = query_const_fn.item_meta.module;
        interpreter.item = query_const_fn.item_meta.item;
        let value = interpreter.eval_value(&query_const_fn.ir_fn.ir, Used::Used)?;
        Ok(value.into_const(self.span)?)
    }

    /// Perform the given operation under a state checkpoint.
    fn with_state_checkpoint<T, O>(&mut self, op: T) -> Result<O>
    where
        T: FnOnce(&mut Self) -> Result<O>,
    {
        let state = self.state;
        let output = op(self)?;
        self.state = state;
        Ok(output)
    }

    /// Run `op` with a temporarily associated span.
    fn with_span<T, O>(&mut self, span: Span, op: T) -> Result<O>
    where
        T: FnOnce(&mut Self) -> Result<O>,
    {
        let span = mem::replace(&mut self.span, span);
        let output = op(self)?;
        self.span = span;
        Ok(output)
    }

    /// Run `op` with a temporarily associated scope.
    fn with_scope<T, O>(&mut self, scope: ScopeId, op: T) -> Result<O>
    where
        T: FnOnce(&mut Self) -> Result<O>,
    {
        let scope = mem::replace(&mut self.scope, scope);
        let output = op(self)?;
        self.scope = scope;
        Ok(output)
    }

    /// Construct a pattern within the current span.
    fn pat(&self, kind: PatKind<'hir>) -> Pat<'hir> {
        Pat::new(self.span, kind)
    }

    /// Construct a new bound pattern with the current span.
    fn bound_pat(&self, kind: BoundPatKind<'hir>) -> BoundPat<'hir> {
        BoundPat::new(self.span, kind)
    }

    /// Construct a new expression associated with the current context span.
    fn expr(&self, kind: ExprKind<'hir>) -> Expr<'hir> {
        Expr::new(self.span, self.scope, kind)
    }

    /// Construct a compile error associated with the current scope.
    fn error<K>(&self, kind: K) -> CompileError
    where
        CompileErrorKind: From<K>,
    {
        CompileError::new(self.span, kind)
    }

    /// Construct a new label.
    fn new_label(&mut self, label: &'static str) -> Label {
        if !matches!(self.state, CtxtState::Unreachable { .. }) {
            self.asm.new_label(label)
        } else {
            Label::EMPTY
        }
    }

    /// Label the current position.
    fn label(&mut self, label: Label) -> Result<()> {
        if !matches!(self.state, CtxtState::Unreachable { .. }) {
            self.asm.label(self.span, label)?;
        }

        Ok(())
    }

    /// Push an instruction using the current context as a span.
    fn push(&mut self, inst: Inst) {
        if !matches!(self.state, CtxtState::Unreachable { .. }) {
            self.asm.push(self.span, inst);
        }
    }

    /// Push with a comment.
    fn push_with_comment<C>(&mut self, inst: Inst, comment: C)
    where
        C: fmt::Display,
    {
        if !matches!(self.state, CtxtState::Unreachable { .. }) {
            self.asm.push_with_comment(self.span, inst, comment);
        }
    }

    /// Allocate space for the specified array.
    fn array(&mut self, array: &[Expr<'hir>]) -> Result<MaybeAlloc> {
        let address = self.scopes.array_index();

        for &hir in array {
            let output = self.scopes.array_index();
            hir.compile(self, Some(output))?.free(self)?;
            self.scopes.alloc_array_item();
        }

        Ok(MaybeAlloc::array(address, array.len()))
    }

    /// Process a maybe addres into a real address, allocating it if necessary.
    fn alloc_or(&mut self, address: Option<AssemblyAddress>) -> MaybeAlloc {
        match address {
            Some(address) => MaybeAlloc::temporary(address),
            None => MaybeAlloc::allocated(self.scopes.alloc()),
        }
    }

    /// Free all addresses given.
    fn free_iter<I>(&mut self, addresses: I) -> Result<()>
    where
        I: IntoIterator<Item = MaybeAlloc>,
    {
        for address in addresses {
            address.free(self)?;
        }

        Ok(())
    }

    /// Attempt to allocate a collection of expressions to a collection addresses
    /// which are required to be distinct.
    ///
    /// The caller can also specify `tmp` as a collection of "temporary variables"
    /// that are available for use. Note that this is typically specified as the
    /// output address of an operation.
    fn addresses<const N: usize, I>(
        &mut self,
        exprs: [Expr<'hir>; N],
        tmp: I,
    ) -> Result<[MaybeAlloc; N]>
    where
        I: IntoIterator<Item = AssemblyAddress>,
    {
        let mut used = HashSet::new();
        let mut tmp = tmp.into_iter();

        let mut spans = [Span::empty(); N];
        let mut out = [mem::MaybeUninit::<MaybeAlloc>::uninit(); N];
        let mut outcomes = Vec::with_capacity(N);

        for ((expr, out), span) in exprs.into_iter().zip(&mut out).zip(&mut spans) {
            *span = expr.span;
            let (address, outcome) = one_address(expr, self, &mut tmp, &mut used)?;
            out.write(MaybeAlloc::allocated(address));
            outcomes.push(outcome);
        }

        // SAFETY: we just initialized the array above.
        let output = unsafe { array_assume_init(out) };

        for outcome in outcomes {
            outcome.free(self)?;
        }

        return Ok(output);

        pub unsafe fn array_assume_init<T, const N: usize>(
            array: [mem::MaybeUninit<T>; N],
        ) -> [T; N] {
            let ret = (&array as *const _ as *const [T; N]).read();
            ret
        }

        /// Attempt to allocate a single managed address.
        fn one_address<'hir>(
            expr: Expr<'hir>,
            cx: &mut Ctxt<'_, 'hir>,
            tmp: &mut impl Iterator<Item = AssemblyAddress>,
            used: &mut HashSet<AssemblyAddress>,
        ) -> Result<(AssemblyAddress, ExprOutcome)> {
            let (address, outcome) = match expr.kind {
                // NB: This is the kind of item produced when a variable is used.
                ExprKind::Address { address, .. } => {
                    cx.scopes.retain(expr.span, address)?;
                    expr.scope.free(cx)?;
                    (address, ExprOutcome::Empty)
                }
                _ => {
                    let address = if let Some(output) =
                        tmp.filter(|address| !used.contains(address)).next()
                    {
                        cx.scopes.retain(expr.span, output)?;
                        output
                    } else {
                        cx.scopes.alloc()
                    };

                    let outcome = expr.compile(cx, Some(address))?;
                    (address, outcome)
                }
            };

            // We maintain a local set of addresses which have been used to assert
            // that we don't inadvertently allocate the same address to say both
            // operands.
            used.insert(address);
            Ok((address, outcome))
        }
    }
}

/// Assemble an async block.
#[instrument]
pub(crate) fn closure_from_block<'hir>(
    cx: &mut Ctxt<'_, 'hir>,
    hir: &hir::Block<'hir>,
    captures: &[CaptureMeta],
) -> Result<()> {
    let scope = cx.scopes.push(cx.span, None)?;

    let expr = cx.with_scope(scope, |cx| {
        for capture in captures {
            let binding_name = cx.scopes.binding_name(capture.ident.as_ref());
            cx.scopes.declare(cx.span, scope, binding_name)?;
        }

        assemble_block(cx, hir)
    })?;

    expr.into_return(cx)?.compile(cx, None)?.free(cx)?;
    cx.scopes.pop(cx.span, scope)?;
    Ok(())
}

/// Assemble the body of a closure function.
#[instrument]
pub(crate) fn closure_from_expr_closure<'hir>(
    cx: &mut Ctxt<'_, 'hir>,
    hir: &hir::ExprClosure<'hir>,
    captures: &[CaptureMeta],
) -> Result<()> {
    let scope = cx.scopes.push(cx.span, None)?;

    cx.with_scope(scope, |cx| {
        for capture in captures {
            let binding_name = cx.scopes.binding_name(capture.ident.as_ref());
            cx.scopes.declare(cx.span, scope, binding_name)?;
        }

        let mut patterns = Vec::new();

        for arg in hir.args {
            match arg {
                hir::FnArg::SelfValue(s) => {
                    return Err(CompileError::new(s, CompileErrorKind::UnsupportedSelf))
                }
                hir::FnArg::Pat(hir) => {
                    let address = cx.scopes.alloc();

                    let expr = cx.expr(ExprKind::Address {
                        address,
                        binding: None,
                    });

                    patterns.push((hir, expr));
                }
            }
        }

        let panic = cx.new_label("argument_patterns");
        let mut outcome = PatOutcome::Irrefutable;

        for (pat, expr) in patterns {
            outcome = outcome.combine(assemble_pat(cx, pat)?.bind(cx, expr)?.compile(cx, panic)?);
        }

        if let PatOutcome::Refutable = outcome {
            let end = cx.new_label("argument_end");
            cx.push(Inst::Jump { label: end });
            cx.label(panic)?;
            cx.push(Inst::Panic {
                reason: PanicReason::UnmatchedPattern,
            });
            cx.label(end)?;
        }

        assemble_expr_value(cx, hir.body)?
            .into_return(cx)?
            .compile(cx, None)?
            .free(cx)?;
        Ok(())
    })?;

    cx.scopes.pop(cx.span, scope)?;
    Ok(())
}

/// Assemble a function from an [hir::ItemFn<'_>].
#[instrument]
pub(crate) fn fn_from_item_fn<'hir>(
    cx: &mut Ctxt<'_, 'hir>,
    hir: &hir::ItemFn<'hir>,
    instance_fn: bool,
) -> Result<()> {
    let span = hir.span();

    let scope = cx.scopes.push(cx.span, None)?;

    cx.with_scope(scope, |cx| {
        let mut first = true;

        let mut patterns = Vec::new();

        for arg in hir.args {
            match *arg {
                hir::FnArg::SelfValue(span) => {
                    if !instance_fn || !first {
                        return Err(CompileError::new(span, CompileErrorKind::UnsupportedSelf));
                    }

                    let binding_name = cx.scopes.binding_name(SELF);
                    cx.scopes.declare(span, scope, binding_name)?;
                }
                hir::FnArg::Pat(hir) => {
                    let address = cx.scopes.alloc();

                    let expr = cx.expr(ExprKind::Address {
                        address,
                        binding: None,
                    });

                    patterns.push((hir, expr));
                }
            }

            first = false;
        }

        let panic = cx.new_label("argument_patterns");
        let mut outcome = PatOutcome::Irrefutable;

        for (hir, expr) in patterns {
            outcome = outcome.combine(assemble_pat(cx, hir)?.bind(cx, expr)?.compile(cx, panic)?);
        }

        if let PatOutcome::Refutable = outcome {
            let end = cx.new_label("argument_end");
            cx.push(Inst::Jump { label: end });
            cx.label(panic)?;
            cx.push(Inst::Panic {
                reason: PanicReason::UnmatchedPattern,
            });
            cx.label(end)?;
        }

        assemble_block(cx, hir.body)?
            .into_return(cx)?
            .compile(cx, None)?
            .free(cx)?;
        Ok(())
    })?;

    cx.scopes.pop(span, scope)?;
    Ok(())
}

/// Compile an expression.
fn assemble_expr_value<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: &hir::Expr<'hir>) -> Result<Expr<'hir>> {
    assemble_expr(cx, hir, Value)
}

/// Custom needs expression.
fn assemble_expr<'hir>(
    cx: &mut Ctxt<'_, 'hir>,
    hir: &hir::Expr<'hir>,
    needs: Needs,
) -> Result<Expr<'hir>> {
    cx.with_span(hir.span, |cx| {
        let expr = match hir.kind {
            hir::ExprKind::Empty => cx.expr(ExprKind::Empty),
            hir::ExprKind::Assign(hir) => assemble_expr_assign(cx, hir.lhs, hir.rhs)?,
            hir::ExprKind::Block(hir) => assemble_expr_block(cx, hir)?,
            hir::ExprKind::Group(hir) => assemble_expr(cx, hir, needs)?,
            hir::ExprKind::Path(path) => assemble_expr_path(cx, path, needs)?,
            hir::ExprKind::Binary(hir) => assemble_expr_binary(cx, hir)?,
            hir::ExprKind::Lit(lit) => assemble_expr_lit(cx, lit)?,
            hir::ExprKind::FieldAccess(hir) => assemble_expr_field_access(cx, hir)?,
            hir::ExprKind::Loop(hir) => assemble_expr_loop(cx, hir)?,
            hir::ExprKind::Call(hir) => assemble_expr_call(cx, hir)?,
            hir::ExprKind::Let(hir) => assemble_expr_let(cx, hir)?,
            hir::ExprKind::If(hir) => assemble_expr_if(cx, hir)?,
            hir::ExprKind::Match(hir) => assemble_expr_match(cx, hir)?,
            hir::ExprKind::Unary(hir) => assemble_expr_unary(cx, hir)?,
            hir::ExprKind::Index(hir) => assemble_expr_index(cx, hir)?,
            hir::ExprKind::Break(hir) => assemble_expr_break(cx, hir)?,
            hir::ExprKind::Continue(ast) => assemble_expr_continue(cx, ast)?,
            hir::ExprKind::Yield(hir) => assemble_expr_yield(cx, hir)?,
            hir::ExprKind::Return(hir) => assemble_expr_return(cx, hir)?,
            hir::ExprKind::Await(hir) => assemble_expr_await(cx, hir)?,
            hir::ExprKind::Try(hir) => assemble_expr_try(cx, hir)?,
            hir::ExprKind::Select(_) => todo!(),
            hir::ExprKind::Closure(hir) => assemble_expr_closure(cx, hir)?,
            hir::ExprKind::Object(hir) => assemble_expr_object(cx, hir)?,
            hir::ExprKind::Tuple(hir) => assemble_expr_tuple(cx, hir)?,
            hir::ExprKind::Vec(hir) => assemble_expr_vec(cx, hir)?,
            hir::ExprKind::Range(hir) => assemble_expr_range(cx, hir)?,
            hir::ExprKind::MacroCall(hir) => assemble_macro_call(cx, hir)?,
        };

        Ok(expr)
    })
}

/// A block expression.
#[instrument]
fn assemble_expr_block<'hir>(
    cx: &mut Ctxt<'_, 'hir>,
    hir: &hir::ExprBlock<'hir>,
) -> Result<Expr<'hir>> {
    cx.with_span(hir.block.span, |cx| {
        if let hir::ExprBlockKind::Default = hir.kind {
            let scope = cx.scopes.push(cx.span, Some(cx.scope))?;
            return cx.with_scope(scope, |cx| Ok(assemble_block(cx, hir.block)?.free_scope()));
        }

        let item = cx.q.item_for(hir.block)?;
        let meta = cx.lookup_meta(hir.block.span, item.item)?;

        let expr = match (hir.kind, &meta.kind) {
            (hir::ExprBlockKind::Async, PrivMetaKind::AsyncBlock { captures, .. }) => {
                let captures = captures.as_ref();
                let mut args = Vec::new();

                for ident in captures {
                    let (binding, address) = cx.scopes.lookup(
                        Location::new(cx.source_id, cx.span),
                        cx.scope,
                        cx.q.visitor,
                        &ident.ident,
                    )?;

                    args.push(cx.expr(ExprKind::Address {
                        binding: Some(binding),
                        address,
                    }));
                }

                let hash = cx.q.pool.item_type_hash(meta.item_meta.item);
                cx.expr(ExprKind::CallHash {
                    hash,
                    args: iter!(cx; args),
                })
            }
            (hir::ExprBlockKind::Const, PrivMetaKind::Const { const_value }) => {
                assemble_expr_const(cx, const_value)?
            }
            _ => {
                return Err(cx.error(CompileErrorKind::ExpectedMeta {
                    meta: meta.info(cx.q.pool),
                    expected: "async or const block",
                }));
            }
        };

        Ok(expr)
    })
}

/// Call a block.
#[instrument]
fn assemble_block<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: &hir::Block<'hir>) -> Result<Expr<'hir>> {
    cx.with_span(hir.span(), |cx| {
        for stmt in hir.statements {
            let expr = match *stmt {
                hir::Stmt::Local(hir) => {
                    let pat = alloc!(cx; assemble_pat(cx, hir.pat)?);
                    let expr = alloc!(cx; assemble_expr_value(cx, hir.expr)?);
                    cx.expr(ExprKind::Let { pat, expr })
                }
                hir::Stmt::Expr(hir) => assemble_expr_value(cx, hir)?,
            };

            expr.compile(cx, None)?.free(cx)?;
        }

        let tail = if let Some(hir) = hir.tail {
            assemble_expr_value(cx, hir)?
        } else {
            cx.expr(ExprKind::Empty)
        };

        Ok(tail)
    })
}

/// Process a pattern binding.
#[instrument]
fn assemble_pat<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: &hir::Pat<'hir>) -> Result<Pat<'hir>> {
    cx.with_span(hir.span(), |cx| {
        let pat = match hir.kind {
            hir::PatKind::PatPath(hir) => {
                let path = alloc!(cx; assemble_pat_path(cx, hir)?);
                cx.pat(PatKind::Path { path })
            }
            hir::PatKind::PatIgnore => cx.pat(PatKind::Ignore),
            hir::PatKind::PatLit(hir) => {
                let lit = alloc!(cx; assemble_expr_value(cx, hir)?);
                cx.pat(PatKind::Lit { lit })
            }
            hir::PatKind::PatVec(hir) => {
                let items = iter!(cx; hir.items, |hir| assemble_pat(cx, hir)?);
                cx.pat(PatKind::Vec {
                    items,
                    is_open: hir.is_open,
                })
            }
            hir::PatKind::PatTuple(hir) => assemble_pat_tuple(cx, hir)?,
            hir::PatKind::PatObject(hir) => assemble_pat_object(cx, hir)?,
        };

        Ok(pat)
    })
}

/// Assemble a binding pattern which is *just* a variable captured from an object.
#[instrument]
fn assemble_object_key<'hir>(
    cx: &mut Ctxt<'_, 'hir>,
    hir: &hir::ObjectKey<'hir>,
) -> Result<Pat<'hir>> {
    match *hir {
        hir::ObjectKey::LitStr(..) => Err(cx.error(CompileErrorKind::UnsupportedPattern)),
        hir::ObjectKey::Path(hir) => {
            let path = alloc!(cx; assemble_pat_path(cx, hir)?);
            Ok(cx.pat(PatKind::Path { path }))
        }
    }
}

/// Assemble a tuple pattern.
#[instrument]
fn assemble_pat_tuple<'hir>(
    cx: &mut Ctxt<'_, 'hir>,
    hir: &hir::PatItems<'hir>,
) -> Result<Pat<'hir>> {
    let path = match hir.path {
        Some(hir) => Some(assemble_pat_path(cx, hir)?),
        None => None,
    };

    let kind = match path {
        Some(PatPath::Meta { meta }) => {
            let (args, type_match) = match type_match::tuple_match_for(cx, meta) {
                Some((args, type_match)) => (args, type_match),
                None => {
                    return Err(CompileError::expected_meta(
                        cx.span,
                        meta.info(cx.q.pool),
                        "tuple type",
                    ));
                }
            };

            if !(args == hir.items.len() || hir.is_open && args >= hir.items.len()) {
                return Err(cx.error(CompileErrorKind::UnsupportedArgumentCount {
                    meta: meta.info(cx.q.pool),
                    expected: args,
                    actual: hir.items.len(),
                }));
            }

            PatTupleKind::Typed { type_match }
        }
        Some(..) => {
            return Err(cx.error(CompileErrorKind::ExpectedTuple));
        }
        None => PatTupleKind::Anonymous,
    };

    let patterns = iter!(cx; hir.items, |hir| assemble_pat(cx, hir)?);

    Ok(cx.pat(PatKind::Tuple {
        kind,
        patterns,
        is_open: hir.is_open,
    }))
}

/// Assemble a pattern object.
#[instrument]
fn assemble_pat_object<'hir>(
    cx: &mut Ctxt<'_, 'hir>,
    hir: &hir::PatObject<'hir>,
) -> Result<Pat<'hir>> {
    let path = match hir.path {
        Some(hir) => Some(assemble_pat_path(cx, hir)?),
        None => None,
    };

    let mut keys_dup = HashMap::new();
    let mut keys = Vec::with_capacity(hir.bindings.len());

    for binding in hir.bindings {
        let key = match binding.key {
            hir::ObjectKey::Path(hir) => match assemble_pat_path(cx, hir)? {
                PatPath::Ident { ident } => ident.resolve(resolve_context!(cx.q))?.to_owned(),
                _ => {
                    return Err(cx.error(CompileErrorKind::UnsupportedPattern));
                }
            },
            hir::ObjectKey::LitStr(lit) => lit.resolve(resolve_context!(cx.q))?.into_owned(),
        };

        if let Some(existing) = keys_dup.insert(key.clone(), binding.span) {
            return Err(cx.error(CompileErrorKind::DuplicateObjectKey {
                existing,
                object: binding.span,
            }));
        }

        keys.push(key);
    }

    let kind = match path {
        Some(PatPath::Meta { meta }) => {
            let (st, type_match) = match type_match::struct_match_for(cx, meta) {
                Some((args, type_match)) => (args, type_match),
                None => {
                    return Err(CompileError::expected_meta(
                        cx.span,
                        meta.info(cx.q.pool),
                        "struct type",
                    ));
                }
            };

            let mut fields = st.fields.clone();

            for key in keys.iter() {
                if !fields.remove(key.as_str()) {
                    return Err(cx.error(CompileErrorKind::LitObjectNotField {
                        field: key.as_str().into(),
                        item: cx.q.pool.item(meta.item_meta.item).to_owned(),
                    }));
                }
            }

            if !hir.is_open && !fields.is_empty() {
                let mut fields = fields
                    .into_iter()
                    .map(Box::<str>::from)
                    .collect::<Box<[_]>>();
                fields.sort();

                return Err(cx.error(CompileErrorKind::PatternMissingFields {
                    item: cx.q.pool.item(meta.item_meta.item).to_owned(),
                    fields,
                }));
            }

            PatObjectKind::Typed {
                type_match,
                keys: iter!(cx; keys, |key| {
                    let hash = Hash::instance_fn_name(key.as_str());
                    let slot = cx.q.unit.new_static_string(cx.span, key.as_str())?;
                    (slot, hash)
                }),
            }
        }
        Some(..) => {
            return Err(cx.error(CompileErrorKind::ExpectedStruct));
        }
        None => {
            let slot =
                cx.q.unit
                    .new_static_object_keys_iter(cx.span, keys.iter().map(String::as_str))?;

            PatObjectKind::Anonymous {
                slot,
                is_open: hir.is_open,
            }
        }
    };

    let patterns = iter!(cx; hir.bindings, |binding| {
        if let Some(pat) = binding.pat {
            assemble_pat(cx, pat)?
        } else {
            assemble_object_key(cx, binding.key)?
        }
    });

    Ok(cx.pat(PatKind::Object { kind, patterns }))
}

#[instrument]
fn assemble_pat_path<'hir>(
    cx: &mut Ctxt<'_, 'hir>,
    hir: &hir::Path<'hir>,
) -> Result<PatPath<'hir>> {
    let span = hir.span();

    let named = cx.convert_path(hir)?;

    if let Some(meta) = cx.try_lookup_meta(span, named.item)? {
        return Ok(PatPath::Meta {
            meta: alloc!(cx; meta),
        });
    }

    if let Some(ident) = hir.try_as_ident() {
        return Ok(PatPath::Ident { ident });
    }

    Err(CompileError::new(
        span,
        CompileErrorKind::MissingItem {
            item: cx.q.pool.item(named.item).to_owned(),
        },
    ))
}

/// Assemble #[builtin] template!(...) macro.
#[instrument]
fn assemble_builtin_template<'hir>(
    cx: &mut Ctxt<'_, 'hir>,
    hir: &hir::BuiltInTemplate<'hir>,
) -> Result<Expr<'hir>> {
    // Template represents a single literal string.
    if hir
        .exprs
        .iter()
        .all(|hir| matches!(hir.kind, hir::ExprKind::Lit(ast::Lit::Str(..))))
    {
        let mut string = String::new();

        if hir.from_literal {
            cx.q.diagnostics
                .template_without_expansions(cx.source_id, cx.span, Some(cx.context()));
        }

        for hir in hir.exprs {
            if let hir::ExprKind::Lit(ast::Lit::Str(s)) = hir.kind {
                string += s.resolve(resolve_context!(cx.q))?.as_ref();
            }
        }

        let string = cx.arena.alloc_str(&string).map_err(arena_error(cx.span))?;
        return Ok(cx.expr(ExprKind::String { string }));
    }

    let exprs = iter!(cx; hir.exprs, |hir| assemble_expr_value(cx, hir)?);
    Ok(cx.expr(ExprKind::StringConcat { exprs }))
}

/// Assemble #[builtin] format!(...) macro.
#[instrument]
fn assemble_builtin_format<'hir>(
    cx: &mut Ctxt<'_, 'hir>,
    hir: &hir::BuiltInFormat<'hir>,
) -> Result<Expr<'hir>> {
    use crate::runtime::format;

    let fill = if let Some((_, fill)) = &hir.fill {
        *fill
    } else {
        ' '
    };

    let align = if let Some((_, align)) = &hir.align {
        *align
    } else {
        format::Alignment::default()
    };

    let flags = if let Some((_, flags)) = &hir.flags {
        *flags
    } else {
        format::Flags::default()
    };

    let width = if let Some((_, width)) = &hir.width {
        *width
    } else {
        None
    };

    let precision = if let Some((_, precision)) = &hir.precision {
        *precision
    } else {
        None
    };

    let format_type = if let Some((_, format_type)) = &hir.format_type {
        *format_type
    } else {
        format::Type::default()
    };

    let spec = format::FormatSpec::new(flags, fill, align, width, precision, format_type);
    let expr = assemble_expr_value(cx, hir.value)?;

    Ok(cx.expr(ExprKind::Format {
        spec: alloc!(cx; spec),
        expr: alloc!(cx; expr),
    }))
}

/// Assemble a closure.
#[instrument]
fn assemble_expr_closure<'hir>(
    cx: &mut Ctxt<'_, 'hir>,
    hir: &hir::ExprClosure<'hir>,
) -> Result<Expr<'hir>> {
    let item = cx.q.item_for((cx.span, hir.id))?;
    let hash = cx.q.pool.item_type_hash(item.item);

    let meta = match cx.q.query_meta(cx.span, item.item, Default::default())? {
        Some(meta) => meta,
        None => {
            return Err(cx.error(CompileErrorKind::MissingItem {
                item: cx.q.pool.item(item.item).to_owned(),
            }))
        }
    };

    let (captures, do_move) = match &meta.kind {
        PrivMetaKind::Closure {
            captures, do_move, ..
        } => (captures.as_ref(), *do_move),
        _ => {
            return Err(cx.error(CompileErrorKind::ExpectedMeta {
                meta: meta.info(cx.q.pool),
                expected: "a closure",
            }));
        }
    };

    tracing::trace!("captures: {} => {:?}", item.item, captures);

    let kind = if captures.is_empty() {
        ExprKind::Function { hash }
    } else {
        // Construct a closure environment.
        let captures = iter!(cx; captures, |capture| {
            if do_move {
                todo!()
            }

            let (binding, address) = cx.scopes.lookup(
                Location::new(cx.source_id, cx.span),
                cx.scope,
                cx.q.visitor,
                &capture.ident,
            )?;

            cx.expr(ExprKind::Address {
                binding: Some(binding),
                address,
            })
        });

        ExprKind::Closure { hash, captures }
    };

    Ok(cx.expr(kind))
}

/// An object expression
#[instrument]
fn assemble_expr_object<'hir>(
    cx: &mut Ctxt<'_, 'hir>,
    hir: &hir::ExprObject<'hir>,
) -> Result<Expr<'hir>> {
    let mut keys = Vec::new();
    let mut keys_dup = HashMap::new();

    let mut exprs = cx
        .arena
        .alloc_iter(hir.assignments.len())
        .map_err(arena_error(cx.span))?;

    for assignment in hir.assignments {
        cx.with_span(assignment.span, |cx| {
            exprs
                .write(match assignment.assign {
                    Some(hir) => assemble_expr_value(cx, hir)?,
                    None => match assignment.key {
                        hir::ObjectKey::Path(hir) => assemble_expr_path(cx, hir, Value)?,
                        _ => {
                            return Err(cx.error(CompileErrorKind::InvalidStructKey));
                        }
                    },
                })
                .map_err(arena_slice_write_error(cx.span))?;

            let key = assignment.key.resolve(resolve_context!(cx.q))?;
            keys.push((key.clone().into_owned(), cx.span));

            if let Some(existing) = keys_dup.insert(key.into_owned(), cx.span) {
                return Err(cx.error(CompileErrorKind::DuplicateObjectKey {
                    existing,
                    object: cx.span,
                }));
            }

            Ok(())
        })?;
    }

    let exprs = exprs.finish();

    let expr = match hir.path {
        Some(hir) => assemble_expr_path(cx, hir, Type)?.map(|kind| {
            let kind = match kind {
                ExprKind::Meta { meta, .. } => {
                    let item = cx.q.pool.item(meta.item_meta.item);
                    let hash = cx.q.pool.item_type_hash(meta.item_meta.item);

                    match &meta.kind {
                        PrivMetaKind::Struct {
                            variant: PrivVariantMeta::Unit,
                            ..
                        } => {
                            check_object_fields(cx, &HashSet::new(), &keys, item)?;
                            ExprStructKind::Unit { hash }
                        }
                        PrivMetaKind::Struct {
                            variant: PrivVariantMeta::Struct(st),
                            ..
                        } => {
                            check_object_fields(cx, &st.fields, &keys, item)?;
                            let slot = cx.q.unit.new_static_object_keys_iter(
                                cx.span,
                                keys.iter().map(|(s, _)| s.as_str()),
                            )?;
                            ExprStructKind::Struct { hash, slot }
                        }
                        PrivMetaKind::Variant {
                            variant: PrivVariantMeta::Struct(st),
                            ..
                        } => {
                            check_object_fields(cx, &st.fields, &keys, item)?;
                            let slot = cx.q.unit.new_static_object_keys_iter(
                                cx.span,
                                keys.iter().map(|(s, _)| s.as_str()),
                            )?;
                            ExprStructKind::StructVariant { hash, slot }
                        }
                        _ => {
                            return Err(cx.error(CompileErrorKind::MetaNotStruct {
                                meta: meta.info(cx.q.pool),
                            }));
                        }
                    }
                }
                _ => {
                    return Err(cx.error(CompileErrorKind::NotStruct));
                }
            };

            Ok(ExprKind::Struct { kind, exprs })
        })?,
        None => {
            let slot =
                cx.q.unit
                    .new_static_object_keys_iter(cx.span, keys.iter().map(|(s, _)| s.as_str()))?;

            cx.expr(ExprKind::Struct {
                kind: ExprStructKind::Anonymous { slot },
                exprs,
            })
        }
    };

    return Ok(expr);

    fn check_object_fields<S>(
        cx: &Ctxt<'_, '_>,
        fields: &HashSet<Box<str>>,
        keys: &[(S, Span)],
        item: &Item,
    ) -> Result<()>
    where
        S: AsRef<str>,
    {
        let mut fields = fields.clone();

        for (field, span) in keys {
            if !fields.remove(field.as_ref()) {
                return Err(CompileError::new(
                    span,
                    CompileErrorKind::LitObjectNotField {
                        field: field.as_ref().into(),
                        item: item.to_owned(),
                    },
                ));
            }
        }

        if let Some(field) = fields.into_iter().next() {
            return Err(cx.error(CompileErrorKind::LitObjectMissingField {
                field,
                item: item.to_owned(),
            }));
        }

        Ok(())
    }
}

/// Assemble a tuple expression.
#[instrument]
fn assemble_expr_tuple<'hir>(
    cx: &mut Ctxt<'_, 'hir>,
    hir: &hir::ExprSeq<'hir>,
) -> Result<Expr<'hir>> {
    let items = iter!(cx; hir.items, |hir| assemble_expr_value(cx, hir)?);
    Ok(cx.expr(ExprKind::Tuple { items }))
}

/// An assignment expression.
#[instrument]
fn assemble_expr_assign<'hir>(
    cx: &mut Ctxt<'_, 'hir>,
    lhs: &'hir hir::Expr<'hir>,
    rhs: &'hir hir::Expr<'hir>,
) -> Result<Expr<'hir>> {
    let rhs = assemble_expr_value(cx, rhs)?;

    assemble_expr_value(cx, lhs)?.map(|kind| match kind {
        ExprKind::Address {
            binding, address, ..
        } => {
            // Custom handling of rhs mapping to perform an immediate reassignment.
            let rhs = rhs.map(|kind| match (kind, binding) {
                (ExprKind::Address { address, .. }, Some(binding)) => {
                    if cx.scopes.assign(cx.span, cx.scope, binding, address)? {
                        Ok(ExprKind::Empty)
                    } else {
                        Ok(kind)
                    }
                }
                (kind, _) => Ok(kind),
            })?;

            Ok(ExprKind::Assign {
                address,
                rhs: alloc!(cx; rhs),
            })
        }
        ExprKind::StructFieldAccess { lhs, slot, .. } => Ok(ExprKind::AssignStructField {
            lhs,
            slot,
            rhs: alloc!(cx; rhs),
        }),
        ExprKind::TupleFieldAccess { lhs, index } => Ok(ExprKind::AssignTupleField {
            lhs,
            index,
            rhs: alloc!(cx; rhs),
        }),
        _ => Err(cx.error(CompileErrorKind::UnsupportedAssignExpr)),
    })
}

/// Decode a field access expression.
#[instrument]
fn assemble_expr_field_access<'hir>(
    cx: &mut Ctxt<'_, 'hir>,
    hir: &hir::ExprFieldAccess<'hir>,
) -> Result<Expr<'hir>> {
    match hir.expr_field {
        hir::ExprField::Path(path) => {
            if let Some(ident) = path.try_as_ident() {
                let n = ident.resolve(resolve_context!(cx.q))?;
                let slot = cx.q.unit.new_static_string(ident.span(), n)?;
                let hash = Hash::instance_fn_name(n);

                let lhs = assemble_expr_value(cx, hir.expr)?;

                return Ok(cx.expr(ExprKind::StructFieldAccess {
                    lhs: alloc!(cx; lhs),
                    slot,
                    hash,
                }));
            }

            if let Some((ident, generics)) = path.try_as_ident_generics() {
                let n = ident.resolve(resolve_context!(cx.q))?;
                let hash = Hash::instance_fn_name(n.as_ref());

                let lhs = assemble_expr_value(cx, hir.expr)?;

                return Ok(cx.expr(ExprKind::StructFieldAccessGeneric {
                    lhs: alloc!(cx; lhs),
                    hash,
                    generics,
                }));
            }
        }
        hir::ExprField::LitNumber(field) => {
            let span = field.span();

            let number = field.resolve(resolve_context!(cx.q))?;
            let index = number.as_tuple_index().ok_or_else(|| {
                CompileError::new(span, CompileErrorKind::UnsupportedTupleIndex { number })
            })?;

            let lhs = assemble_expr_value(cx, hir.expr)?;

            return Ok(cx.expr(ExprKind::TupleFieldAccess {
                lhs: alloc!(cx; lhs),
                index,
            }));
        }
    }

    Err(cx.error(CompileErrorKind::BadFieldAccess))
}

/// Assemble a loop expression.
#[instrument]
fn assemble_expr_loop<'hir>(
    cx: &mut Ctxt<'_, 'hir>,
    hir: &'hir hir::ExprLoop,
) -> Result<Expr<'hir>> {
    Ok(cx.expr(ExprKind::Loop { hir }))
}

/// Assembling of a binary expression.
#[instrument]
fn assemble_expr_path<'hir>(
    cx: &mut Ctxt<'_, 'hir>,
    hir: &'hir hir::Path,
    needs: Needs,
) -> Result<Expr<'hir>> {
    let span = hir.span();
    let loc = Location::new(cx.source_id, span);

    if let Some(ast::PathKind::SelfValue) = hir.as_kind() {
        let (binding, address) = cx.scopes.lookup(loc, cx.scope, cx.q.visitor, SELF)?;

        return Ok(cx.expr(ExprKind::Address {
            binding: Some(binding),
            address,
        }));
    }

    let named = cx.convert_path(hir)?;

    if let Value = needs {
        if let Some(local) = named.as_local() {
            let local = local.resolve(resolve_context!(cx.q))?;

            if let Some((binding, address)) =
                cx.scopes.try_lookup(loc, cx.scope, cx.q.visitor, local)?
            {
                return Ok(cx.expr(ExprKind::Address {
                    binding: Some(binding),
                    address,
                }));
            }
        }
    }

    if let Some(meta) = cx.try_lookup_meta(span, named.item)? {
        return Ok(cx.expr(ExprKind::Meta {
            meta: alloc!(cx; meta),
            needs,
            named: alloc!(cx; named),
        }));
    }

    if let (Value, Some(local)) = (needs, named.as_local()) {
        let local = local.resolve(resolve_context!(cx.q))?;

        // light heuristics, treat it as a type error in case the first
        // character is uppercase.
        if !local.starts_with(char::is_uppercase) {
            return Err(CompileError::new(
                span,
                CompileErrorKind::MissingLocal {
                    name: local.to_owned(),
                },
            ));
        }
    }

    Err(CompileError::new(
        span,
        CompileErrorKind::MissingItem {
            item: cx.q.pool.item(named.item).to_owned(),
        },
    ))
}

/// Assemble a binary expression.
#[instrument]
fn assemble_expr_binary<'hir>(
    cx: &mut Ctxt<'_, 'hir>,
    hir: &hir::ExprBinary<'hir>,
) -> Result<Expr<'hir>> {
    let lhs = alloc!(cx; assemble_expr_value(cx, hir.lhs)?);
    let rhs = alloc!(cx; assemble_expr(cx, hir.rhs, rhs_needs_of(&hir.op))?);

    return Ok(cx.expr(ExprKind::Binary {
        lhs,
        op: hir.op,
        rhs,
    }));

    /// Get the need of the right-hand side operator from the type of the operator.
    fn rhs_needs_of(op: &ast::BinOp) -> Needs {
        match op {
            ast::BinOp::Is(..) | ast::BinOp::IsNot(..) => Type,
            _ => Value,
        }
    }
}

/// Assemble a let expression.
#[instrument]
fn assemble_expr_let<'hir>(
    cx: &mut Ctxt<'_, 'hir>,
    hir: &hir::ExprLet<'hir>,
) -> Result<Expr<'hir>> {
    let pat = alloc!(cx; assemble_pat(cx, hir.pat)?);
    let expr = alloc!(cx; assemble_expr_value(cx, hir.expr)?);
    Ok(cx.expr(ExprKind::Let { pat, expr }))
}

/// Assemble an if statement.
#[instrument]
fn assemble_expr_if<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: &hir::ExprIf<'hir>) -> Result<Expr<'hir>> {
    let count = 1 + hir.expr_else_ifs.len() + if hir.expr_else.is_some() { 1 } else { 0 };

    let mut conditions = Conditions::new(cx, count)?;

    conditions.add_branch(cx, Some(hir.condition), hir.block)?;

    for branch in hir.expr_else_ifs {
        conditions.add_branch(cx, Some(branch.condition), branch.block)?;
    }

    if let Some(branch) = hir.expr_else {
        conditions.add_branch(cx, None, branch.block)?;
    }

    conditions.assemble(cx)
}

/// Assemble a unary expression.
#[instrument]
fn assemble_expr_match<'hir>(
    cx: &mut Ctxt<'_, 'hir>,
    hir: &hir::ExprMatch<'hir>,
) -> Result<Expr<'hir>> {
    let mut matches = Matches::new(cx, hir.branches.len(), hir.expr)?;

    for branch in hir.branches {
        cx.with_span(branch.span, |cx| {
            matches.add_branch(cx, Some(branch.pat), branch.condition, branch.body)?;
            Ok(())
        })?;
    }

    matches.assemble(cx)
}

/// Assemble a unary expression.
#[instrument]
fn assemble_expr_unary<'hir>(
    cx: &mut Ctxt<'_, 'hir>,
    hir: &hir::ExprUnary<'hir>,
) -> Result<Expr<'hir>> {
    // NB: special unary expressions.
    if let ast::UnOp::BorrowRef { .. } = hir.op {
        return Err(cx.error(CompileErrorKind::UnsupportedRef));
    }

    if let (ast::UnOp::Neg(..), hir::ExprKind::Lit(ast::Lit::Number(n))) = (hir.op, hir.expr.kind) {
        match n.resolve(resolve_context!(cx.q))? {
            ast::Number::Float(n) => {
                return Ok(cx.expr(ExprKind::Store {
                    value: InstValue::Float(-n),
                }));
            }
            ast::Number::Integer(int) => {
                let n = match int.neg().to_i64() {
                    Some(n) => n,
                    None => {
                        return Err(cx.error(ParseErrorKind::BadNumberOutOfBounds));
                    }
                };

                return Ok(cx.expr(ExprKind::Store {
                    value: InstValue::Integer(n),
                }));
            }
        }
    }

    let op = match hir.op {
        ast::UnOp::Not(..) => ExprUnOp::Not,
        ast::UnOp::Neg(..) => ExprUnOp::Neg,
        op => {
            return Err(cx.error(CompileErrorKind::UnsupportedUnaryOp { op }));
        }
    };

    let expr = assemble_expr_value(cx, hir.expr)?;

    Ok(cx.expr(ExprKind::Unary {
        op,
        expr: alloc!(cx; expr),
    }))
}

/// Assemble an index expression
#[instrument]
fn assemble_expr_index<'hir>(
    cx: &mut Ctxt<'_, 'hir>,
    hir: &hir::ExprIndex<'hir>,
) -> Result<Expr<'hir>> {
    let target = alloc!(cx; assemble_expr_value(cx, hir.target)?);
    let index = alloc!(cx; assemble_expr_value(cx, hir.index)?);
    Ok(cx.expr(ExprKind::Index { target, index }))
}

/// Assemble a break expression
#[instrument]
fn assemble_expr_break<'hir>(
    cx: &mut Ctxt<'_, 'hir>,
    hir: &hir::ExprBreakValue<'hir>,
) -> Result<Expr<'hir>> {
    let kind = ExprKind::Break {
        value: alloc!(cx; match *hir {
            hir::ExprBreakValue::None => {
                ExprBreakValue::None
            },
            hir::ExprBreakValue::Expr(hir) => ExprBreakValue::Expr(alloc!(cx; assemble_expr_value(cx, hir)?)),
            hir::ExprBreakValue::Label(ast) => ExprBreakValue::Label(cx.scopes.binding_name(ast.resolve(resolve_context!(cx.q))?)),
        }),
    };

    Ok(cx.expr(kind))
}

/// Assemble a continue expression
#[instrument]
fn assemble_expr_continue<'hir>(
    cx: &mut Ctxt<'_, 'hir>,
    ast: Option<&'hir ast::Label>,
) -> Result<Expr<'hir>> {
    let label = match ast {
        Some(ast) => Some(cx.scopes.binding_name(ast.resolve(resolve_context!(cx.q))?)),
        None => None,
    };

    Ok(cx.expr(ExprKind::Continue { label }))
}

/// Assemble a yield expression.
#[instrument]
fn assemble_expr_yield<'hir>(
    cx: &mut Ctxt<'_, 'hir>,
    hir: Option<&'hir hir::Expr>,
) -> Result<Expr<'hir>> {
    let expr = option!(cx; hir, |hir| assemble_expr_value(cx, hir)?);
    Ok(cx.expr(ExprKind::Yield { expr }))
}

/// Assemble an await expression.
#[instrument]
fn assemble_expr_await<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: &'hir hir::Expr) -> Result<Expr<'hir>> {
    let expr = alloc!(cx; assemble_expr_value(cx, hir)?);
    Ok(cx.expr(ExprKind::Await { expr }))
}

/// Assemble a try expression.
#[instrument]
fn assemble_expr_try<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: &'hir hir::Expr) -> Result<Expr<'hir>> {
    let expr = alloc!(cx; assemble_expr_value(cx, hir)?);
    Ok(cx.expr(ExprKind::Try { expr }))
}

/// Assemble a return expression.
#[instrument]
fn assemble_expr_return<'hir>(
    cx: &mut Ctxt<'_, 'hir>,
    hir: Option<&'hir hir::Expr<'hir>>,
) -> Result<Expr<'hir>> {
    let kind = match hir {
        Some(hir) => ExprKind::Return {
            expr: alloc!(cx; assemble_expr_value(cx, hir)?),
        },
        None => ExprKind::Return {
            expr: alloc!(cx; cx.expr(ExprKind::Store { value: InstValue::Unit })),
        },
    };

    Ok(cx.expr(kind))
}

/// Construct a literal value.
#[instrument]
fn assemble_expr_lit<'hir>(cx: &mut Ctxt<'_, 'hir>, ast: &'hir ast::Lit) -> Result<Expr<'hir>> {
    cx.with_span(ast.span(), |cx| {
        let expr = match ast {
            ast::Lit::Bool(lit) => cx.expr(ExprKind::Store {
                value: InstValue::Bool(lit.value),
            }),
            ast::Lit::Char(lit) => {
                let ch = lit.resolve(resolve_context!(cx.q))?;
                cx.expr(ExprKind::Store {
                    value: InstValue::Char(ch),
                })
            }
            ast::Lit::ByteStr(lit) => {
                let b = lit.resolve(resolve_context!(cx.q))?;
                cx.expr(ExprKind::Bytes {
                    bytes: cx
                        .arena
                        .alloc_bytes(b.as_ref())
                        .map_err(arena_error(cx.span))?,
                })
            }
            ast::Lit::Str(lit) => {
                let b = lit.resolve(resolve_context!(cx.q))?;
                cx.expr(ExprKind::String {
                    string: cx
                        .arena
                        .alloc_str(b.as_ref())
                        .map_err(arena_error(cx.span))?,
                })
            }
            ast::Lit::Byte(lit) => {
                let b = lit.resolve(resolve_context!(cx.q))?;
                cx.expr(ExprKind::Store {
                    value: InstValue::Byte(b),
                })
            }
            ast::Lit::Number(number) => assemble_lit_number(cx, number)?,
        };

        Ok(expr)
    })
}

/// Convert a literal number into an expression kind.
#[instrument]
fn assemble_lit_number<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: &ast::LitNumber) -> Result<Expr<'hir>> {
    let number = hir.resolve(resolve_context!(cx.q))?;

    match number {
        ast::Number::Float(float) => Ok(cx.expr(ExprKind::Store {
            value: InstValue::Float(float),
        })),
        ast::Number::Integer(integer) => {
            let n = match integer.to_i64() {
                Some(n) => n,
                None => {
                    return Err(CompileError::new(
                        hir.span,
                        ParseErrorKind::BadNumberOutOfBounds,
                    ));
                }
            };

            Ok(cx.expr(ExprKind::Store {
                value: InstValue::Integer(n),
            }))
        }
    }
}

/// Assemble a call expression.
#[instrument]
fn assemble_expr_call<'hir>(
    cx: &mut Ctxt<'_, 'hir>,
    hir: &hir::ExprCall<'hir>,
) -> Result<Expr<'hir>> {
    assemble_expr(cx, hir.expr, Value)?.map(|kind| match kind {
        ExprKind::Address { address, .. } => Ok(ExprKind::CallAddress {
            address,
            args: iter!(cx; hir.args, |hir| assemble_expr_value(cx, hir)?),
        }),
        ExprKind::Meta { meta, named, .. } => {
            match &meta.kind {
                PrivMetaKind::Struct {
                    variant: PrivVariantMeta::Unit,
                    ..
                }
                | PrivMetaKind::Variant {
                    variant: PrivVariantMeta::Unit,
                    ..
                } => {
                    named.assert_not_generic()?;

                    if !hir.args.is_empty() {
                        return Err(cx.error(CompileErrorKind::UnsupportedArgumentCount {
                            meta: meta.info(cx.q.pool),
                            expected: 0,
                            actual: hir.args.len(),
                        }));
                    }
                }
                PrivMetaKind::Struct {
                    variant: PrivVariantMeta::Tuple(tuple),
                    ..
                }
                | PrivMetaKind::Variant {
                    variant: PrivVariantMeta::Tuple(tuple),
                    ..
                } => {
                    named.assert_not_generic()?;

                    if tuple.args != hir.args.len() {
                        return Err(cx.error(CompileErrorKind::UnsupportedArgumentCount {
                            meta: meta.info(cx.q.pool),
                            expected: tuple.args,
                            actual: hir.args.len(),
                        }));
                    }

                    if tuple.args == 0 {
                        let tuple = hir.expr.span();

                        cx.q.diagnostics.remove_tuple_call_parens(
                            cx.source_id,
                            cx.span,
                            tuple,
                            Some(cx.context()),
                        );
                    }
                }
                PrivMetaKind::Function { .. } => (),
                PrivMetaKind::ConstFn { id, .. } => {
                    named.assert_not_generic()?;

                    let from = cx.q.item_for((cx.span, hir.id))?;
                    let const_fn = cx.q.const_fn_for((cx.span, *id))?;
                    let value = cx.call_const_fn(meta, &from, &const_fn, hir.args)?;
                    // NB: It is valid to coerce an expr_const into a kind
                    // because it belongs to the constant scope which requires
                    // no cleanup.
                    return Ok(assemble_expr_const(cx, &value)?.into_kind()?);
                }
                _ => {
                    return Err(cx.error(CompileErrorKind::ExpectedMeta {
                        meta: meta.info(cx.q.pool),
                        expected: "something that can be called as a function",
                    }));
                }
            };

            let hash = cx.q.pool.item_type_hash(meta.item_meta.item);

            let hash = if let Some((span, generics)) = named.generics {
                let parameters = cx.with_span(span, |cx| generics_parameters(cx, generics))?;
                hash.with_parameters(parameters)
            } else {
                hash
            };

            Ok(ExprKind::CallHash {
                hash,
                args: iter!(cx; hir.args, |hir| assemble_expr_value(cx, hir)?),
            })
        }
        ExprKind::StructFieldAccess { lhs, hash, .. } => Ok(ExprKind::CallInstance {
            lhs,
            hash,
            args: iter!(cx; hir.args, |hir| assemble_expr_value(cx, hir)?),
        }),
        ExprKind::StructFieldAccessGeneric {
            lhs,
            hash,
            generics,
            ..
        } => {
            let hash = if let Some((span, generics)) = generics {
                let parameters = cx.with_span(span, |cx| generics_parameters(cx, generics))?;
                hash.with_parameters(parameters)
            } else {
                hash
            };

            Ok(ExprKind::CallInstance {
                lhs,
                hash,
                args: iter!(cx; hir.args, |hir| assemble_expr_value(cx, hir)?),
            })
        }
        kind => Ok(ExprKind::CallExpr {
            expr: alloc!(cx; cx.expr(kind)),
            args: iter!(cx; hir.args, |hir| assemble_expr_value(cx, hir)?),
        }),
    })
}

/// Compile a constant value into an expression.
#[instrument]
fn assemble_expr_const<'hir>(cx: &mut Ctxt<'_, 'hir>, value: &ConstValue) -> Result<Expr<'hir>> {
    let kind = match *value {
        ConstValue::Unit => ExprKind::Store {
            value: InstValue::Unit,
        },
        ConstValue::Byte(b) => ExprKind::Store {
            value: InstValue::Byte(b),
        },
        ConstValue::Char(ch) => ExprKind::Store {
            value: InstValue::Char(ch),
        },
        ConstValue::Integer(n) => {
            let n = match n.to_i64() {
                Some(n) => n,
                None => {
                    return Err(cx.error(ParseErrorKind::BadNumberOutOfBounds));
                }
            };

            ExprKind::Store {
                value: InstValue::Integer(n),
            }
        }
        ConstValue::Float(n) => ExprKind::Store {
            value: InstValue::Float(n),
        },
        ConstValue::Bool(b) => ExprKind::Store {
            value: InstValue::Bool(b),
        },
        ConstValue::String(ref s) => ExprKind::String {
            string: cx
                .arena
                .alloc_str(s.as_str())
                .map_err(arena_error(cx.span))?,
        },
        ConstValue::StaticString(ref s) => ExprKind::String {
            string: cx
                .arena
                .alloc_str(s.as_ref())
                .map_err(arena_error(cx.span))?,
        },
        ConstValue::Bytes(ref b) => ExprKind::Bytes {
            bytes: cx
                .arena
                .alloc_bytes(b.as_ref())
                .map_err(arena_error(cx.span))?,
        },
        ConstValue::Option(ref option) => ExprKind::Option {
            value: option!(cx; option.as_deref(), |value| assemble_expr_const(cx, value)?),
        },
        ConstValue::Vec(ref vec) => {
            let args = iter!(cx; vec, |value| assemble_expr_const(cx, &value)?);
            ExprKind::Vec { items: args }
        }
        ConstValue::Tuple(ref tuple) => {
            let args = iter!(cx; tuple.iter(), |value| assemble_expr_const(cx, value)?);
            ExprKind::Tuple { items: args }
        }
        ConstValue::Object(ref object) => {
            let mut entries = object.iter().collect::<vec::Vec<_>>();
            entries.sort_by_key(|k| k.0);

            let exprs = iter!(cx; entries.iter(), |(_, value)| assemble_expr_const(cx, value)?);

            let slot =
                cx.q.unit
                    .new_static_object_keys_iter(cx.span, entries.iter().map(|e| e.0))?;

            ExprKind::Struct {
                kind: ExprStructKind::Anonymous { slot },
                exprs,
            }
        }
    };

    Ok(cx.expr(kind))
}

/// Assemble a vector expression.
#[instrument]
fn assemble_expr_vec<'hir>(
    cx: &mut Ctxt<'_, 'hir>,
    hir: &hir::ExprSeq<'hir>,
) -> Result<Expr<'hir>> {
    let items = iter!(cx; hir.items, |hir| assemble_expr_value(cx, hir)?);
    Ok(cx.expr(ExprKind::Vec { items }))
}

/// Assemble a range expression.
#[instrument]
fn assemble_expr_range<'hir>(
    cx: &mut Ctxt<'_, 'hir>,
    hir: &hir::ExprRange<'hir>,
) -> Result<Expr<'hir>> {
    let limits = match hir.limits {
        hir::ExprRangeLimits::HalfOpen => InstRangeLimits::HalfOpen,
        hir::ExprRangeLimits::Closed => InstRangeLimits::Closed,
    };

    let from = ExprKind::Option {
        value: match hir.from {
            Some(hir) => Some(alloc!(cx; assemble_expr_value(cx, hir)?)),
            None => None,
        },
    };

    let to = ExprKind::Option {
        value: match hir.to {
            Some(hir) => Some(alloc!(cx; assemble_expr_value(cx, hir)?)),
            None => None,
        },
    };

    Ok(cx.expr(ExprKind::Range {
        from: alloc!(cx; cx.expr(from)),
        limits,
        to: alloc!(cx; cx.expr(to)),
    }))
}

/// Assemble a macro call.
#[instrument]
fn assemble_macro_call<'hir>(
    cx: &mut Ctxt<'_, 'hir>,
    hir: &hir::MacroCall<'hir>,
) -> Result<Expr<'hir>> {
    let expr = match hir {
        hir::MacroCall::Template(hir) => assemble_builtin_template(cx, hir)?,
        hir::MacroCall::Format(hir) => assemble_builtin_format(cx, hir)?,
        hir::MacroCall::File(hir) => {
            let current = hir.value.resolve(resolve_context!(cx.q))?;
            let string = cx
                .arena
                .alloc_str(current.as_ref())
                .map_err(arena_error(cx.span))?;
            cx.expr(ExprKind::String { string })
        }
        hir::MacroCall::Line(hir) => assemble_lit_number(cx, &hir.value)?,
    };

    Ok(expr)
}

#[instrument]
fn generics_parameters<'hir>(
    cx: &mut Ctxt<'_, 'hir>,
    generics: &'hir [hir::Expr<'hir>],
) -> Result<Hash> {
    let mut parameters = ParametersBuilder::new();

    for expr in generics {
        let path = match expr.kind {
            hir::ExprKind::Path(path) => path,
            _ => {
                return Err(cx.error(CompileErrorKind::UnsupportedGenerics));
            }
        };

        let named = cx.convert_path(path)?;
        named.assert_not_generic()?;

        let meta = cx.lookup_meta(expr.span(), named.item)?;

        let hash = match meta.kind {
            PrivMetaKind::Unknown { type_hash, .. } => type_hash,
            PrivMetaKind::Struct { type_hash, .. } => type_hash,
            PrivMetaKind::Enum { type_hash, .. } => type_hash,
            _ => {
                return Err(cx.error(CompileErrorKind::UnsupportedGenerics));
            }
        };

        parameters.add(hash);
    }

    Ok(parameters.finish())
}

fn arena_error(span: Span) -> impl FnOnce(ArenaAllocError) -> CompileError {
    move |e| {
        CompileError::new(
            span,
            CompileErrorKind::ArenaAllocError {
                requested: e.requested,
            },
        )
    }
}

fn arena_slice_write_error(span: Span) -> impl FnOnce(ArenaWriteSliceOutOfBounds) -> CompileError {
    move |e| {
        CompileError::new(
            span,
            CompileErrorKind::ArenaWriteSliceOutOfBounds { index: e.index },
        )
    }
}
