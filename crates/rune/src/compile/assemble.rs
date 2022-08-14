use std::borrow::Cow;
use std::fmt;
use std::mem;
use std::num::NonZeroUsize;
use std::ops::Neg as _;
use std::vec;

use num::ToPrimitive;
use rune_macros::__instrument_hir as instrument;

use crate::arena::{Arena, ArenaAllocError, ArenaWriteSliceOutOfBounds};
use crate::ast::{self, Span, Spanned};
use crate::collections::{HashMap, HashSet};
use crate::compile::UnitBuilder;
use crate::compile::{
    ir, Assembly, AssemblyAddress, CaptureMeta, CompileError, CompileErrorKind, CompileVisitor,
    IrBudget, IrCompiler, IrInterpreter, Item, ItemId, ItemMeta, Label, Location, Options,
    PrivMeta, PrivMetaKind, PrivStructMeta, PrivVariantMeta,
};
use crate::hash::ParametersBuilder;
use crate::hir;
use crate::parse::{ParseErrorKind, Resolve};
use crate::query::{Named, Query, QueryConstFn, Used};
use crate::runtime::{
    Address, AssemblyInst as Inst, ConstValue, InstAssignOp, InstOp, InstRangeLimits, InstTarget,
    InstValue, InstVariant, PanicReason, TypeCheck,
};
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

macro_rules! str {
    ($cx:expr; $expr:expr) => {
        $cx.arena.alloc_str($expr).map_err(arena_error($cx.span))?
    };
}

/// The needs of an expression.
#[derive(Debug, Clone, Copy)]
enum Needs {
    Value,
    Type,
}

/// `self` variable.
const SELF: &str = "self";

type Result<T> = std::result::Result<T, CompileError>;

/// An address which is associated with an array.
struct ArrayAddress {
    index: usize,
    count: usize,
}

impl ArrayAddress {
    const fn new(index: usize, count: usize) -> Self {
        Self { index, count }
    }

    /// Coerce into assembly address.
    fn address(&self) -> AssemblyAddress {
        AssemblyAddress::Array(self.index)
    }

    /// Get the number of elements in the allocated address.
    fn count(&self) -> usize {
        self.count
    }

    /// Free an arrey address.
    fn free<'hir>(self, cx: &mut Ctxt<'_, 'hir>) -> Result<()> {
        cx.allocator.free_array(cx.span, self.count)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Default)]
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
    q: Query<'a>,
    /// The assembly we are generating.
    asm: &'a mut Assembly,
    /// Enabled optimizations.
    // TODO: remove this
    #[allow(unused)]
    options: &'a Options,
    /// Arena to use for allocations.
    arena: &'hir Arena,
    /// Source being processed.
    source_id: SourceId,
    /// Context for which to emit warnings.
    span: Span,
    /// Scope associated with the context.
    scope: ScopeId,
    /// Scopes declared in context.
    scopes: Scopes<'hir>,
    /// State of code generation.
    state: CtxtState,
    /// A memory allocator.
    allocator: Allocator,
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
            allocator: Allocator::new(),
        }
    }

    /// Access the scopes built by the context.
    pub(crate) fn into_allocator(self) -> Allocator {
        self.allocator
    }

    /// Declare a variable.
    fn declare(&mut self, name: Name, address: Slot) -> Result<Option<Slot>> {
        self.scopes.declare(self.span, self.scope, name, address)
    }

    /// Insert a expression.
    fn insert_expr(&mut self, kind: ExprKind<'hir>) -> Slot {
        self.scopes.insert_expr(self.span, kind)
    }

    /// Insert a expression which during constrution has access to the address
    /// being constructed.
    fn insert_expr_with<T>(&mut self, builder: T) -> Slot
    where
        T: FnOnce(Slot) -> ExprKind<'hir>,
    {
        self.scopes.insert_expr_with(self.span, builder)
    }

    /// Insert a pattern.
    fn insert_pat(&mut self, kind: PatKind<'hir>, outputs: &'hir [Slot]) -> Slot {
        self.scopes.insert_pat(self.span, kind, outputs)
    }

    /// Insert a bound pattern.
    fn bound_pat(&mut self, kind: BoundPatKind<'hir>) -> BoundPat<'hir> {
        BoundPat::new(self.span, kind)
    }

    /// Access the meta for the given language item.
    #[tracing::instrument(skip_all)]
    fn try_lookup_meta(&mut self, span: Span, item: ItemId) -> Result<Option<PrivMeta>> {
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
    fn lookup_meta(&mut self, spanned: Span, item: ItemId) -> Result<PrivMeta> {
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
    fn convert_path(&mut self, path: &hir::Path<'hir>) -> Result<Named<'hir>> {
        self.q.convert_path(self.context, path)
    }

    /// Get the latest relevant warning context.
    fn context(&self) -> Span {
        self.span
    }

    /// Calling a constant function by id and return the resuling value.
    fn call_const_fn(
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

    /// Construct a compile error associated with the current scope.
    fn error<K>(&self, kind: K) -> CompileError
    where
        CompileErrorKind: From<K>,
    {
        CompileError::new(self.span, kind)
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

    /// Construct a new label.
    fn new_label(&mut self, label: &'static str) -> Label {
        self.asm.new_label(label)
    }

    /// Label the current position.
    fn label(&mut self, label: Label) -> Result<()> {
        if !matches!(self.state, CtxtState::Unreachable { .. }) {
            self.asm.label(self.span, label)?;
        }

        Ok(())
    }
}

/// Assemble an async block.
#[instrument]
pub(crate) fn closure_from_block<'hir>(
    cx: &mut Ctxt<'_, 'hir>,
    hir: &hir::Block<'hir>,
    captures: &[CaptureMeta],
) -> Result<()> {
    todo!()
}

/// Assemble the body of a closure function.
#[instrument]
pub(crate) fn closure_from_expr_closure<'hir>(
    cx: &mut Ctxt<'_, 'hir>,
    hir: &hir::ExprClosure<'hir>,
    captures: &[CaptureMeta],
) -> Result<()> {
    todo!()
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

    let address = cx.with_scope(scope, |cx| {
        let mut first = true;

        let mut patterns = Vec::new();

        for arg in hir.args {
            match *arg {
                hir::FnArg::SelfValue(span) => {
                    if !instance_fn || !first {
                        return Err(CompileError::new(span, CompileErrorKind::UnsupportedSelf));
                    }

                    let name = cx.scopes.name(SELF);
                    let address = cx.insert_expr_with(|address| ExprKind::Address { address });
                    cx.declare(name, address)?;
                }
                hir::FnArg::Pat(hir) => {
                    let address = cx.insert_expr_with(|address| ExprKind::Address { address });
                    patterns.push((hir, address));
                }
            }

            first = false;
        }

        block(cx, hir.body)
    })?;

    cx.scopes.pop(span, scope)?;

    let output = cx.allocator.alloc_for(address);
    asm(cx, address, output)?;
    cx.push(Inst::Return { address: output });
    Ok(())
}

/// An expression that can be assembled.
#[must_use]
#[derive(Debug, Clone, Copy)]
struct Expr<'hir> {
    /// The span of the assembled expression.
    span: Span,
    /// The kind of the expression.
    kind: ExprKind<'hir>,
    /// The output address of the expression.
    slot: Slot,
}

impl<'hir> Expr<'hir> {
    const fn new(span: Span, kind: ExprKind<'hir>, slot: Slot) -> Self {
        Self { span, kind, slot }
    }
}

/// The outcome of a pattern being applied.
#[derive(Debug)]
pub(crate) enum PatOutcome {
    Irrefutable,
    Refutable,
}

impl PatOutcome {
    /// Combine this outcome with another.
    pub(crate) fn combine(self, other: Self) -> Self {
        match (self, other) {
            (PatOutcome::Irrefutable, PatOutcome::Irrefutable) => PatOutcome::Irrefutable,
            _ => PatOutcome::Refutable,
        }
    }
}
/// An expression kind.
#[derive(Debug, Clone, Copy)]
enum ExprKind<'hir> {
    /// An empty value.
    Empty,
    /// An expression referencing another address.
    Address { address: Slot },
    /// An expression referencing a binding.
    Binding { binding: Binding, address: Slot },
    /// A tuple field access.
    TupleFieldAccess { lhs: Slot, index: usize },
    /// A struct field access where the index is the slot used.
    StructFieldAccess {
        lhs: Slot,
        field: &'hir str,
        hash: Hash,
    },
    StructFieldAccessGeneric {
        lhs: Slot,
        hash: Hash,
        generics: Option<(Span, &'hir [hir::Expr<'hir>])>,
    },
    /// An assignment.
    Assign {
        /// Address to assign to.
        address: Slot,
        /// The expression to assign.
        rhs: Slot,
    },
    /// An assignment to a struct field.
    AssignStructField {
        lhs: Slot,
        field: &'hir str,
        rhs: Slot,
    },
    /// An assignment to a tuple field.
    AssignTupleField { lhs: Slot, index: usize, rhs: Slot },
    /// A block with a sequence of statements.
    Block {
        /// Statements in the block.
        statements: &'hir [Slot],
        /// The tail of the block expression.
        tail: Option<Slot>,
    },
    Let {
        /// The assembled pattern.
        pat: Slot,
        /// The expression being bound.
        expr: Slot,
    },
    /// A literal value.
    Store { value: InstValue },
    /// A byte blob.
    Bytes { bytes: &'hir [u8] },
    /// A string.
    String { string: &'hir str },
    /// A unary expression.
    Unary { op: ExprUnOp, expr: Slot },
    /// A binary assign operation.
    BinaryAssign {
        /// The left-hand side of a binary operation.
        lhs: Slot,
        /// The operator.
        op: ast::BinOp,
        /// The right-hand side of a binary operation.
        rhs: Slot,
    },
    /// A binary conditional operation.
    BinaryConditional {
        /// The left-hand side of a binary operation.
        lhs: Slot,
        /// The operator.
        op: ast::BinOp,
        /// The right-hand side of a binary operation.
        rhs: Slot,
    },
    /// A binary expression.
    Binary {
        /// The left-hand side of a binary operation.
        lhs: Slot,
        /// The operator.
        op: ast::BinOp,
        /// The right-hand side of a binary operation.
        rhs: Slot,
    },
    /// The `<target>[<value>]` operation.
    Index { target: Slot, index: Slot },
    /// Looked up metadata.
    Meta {
        /// Private meta.
        meta: &'hir PrivMeta,
        /// The need of the meta expression.
        needs: Needs,
        /// The resolved name.
        named: &'hir Named<'hir>,
    },
    /// Allocate a struct.
    Struct {
        kind: ExprStructKind,
        exprs: &'hir [Slot],
    },
    /// An anonymous tuple.
    Tuple { items: &'hir [Slot] },
    /// Allocate a vector.
    Vec { items: &'hir [Slot] },
    /// A range expression.
    Range {
        from: Slot,
        limits: InstRangeLimits,
        to: Slot,
    },
    /// Allocate an optional value.
    Option {
        /// The value to allocate.
        value: Option<Slot>,
    },
    /// Call the given address.
    CallAddress { address: Slot, args: &'hir [Slot] },
    /// Call the given hash.
    CallHash { hash: Hash, args: &'hir [Slot] },
    /// Call the given instance fn.
    CallInstance {
        lhs: Slot,
        hash: Hash,
        args: &'hir [Slot],
    },
    /// Call the given expression.
    CallExpr { expr: Slot, args: &'hir [Slot] },
    /// Yield the given value.
    Yield {
        /// Yield the given expression.
        expr: Option<Slot>,
    },
    /// Perform an await operation.
    Await {
        /// The expression to await.
        expr: Slot,
    },
    /// Return a kind.
    Return { expr: Slot },
    /// Perform a try operation.
    Try {
        /// The expression to try.
        expr: Slot,
    },
    /// Load a function address.
    Function { hash: Hash },
    /// Load a closure.
    Closure {
        /// The hash of the closure function to load.
        hash: Hash,
        /// Captures to this closure.
        captures: &'hir [Slot],
    },
}

/// An expression that can be assembled.
#[must_use]
#[derive(Debug, Clone, Copy)]
struct Pat<'hir> {
    /// The span of the assembled expression.
    span: Span,
    /// The kind of the expression.
    kind: PatKind<'hir>,
    /// The slot that this pattern belongs to.
    slot: Slot,
    /// The outputs once this pattern has been bound. It also corresponds to the
    /// number of array slots used by the pattern binding.
    outputs: &'hir [Slot],
}

impl<'hir> Pat<'hir> {
    const fn new(span: Span, kind: PatKind<'hir>, slot: Slot, outputs: &'hir [Slot]) -> Self {
        Self {
            span,
            kind,
            slot,
            outputs,
        }
    }
}

/// The kind of a pattern.
#[derive(Debug, Clone, Copy)]
enum PatKind<'hir> {
    /// An ignore pattern.
    Ignore,
    /// A literal value.
    Lit { lit: Slot },
    /// A path pattern.
    Name { name: &'hir str },
    /// A meta binding.
    Meta { meta: &'hir PrivMeta },
    /// A vector pattern.
    Vec { items: &'hir [Slot], is_open: bool },
    /// A tuple pattern.
    Tuple {
        kind: PatTupleKind,
        patterns: &'hir [Slot],
        is_open: bool,
    },
    /// An object pattern.
    Object {
        kind: PatObjectKind<'hir>,
        patterns: &'hir [Slot],
    },
}

/// A bound pattern.
#[must_use]
#[derive(Debug, Clone, Copy)]
struct BoundPat<'hir> {
    /// The span of the assembled bound pattern.
    span: Span,
    /// The kind of the bound pattern.
    kind: BoundPatKind<'hir>,
}

impl<'hir> BoundPat<'hir> {
    const fn new(span: Span, kind: BoundPatKind<'hir>) -> Self {
        Self { span, kind }
    }
}

/// The kind of a bound pattern.
#[derive(Debug, Clone, Copy)]
enum BoundPatKind<'hir> {
    Irrefutable,
    IrrefutableSequence {
        items: &'hir [BoundPat<'hir>],
    },
    /// Bind an expression to the given address.
    Expr {
        name: &'hir str,
        expr: Slot,
    },
    Lit {
        lit: Slot,
        expr: Slot,
    },
    Vec {
        expr: Slot,
        is_open: bool,
        items: &'hir [BoundPat<'hir>],
    },
    AnonymousTuple {
        expr: Slot,
        is_open: bool,
        items: &'hir [BoundPat<'hir>],
    },
    AnonymousObject {
        expr: Slot,
        slot: usize,
        is_open: bool,
        items: &'hir [BoundPat<'hir>],
    },
    TypedSequence {
        type_match: TypeMatch,
        expr: Slot,
        items: &'hir [BoundPat<'hir>],
    },
}

/// The type of a pattern object.
#[derive(Debug, Clone, Copy)]
enum PatObjectKind<'hir> {
    Typed {
        type_match: TypeMatch,
        keys: &'hir [(&'hir str, Hash)],
    },
    Anonymous {
        slot: usize,
        is_open: bool,
    },
}

#[derive(Debug, Clone, Copy)]
enum PatTupleKind {
    Typed { type_match: TypeMatch },
    Anonymous,
}

#[derive(Debug, Clone, Copy)]
enum ExprStructKind {
    /// An anonymous struct.
    Anonymous { slot: usize },
    /// A unit struct.
    Unit { hash: Hash },
    /// A struct with named fields.
    Struct { hash: Hash, slot: usize },
    /// A variant struct with named fields.
    StructVariant { hash: Hash, slot: usize },
}

#[derive(Debug, Clone)]
enum ExprOutcome {
    /// The expression produced no value.
    Empty,
    /// The expression produced output.
    Output,
    /// The expression produced a value which is unreachable.
    Unreachable,
}

#[derive(Debug, Clone, Copy)]
enum ExprUnOp {
    Neg,
    Not,
}

/// An address.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
struct Slot(usize);

impl Slot {
    /// Map the current expression into some other expression.
    fn map<'hir, M>(self, cx: &mut Ctxt<'_, 'hir>, map: M) -> Result<Slot>
    where
        M: FnOnce(&mut Ctxt<'_, 'hir>, ExprKind<'hir>) -> Result<ExprKind<'hir>>,
    {
        let (expr, _) = cx.scopes.expr(cx.span, self)?;
        let kind = map(cx, expr.kind)?;
        cx.scopes.expr_mut(cx.span, self)?.kind = kind;
        Ok(self)
    }
}

/// Scopes to use when compiling.
#[derive(Default)]
struct Scopes<'hir> {
    /// Stack of scopes.
    scopes: slab::Slab<Scope>,
    /// Keeping track of every slot.
    slots: Vec<SlotStorage<'hir>>,
    /// The collection of known names in the scope, so that we can generate the
    /// [BindingName] identifier.
    names_by_id: Vec<Box<str>>,
    /// Names cache.
    names_by_name: HashMap<Box<str>, Name>,
}

impl<'hir> Scopes<'hir> {
    /// Construct an empty scope.
    fn new() -> Self {
        Self {
            scopes: slab::Slab::new(),
            slots: Vec::new(),
            names_by_id: Vec::new(),
            names_by_name: HashMap::new(),
        }
    }

    /// Look up the expression associated with the given output address.
    fn slot_data(&self, span: Span, slot: Slot) -> Result<(&SlotData<'hir>, usize)> {
        let slot = match self.slots.get(slot.0) {
            Some(slot) => slot,
            None => {
                return Err(CompileError::msg(
                    span,
                    format_args!("failed to look up slot {slot:?}"),
                ))
            }
        };

        Ok((&slot.data, slot.uses))
    }

    /// Look up the expression associated with the given slot.
    fn expr(&self, span: Span, slot: Slot) -> Result<(&Expr<'hir>, usize)> {
        let (storage, uses) = self.slot_data(span, slot)?;

        match storage {
            SlotData::Expr(expr) => Ok((expr, uses)),
            _ => {
                return Err(CompileError::msg(
                    span,
                    format_args!("slot {slot:?} exists but isn't an expression"),
                ))
            }
        }
    }

    /// Look up the mutable expression associated with the given slot.
    fn expr_mut(&mut self, span: Span, slot: Slot) -> Result<&mut Expr<'hir>> {
        let storage = match self.slots.get_mut(slot.0) {
            Some(storage) => storage,
            None => {
                return Err(CompileError::msg(
                    span,
                    format_args!("failed to look up slot {slot:?}"),
                ))
            }
        };

        match &mut storage.data {
            SlotData::Expr(expr) => Ok(expr),
            _ => {
                return Err(CompileError::msg(
                    span,
                    format_args!("slot {slot:?} exists but isn't an expression"),
                ))
            }
        }
    }

    /// Look up the pattern associated with the given slot.
    fn pat(&self, span: Span, slot: Slot) -> Result<(&Pat<'hir>, usize)> {
        let (storage, uses) = self.slot_data(span, slot)?;

        match storage {
            SlotData::Pat(pat) => Ok((pat, uses)),
            _ => {
                return Err(CompileError::msg(
                    span,
                    format_args!("slot {slot:?} exists but isn't a pattern"),
                ))
            }
        }
    }

    /// Perform a shallow clone of a scope (without updating users) and return the id of the cloned scope.
    fn clone_scope(&mut self, span: Span, scope: ScopeId) -> Result<ScopeId> {
        let scope = match self.scopes.get(scope.0) {
            Some(scope) => scope.clone(),
            None => {
                return Err(CompileError::msg(span, "missing scope"));
            }
        };

        let scope = ScopeId(self.scopes.insert(scope));
        Ok(scope)
    }

    /// Replace the contents of one scope with another.
    fn replace_scope(&mut self, span: Span, old_scope: ScopeId, new_scope: ScopeId) -> Result<()> {
        // NB: we're removing storage for the old scope.
        let new_scope = match self.scopes.try_remove(new_scope.0) {
            Some(scope) => scope,
            None => {
                return Err(CompileError::msg(span, "missing new scope"));
            }
        };

        let old_scope = match self.scopes.get_mut(old_scope.0) {
            Some(scope) => scope,
            None => {
                return Err(CompileError::msg(span, "missing old scope"));
            }
        };

        *old_scope = new_scope;
        Ok(())
    }

    /// Insert an expression and return its slot address.
    fn insert_expr(&mut self, span: Span, kind: ExprKind<'hir>) -> Slot {
        self.insert_expr_with(span, |_| kind)
    }

    /// Insert an expression and return its slot address.
    fn insert_expr_with<T>(&mut self, span: Span, builder: T) -> Slot
    where
        T: FnOnce(Slot) -> ExprKind<'hir>,
    {
        self.insert_with(span, |address| {
            SlotData::Expr(Expr::new(span, builder(address), address))
        })
    }

    /// Insert a pattern and return its slot address.
    fn insert_pat(&mut self, span: Span, kind: PatKind<'hir>, outputs: &'hir [Slot]) -> Slot {
        self.insert_pat_with(span, |_| kind, outputs)
    }

    /// Insert a pattern and return its slot address.
    fn insert_pat_with<T>(&mut self, span: Span, builder: T, outputs: &'hir [Slot]) -> Slot
    where
        T: FnOnce(Slot) -> PatKind<'hir>,
    {
        self.insert_with(span, |slot| {
            SlotData::Pat(Pat::new(span, builder(slot), slot, outputs))
        })
    }

    /// Allocate an alot address associated with an unknown type.
    #[tracing::instrument(skip_all)]
    fn insert_with<T>(&mut self, span: Span, builder: T) -> Slot
    where
        T: FnOnce(Slot) -> SlotData<'hir>,
    {
        let slot = self.slots.len();
        let address = Slot(slot);
        self.slots.push(SlotStorage {
            uses: 0,
            data: builder(address),
        });
        address
    }

    /// Declare a variable with an already known address.
    #[tracing::instrument(skip_all)]
    fn declare(
        &mut self,
        span: Span,
        scope: ScopeId,
        name: Name,
        slot: Slot,
    ) -> Result<Option<Slot>> {
        let top = match self.scopes.get_mut(scope.0) {
            Some(scope) => scope,
            None => {
                return Err(CompileError::msg(
                    span,
                    format_args!("missing scope {scope:?} to declare variable in"),
                ))
            }
        };

        tracing::trace!(?scope, ?name, ?slot);
        let replaced = top.names.insert(name, slot);
        Ok(replaced)
    }

    fn push_inner(
        &mut self,
        span: Span,
        parent: Option<ScopeId>,
        control_flow: ControlFlow,
    ) -> Result<ScopeId> {
        let scope = match parent {
            Some(scope) => match self.scopes.get_mut(scope.0) {
                Some(parent) => Some(parent),
                None => {
                    return Err(CompileError::msg(
                        span,
                        format_args!("missing parent scope {parent:?}"),
                    ));
                }
            },
            None => None,
        };

        if let Some(scope) = scope {
            scope.children = match scope.children.checked_add(1) {
                Some(children) => children,
                None => return Err(CompileError::msg(span, "overflow adding children")),
            };
        }

        let index = self.scopes.insert(Scope::new(parent, control_flow));

        let id = ScopeId(index);
        debug_assert_ne!(id, ScopeId::CONST);
        tracing::trace!(?id);
        Ok(id)
    }

    /// Push a new scope.
    #[tracing::instrument(skip_all)]
    fn push(&mut self, span: Span, parent: Option<ScopeId>) -> Result<ScopeId> {
        self.push_inner(span, parent, ControlFlow::None)
    }

    /// Push a new scope with the given control flow flag.
    #[tracing::instrument(skip(self))]
    fn push_loop(
        &mut self,
        span: Span,
        parent: Option<ScopeId>,
        label: Option<Name>,
        start: Label,
        end: Label,
        output: AssemblyAddress,
    ) -> Result<ScopeId> {
        self.push_inner(
            span,
            parent,
            ControlFlow::Loop(LoopControlFlow {
                label,
                start,
                end,
                output,
            }),
        )
    }

    /// Push a single branch.
    #[tracing::instrument(skip(self))]
    fn push_branch(&mut self, span: Span, parent: Option<ScopeId>) -> Result<ScopeId> {
        self.push_inner(span, parent, ControlFlow::Branch)
    }

    /// Pop the last scope.
    #[tracing::instrument(skip_all)]
    fn pop(&mut self, span: Span, id: ScopeId) -> Result<()> {
        let scope = match self.scopes.try_remove(id.0) {
            Some(scope) => scope,
            None => return Err(CompileError::msg(span, "missing scope")),
        };

        tracing::trace!(?id, ?scope);

        if scope.children != 0 {
            return Err(CompileError::msg(
                span,
                format_args!("tried to remove scope {id:?} which still has children"),
            ));
        }

        if let Some(parent) = scope.parent.and_then(|p| self.scopes.get_mut(p.0)) {
            parent.children = parent.children.saturating_sub(1);
        }

        Ok(())
    }

    /// Try to get the local with the given name. Returns `None` if it's
    /// missing.
    #[tracing::instrument(skip_all)]
    fn try_lookup(
        &mut self,
        _: Location,
        scope: ScopeId,
        _: &mut dyn CompileVisitor,
        string: &str,
    ) -> Result<Option<(Binding, Slot)>> {
        let name = self.name(string);
        tracing::trace!(string = string, name = ?name);

        let mut current = Some(scope);

        while let Some((id, scope)) = current
            .take()
            .and_then(|s| Some((s, self.scopes.get(s.0)?)))
        {
            if let Some(Slot(slot)) = scope.lookup(name) {
                tracing::trace!("found: {name:?} => {slot:?}");

                if let Some(slot_data) = self.slots.get_mut(slot) {
                    slot_data.uses += 1;
                    let binding = Binding { scope: id, name };
                    return Ok(Some((binding, Slot(slot))));
                }
            }

            current = scope.parent;
        }

        Ok(None)
    }

    /// Lookup the given variable.
    #[tracing::instrument(skip_all)]
    fn lookup(
        &mut self,
        location: Location,
        scope: ScopeId,
        visitor: &mut dyn CompileVisitor,
        name: &str,
    ) -> Result<(Binding, Slot)> {
        match self.try_lookup(location, scope, visitor, name)? {
            Some(output) => Ok(output),
            None => Err(CompileError::new(
                location.span,
                CompileErrorKind::MissingLocal {
                    name: name.to_owned(),
                },
            )),
        }
    }

    /// Find the first ancestor that matches the given predicate.
    #[tracing::instrument(skip(self, filter))]
    fn find_ancestor<T>(&self, scope: ScopeId, mut filter: T) -> Option<ControlFlow>
    where
        T: FnMut(&ControlFlow) -> bool,
    {
        tracing::trace!("find ancestor");
        let mut current = Some(scope);

        while let Some(scope) = current.take().and_then(|s| self.scopes.get(s.0)) {
            tracing::trace!(?scope, "looking");

            if filter(&scope.control_flow) {
                tracing::trace!(?scope, "found");
                return Some(scope.control_flow);
            }

            current = scope.parent;
        }

        None
    }

    /// Get the name of a binding.
    fn name_to_string(&self, span: Span, name: Name) -> Result<&str> {
        match self.names_by_id.get(name.index()) {
            Some(name) => Ok(name),
            None => Err(CompileError::new(
                span,
                CompileErrorKind::MissingBindingName { name },
            )),
        }
    }

    /// Translate the name of a binding.
    fn name(&mut self, string: &str) -> Name {
        if let Some(name) = self.names_by_name.get(string) {
            return *name;
        }

        let name = Name::new(self.names_by_id.len()).expect("ran out of name slots");
        self.names_by_id.push(string.into());
        self.names_by_name.insert(string.into(), name);
        name
    }
}

impl<'hir> fmt::Debug for Scopes<'hir> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        return f
            .debug_struct("Scopes")
            .field("scopes", &ScopesIter(&self.scopes))
            .field("slots", &self.slots)
            .field("names_by_id", &self.names_by_id)
            .field("names_by_name", &self.names_by_name)
            .finish();

        struct ScopesIter<'a, T>(&'a slab::Slab<T>);

        impl<'a, T> fmt::Debug for ScopesIter<'a, T>
        where
            T: fmt::Debug,
        {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                let mut list = f.debug_list();

                for e in self.0 {
                    list.entry(&e);
                }

                list.finish()
            }
        }
    }
}

/// The unique identifier of a scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
struct ScopeId(usize);

impl ScopeId {
    /// The constant scope.
    const CONST: Self = Self(usize::MAX);
}

/// A control flow.
#[derive(Debug, Clone, Copy, Default)]
enum ControlFlow {
    #[default]
    None,
    Loop(LoopControlFlow),
    Branch,
}

/// A loop control flow.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
struct LoopControlFlow {
    label: Option<Name>,
    start: Label,
    end: Label,
    output: AssemblyAddress,
}

#[derive(Debug, Clone, Default)]
struct Scope {
    parent: Option<ScopeId>,
    names: HashMap<Name, Slot>,
    control_flow: ControlFlow,
    children: usize,
}

impl Scope {
    /// Construct a new scope with the specified control flow marker.
    fn new(parent: Option<ScopeId>, control_flow: ControlFlow) -> Self {
        Self {
            parent,
            names: HashMap::new(),
            control_flow,
            children: 0,
        }
    }

    /// Get the given name.
    fn lookup<'data>(&self, name: Name) -> Option<Slot> {
        Some(*self.names.get(&name)?)
    }
}

/// The name of a binding.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Name(NonZeroUsize);

impl fmt::Debug for Name {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("BindingName").field(&self.index()).finish()
    }
}

impl Name {
    fn new(index: usize) -> Option<Self> {
        Some(Self(NonZeroUsize::new(index ^ usize::MAX)?))
    }

    const fn index(self) -> usize {
        self.0.get() ^ usize::MAX
    }
}

/// The exact binding for a single variable in a scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
struct Binding {
    scope: ScopeId,
    name: Name,
}

#[derive(Debug, Clone, Copy)]
enum SlotData<'hir> {
    Expr(Expr<'hir>),
    Pat(Pat<'hir>),
}

#[derive(Debug, Clone, Copy)]
struct SlotStorage<'hir> {
    /// The downstream users of this slot.
    uses: usize,
    /// The expression associated with this slot.
    data: SlotData<'hir>,
}

/// Memory allocator.
#[derive(Debug, Clone)]
pub(crate) struct Allocator {
    slots: slab::Slab<()>,
    /// Translate expression to its corresponding slot.
    expr_to_slot: HashMap<Slot, AssemblyAddress>,
    /// Keep track of the total number of slots used.
    count: usize,
    /// Maximum number of array elements.
    array_index: usize,
    /// The current array count.
    array_count: usize,
}

impl Allocator {
    fn new() -> Self {
        Self {
            slots: slab::Slab::new(),
            expr_to_slot: HashMap::new(),
            count: 0,
            array_index: 0,
            array_count: 0,
        }
    }

    /// Translate an assembly address into a stack address.
    #[tracing::instrument(skip_all)]
    pub(crate) fn translate(&self, span: Span, address: AssemblyAddress) -> Result<Address> {
        fn convert<T>(span: Span, v: Option<T>, msg: &'static str) -> Result<T> {
            match v {
                Some(v) => Ok(v),
                None => Err(CompileError::msg(span, msg)),
            }
        }

        let slot = match address {
            AssemblyAddress::Slot(slot) => slot,
            AssemblyAddress::Array(index) => convert(
                span,
                self.count.checked_add(index),
                "array index out of bound",
            )?,
        };

        Ok(Address(convert(
            span,
            u32::try_from(slot).ok(),
            "slot out of bound",
        )?))
    }

    /// Allocate a new assembly address.
    fn alloc(&mut self) -> AssemblyAddress {
        let slot = self.slots.insert(());
        let address = AssemblyAddress::Slot(slot);
        self.count = self.slots.len().max(self.count);
        tracing::trace!(count = self.count, address = ?address);
        address
    }

    /// Allocate a new output address slotted by the given expression address.
    fn alloc_for(&mut self, expr: Slot) -> AssemblyAddress {
        if let Some(address) = self.expr_to_slot.get(&expr) {
            return *address;
        }

        let address = self.alloc();
        self.expr_to_slot.insert(expr, address);
        address
    }

    /// Free an assembly address.
    fn free(&mut self, span: Span, address: AssemblyAddress) -> Result<()> {
        if let AssemblyAddress::Slot(slot) = address {
            if self.slots.try_remove(slot).is_none() {
                return Err(CompileError::msg(span, format_args!("missing slot {slot}")));
            }
        }

        Ok(())
    }

    /// Allocate many addresses.
    fn alloc_many<const N: usize>(&mut self) -> [AssemblyAddress; N] {
        std::array::from_fn(|_| self.alloc())
    }

    /// Free many addresses.
    fn free_many<const N: usize>(
        &mut self,
        span: Span,
        addresses: [AssemblyAddress; N],
    ) -> Result<()> {
        for address in addresses {
            self.free(span, address)?;
        }

        Ok(())
    }

    /// Get the current array index as an assembly address.
    fn array_index(&self) -> usize {
        self.array_index
    }

    /// Get the current array index as an address.
    fn array_address(&self) -> AssemblyAddress {
        AssemblyAddress::Array(self.array_index())
    }

    /// Mark multiple array items as occupied.
    #[tracing::instrument(skip_all)]
    fn alloc_array_items(&mut self, len: usize) {
        self.array_index = self.array_index.saturating_add(len);
        self.array_count = self.array_count.max(self.array_index);
        tracing::trace!(?self.array_count, ?self.array_index);
    }

    /// Mark one array item as occupied, forcing additional allocations to
    /// utilize higher array indexes.
    #[tracing::instrument(skip_all)]
    fn alloc_array_item(&mut self) {
        self.alloc_array_items(1)
    }

    /// Allocate an array item.
    #[tracing::instrument(skip_all)]
    fn free_array(&mut self, span: Span, items: usize) -> Result<()> {
        self.array_index = match self.array_index.checked_sub(items) {
            Some(array) => array,
            None => return Err(CompileError::msg(span, "scope array index out-of-bounds")),
        };

        tracing::trace!(?self.array_count, ?self.array_index);
        Ok(())
    }

    /// Free a single array item.
    #[tracing::instrument(skip_all)]
    fn free_array_item(&mut self, span: Span) -> Result<()> {
        self.free_array(span, 1)
    }

    /// The size of the frame.
    pub(crate) fn frame(&self) -> usize {
        self.count.saturating_add(self.array_count)
    }
}

/// Assemble the given address.
fn asm<'hir>(
    cx: &mut Ctxt<'_, 'hir>,
    address: Slot,
    output: AssemblyAddress,
) -> Result<ExprOutcome> {
    let (expr, uses) = cx.scopes.expr(cx.span, address)?;

    match expr.kind {
        ExprKind::Empty => {}
        ExprKind::Address { address } => {
            asm_address(cx, address, output);
        }
        ExprKind::Binding { binding, address } => {
            asm_binding(cx, binding, address, output)?;
        }
        ExprKind::Block { statements, tail } => {
            asm_block(cx, statements, tail, output)?;
        }
        ExprKind::Let { pat, expr } => asm_let(cx, pat, expr)?,
        ExprKind::Store { value } => {
            asm_store(cx, value, output)?;
        }
        ExprKind::Bytes { bytes } => {
            asm_bytes(cx, bytes, output)?;
        }
        ExprKind::String { string } => {
            asm_string(cx, string, output)?;
        }
        ExprKind::Unary { op, expr } => {
            asm_unary(cx, op, expr, output)?;
        }
        ExprKind::BinaryAssign { lhs, op, rhs } => {
            asm_binary_assign(cx, lhs, op, rhs, output)?;
        }
        ExprKind::BinaryConditional { lhs, op, rhs } => {
            asm_binary_conditional(cx, lhs, op, rhs, output)?;
        }
        ExprKind::Binary { lhs, op, rhs } => {
            asm_binary(cx, lhs, op, rhs, output)?;
        }
        ExprKind::Index { target, index } => {
            asm_index(cx, target, index, output)?;
        }
        ExprKind::Meta { meta, needs, named } => {
            asm_meta(cx, meta, needs, named, output)?;
        }
        ExprKind::Struct { kind, exprs } => {
            asm_struct(cx, kind, exprs, output)?;
        }
        ExprKind::Vec { items } => {
            asm_vec(cx, items, output)?;
        }
        ExprKind::Range { from, limits, to } => {
            asm_range(cx, from, limits, to, output)?;
        }
        ExprKind::Tuple { items } => {
            asm_tuple(cx, items, output)?;
        }
        ExprKind::Option { value } => {
            asm_option(cx, value, output)?;
        }
        ExprKind::TupleFieldAccess { lhs, index } => {
            asm_tuple_field_access(cx, lhs, index, output)?;
        }
        ExprKind::StructFieldAccess { lhs, field, .. } => {
            asm_struct_field_access(cx, lhs, field, output)?;
        }
        ExprKind::StructFieldAccessGeneric { .. } => {
            return Err(cx.error(CompileErrorKind::ExpectedExpr));
        }
        ExprKind::Assign { address, rhs } => {
            asm_assign(cx, address, rhs)?;
        }
        ExprKind::AssignStructField { lhs, field, rhs } => {
            asm_assign_struct_field(cx, lhs, field, rhs, output)?;
        }
        ExprKind::AssignTupleField { lhs, index, rhs } => {
            asm_assign_tuple_field(cx, lhs, index, rhs, output)?;
        }
        ExprKind::CallAddress {
            address: function,
            args,
        } => {
            asm_call_address(cx, function, args, output)?;
        }
        ExprKind::CallHash { hash, args } => {
            asm_call_hash(cx, args, hash, output)?;
        }
        ExprKind::CallInstance { lhs, hash, args } => {
            asm_call_instance(cx, lhs, args, hash, output)?;
        }
        ExprKind::CallExpr { expr, args } => {
            asm_call_expr(cx, expr, args, output)?;
        }
        ExprKind::Yield { expr } => {
            asm_yield(cx, expr, output)?;
        }
        ExprKind::Await { expr } => {
            asm_await(cx, expr, output)?;
        }
        ExprKind::Return { expr } => {
            asm_return(cx, expr, output)?;
            return Ok(ExprOutcome::Unreachable);
        }
        ExprKind::Try { expr } => {
            asm_try(cx, expr, output)?;
        }
        ExprKind::Function { hash } => {
            asm_function(cx, hash, output);
        }
        ExprKind::Closure { hash, captures } => {
            asm_closure(cx, captures, hash, output)?;
        }
    }

    return Ok(ExprOutcome::Output);

    /// Allocate space for the specified array.
    fn asm_array<'hir, I>(cx: &mut Ctxt<'_, 'hir>, expressions: I) -> Result<ArrayAddress>
    where
        I: IntoIterator<Item = Slot>,
    {
        let index = cx.allocator.array_index();
        let mut count = 0;

        for expr in expressions {
            let output = cx.allocator.array_index();
            asm(cx, expr, AssemblyAddress::Array(output))?;
            cx.allocator.alloc_array_item();
            count += 1;
        }

        Ok(ArrayAddress::new(index, count))
    }

    #[instrument]
    fn asm_address<'hir>(cx: &mut Ctxt<'_, 'hir>, address: Slot, output: AssemblyAddress) {
        let address = cx.allocator.alloc_for(address);

        if address != output {
            cx.push(Inst::Copy { address, output });
        }
    }

    #[instrument]
    fn asm_binding<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        binding: Binding,
        address: Slot,
        output: AssemblyAddress,
    ) -> Result<()> {
        let address = cx.allocator.alloc_for(address);

        if address != output {
            let name = cx.scopes.name_to_string(cx.span, binding.name)?;
            let comment = format!("copy `{name}`");
            cx.push_with_comment(Inst::Copy { address, output }, comment);
        }

        Ok(())
    }

    #[instrument]
    fn asm_block<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        statements: &[Slot],
        tail: Option<Slot>,
        output: AssemblyAddress,
    ) -> Result<()> {
        for address in statements {
            asm(cx, *address, output)?;
        }

        if let Some(tail) = tail {
            asm(cx, tail, output)?;
        }

        Ok(())
    }

    #[instrument]
    fn asm_let<'hir>(cx: &mut Ctxt<'_, 'hir>, pat: Slot, expr: Slot) -> Result<()> {
        let panic_label = cx.new_label("let_panic");

        if let PatOutcome::Refutable = asm_pat(cx, pat, expr, panic_label)? {
            let end = cx.new_label("pat_end");
            cx.push(Inst::Jump { label: end });
            cx.label(panic_label)?;
            cx.push(Inst::Panic {
                reason: PanicReason::UnmatchedPattern,
            });
            cx.label(end)?;
        }

        Ok(())
    }

    #[instrument]
    fn asm_pat<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        pat: Slot,
        expr: Slot,
        label: Label,
    ) -> Result<PatOutcome> {
        let bound_pat = bind_pat(cx, pat, expr)?;

        todo!()
    }

    #[instrument]
    fn asm_store<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        value: InstValue,
        output: AssemblyAddress,
    ) -> Result<()> {
        cx.push(Inst::Store { value, output });
        Ok(())
    }

    #[instrument]
    fn asm_bytes<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        bytes: &[u8],
        output: AssemblyAddress,
    ) -> Result<()> {
        let slot = cx.q.unit.new_static_bytes(cx.span, bytes)?;
        cx.push(Inst::Bytes { slot, output });
        Ok(())
    }

    #[instrument]
    fn asm_string<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        string: &str,
        output: AssemblyAddress,
    ) -> Result<()> {
        let slot = cx.q.unit.new_static_string(cx.span, string)?;
        cx.push(Inst::String { slot, output });
        Ok(())
    }

    #[instrument]
    fn asm_unary<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        op: ExprUnOp,
        expr: Slot,
        output: AssemblyAddress,
    ) -> Result<()> {
        asm(cx, expr, output)?;
        Ok(match op {
            ExprUnOp::Neg => {
                cx.push(Inst::Neg {
                    address: output,
                    output,
                });
            }
            ExprUnOp::Not => {
                cx.push(Inst::Not {
                    address: output,
                    output,
                });
            }
        })
    }

    #[instrument]
    fn asm_binary_assign<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        lhs: Slot,
        op: ast::BinOp,
        rhs: Slot,
        output: AssemblyAddress,
    ) -> Result<()> {
        let lhs = cx.allocator.alloc_for(lhs);

        let op = match op {
            ast::BinOp::AddAssign(..) => InstAssignOp::Add,
            ast::BinOp::SubAssign(..) => InstAssignOp::Sub,
            ast::BinOp::MulAssign(..) => InstAssignOp::Mul,
            ast::BinOp::DivAssign(..) => InstAssignOp::Div,
            ast::BinOp::RemAssign(..) => InstAssignOp::Rem,
            ast::BinOp::BitAndAssign(..) => InstAssignOp::BitAnd,
            ast::BinOp::BitXorAssign(..) => InstAssignOp::BitXor,
            ast::BinOp::BitOrAssign(..) => InstAssignOp::BitOr,
            ast::BinOp::ShlAssign(..) => InstAssignOp::Shl,
            ast::BinOp::ShrAssign(..) => InstAssignOp::Shr,
            _ => {
                return Err(cx.error(CompileErrorKind::UnsupportedBinaryExpr));
            }
        };

        cx.push(Inst::Assign {
            lhs,
            rhs: output,
            target: InstTarget::Offset,
            op,
            // NB: while an assign operation doesn't output anything, this
            // might result in a call to an external function which expects
            // to always have a writable address slot available, regardless
            // of what it outputs. So we hand it a temporary slot, which is
            // just an address on the stack which will probably immediately
            // be used for something else.
            //
            // The virtual machine ensures that this address is cleared
            // immediately after it's been called.
            output,
        });

        Ok(())
    }

    #[instrument]
    fn asm_binary_conditional<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        lhs: Slot,
        op: ast::BinOp,
        rhs: Slot,
        output: AssemblyAddress,
    ) -> Result<()> {
        let end_label = cx.new_label("conditional_end");

        cx.with_state_checkpoint(|cx| asm(cx, lhs, output))?;

        match op {
            ast::BinOp::And(..) => {
                cx.push(Inst::JumpIfNot {
                    address: output,
                    label: end_label,
                });
            }
            ast::BinOp::Or(..) => {
                cx.push(Inst::JumpIf {
                    address: output,
                    label: end_label,
                });
            }
            op => {
                return Err(cx.error(CompileErrorKind::UnsupportedBinaryOp { op }));
            }
        }

        cx.with_state_checkpoint(|cx| asm(cx, rhs, output))?;
        cx.label(end_label)?;
        Ok(())
    }

    /// Assembling of a binary expression.
    #[instrument]
    fn asm_binary<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        lhs: Slot,
        op: ast::BinOp,
        rhs: Slot,
        output: AssemblyAddress,
    ) -> Result<()> {
        let op = match op {
            ast::BinOp::Eq(..) => InstOp::Eq,
            ast::BinOp::Neq(..) => InstOp::Neq,
            ast::BinOp::Lt(..) => InstOp::Lt,
            ast::BinOp::Gt(..) => InstOp::Gt,
            ast::BinOp::Lte(..) => InstOp::Lte,
            ast::BinOp::Gte(..) => InstOp::Gte,
            ast::BinOp::Is(..) => InstOp::Is,
            ast::BinOp::IsNot(..) => InstOp::IsNot,
            ast::BinOp::And(..) => InstOp::And,
            ast::BinOp::Or(..) => InstOp::Or,
            ast::BinOp::Add(..) => InstOp::Add,
            ast::BinOp::Sub(..) => InstOp::Sub,
            ast::BinOp::Div(..) => InstOp::Div,
            ast::BinOp::Mul(..) => InstOp::Mul,
            ast::BinOp::Rem(..) => InstOp::Rem,
            ast::BinOp::BitAnd(..) => InstOp::BitAnd,
            ast::BinOp::BitXor(..) => InstOp::BitXor,
            ast::BinOp::BitOr(..) => InstOp::BitOr,
            ast::BinOp::Shl(..) => InstOp::Shl,
            ast::BinOp::Shr(..) => InstOp::Shr,
            op => {
                return Err(cx.error(CompileErrorKind::UnsupportedBinaryOp { op }));
            }
        };

        let b_output = cx.allocator.alloc();
        asm(cx, lhs, output)?;
        asm(cx, rhs, b_output)?;

        cx.push(Inst::Op {
            op,
            a: output,
            b: b_output,
            output,
        });

        cx.allocator.free(cx.span, b_output)?;
        Ok(())
    }

    #[instrument]
    fn asm_index<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        target: Slot,
        index: Slot,
        output: AssemblyAddress,
    ) -> Result<()> {
        let index_output = cx.allocator.alloc();
        asm(cx, target, output)?;
        asm(cx, index, index_output)?;
        cx.push(Inst::IndexGet {
            address: output,
            index: index_output,
            output,
        });
        cx.allocator.free(cx.span, index_output)?;
        Ok(())
    }

    /// Compile an item.
    #[instrument]
    fn asm_meta<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        meta: &PrivMeta,
        needs: Needs,
        named: &'hir Named<'hir>,
        output: AssemblyAddress,
    ) -> Result<ExprOutcome> {
        let outcome = if let Needs::Value = needs {
            match &meta.kind {
                PrivMetaKind::Struct {
                    type_hash,
                    variant: PrivVariantMeta::Unit,
                    ..
                }
                | PrivMetaKind::Variant {
                    type_hash,
                    variant: PrivVariantMeta::Unit,
                    ..
                } => {
                    named.assert_not_generic()?;

                    cx.push_with_comment(
                        Inst::Call {
                            hash: *type_hash,
                            address: output,
                            count: 0,
                            output,
                        },
                        meta.info(cx.q.pool),
                    );

                    ExprOutcome::Output
                }
                PrivMetaKind::Variant {
                    variant: PrivVariantMeta::Tuple(tuple),
                    ..
                }
                | PrivMetaKind::Struct {
                    variant: PrivVariantMeta::Tuple(tuple),
                    ..
                } if tuple.args == 0 => {
                    named.assert_not_generic()?;

                    cx.push_with_comment(
                        Inst::Call {
                            hash: tuple.hash,
                            address: output,
                            count: 0,
                            output,
                        },
                        meta.info(cx.q.pool),
                    );

                    ExprOutcome::Output
                }
                PrivMetaKind::Struct {
                    variant: PrivVariantMeta::Tuple(tuple),
                    ..
                } => {
                    named.assert_not_generic()?;

                    cx.push_with_comment(
                        Inst::LoadFn {
                            hash: tuple.hash,
                            output,
                        },
                        meta.info(cx.q.pool),
                    );

                    ExprOutcome::Output
                }
                PrivMetaKind::Variant {
                    variant: PrivVariantMeta::Tuple(tuple),
                    ..
                } => {
                    named.assert_not_generic()?;

                    cx.push_with_comment(
                        Inst::LoadFn {
                            hash: tuple.hash,
                            output,
                        },
                        meta.info(cx.q.pool),
                    );

                    ExprOutcome::Output
                }
                PrivMetaKind::Function { type_hash, .. } => {
                    let hash = if let Some((span, generics)) = named.generics {
                        let parameters =
                            cx.with_span(span, |cx| generics_parameters(cx, generics))?;
                        type_hash.with_parameters(parameters)
                    } else {
                        *type_hash
                    };

                    cx.push_with_comment(Inst::LoadFn { hash, output }, meta.info(cx.q.pool));
                    ExprOutcome::Output
                }
                PrivMetaKind::Const {
                    const_value: value, ..
                } => {
                    named.assert_not_generic()?;
                    let expr = const_value(cx, value)?;
                    asm(cx, expr, output)?
                }
                _ => {
                    return Err(cx.error(CompileErrorKind::ExpectedMeta {
                        meta: meta.info(cx.q.pool),
                        expected: "something that can be used as a value",
                    }));
                }
            }
        } else {
            named.assert_not_generic()?;

            let type_hash = match meta.type_hash_of() {
                Some(type_hash) => type_hash,
                None => {
                    return Err(cx.error(CompileErrorKind::ExpectedMeta {
                        meta: meta.info(cx.q.pool),
                        expected: "something that has a type",
                    }));
                }
            };

            cx.push(Inst::Store {
                value: InstValue::Type(type_hash),
                output,
            });

            ExprOutcome::Output
        };

        Ok(outcome)
    }

    #[instrument]
    fn asm_struct<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        kind: ExprStructKind,
        exprs: &[Slot],
        output: AssemblyAddress,
    ) -> Result<()> {
        let address = asm_array(cx, exprs.iter().copied())?;
        match kind {
            ExprStructKind::Anonymous { slot } => {
                cx.push(Inst::Object {
                    address: address.address(),
                    slot,
                    output,
                });
            }
            ExprStructKind::Unit { hash } => {
                cx.push(Inst::UnitStruct { hash, output });
            }
            ExprStructKind::Struct { hash, slot } => {
                cx.push(Inst::Struct {
                    hash,
                    address: address.address(),
                    slot,
                    output,
                });
            }
            ExprStructKind::StructVariant { hash, slot } => {
                cx.push(Inst::StructVariant {
                    hash,
                    address: address.address(),
                    slot,
                    output,
                });
            }
        }
        address.free(cx)?;
        Ok(())
    }

    #[instrument]
    fn asm_vec<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        items: &[Slot],
        output: AssemblyAddress,
    ) -> Result<()> {
        let address = asm_array(cx, items.iter().copied())?;

        cx.push(Inst::Vec {
            address: address.address(),
            count: address.count(),
            output,
        });

        address.free(cx)?;
        Ok(())
    }

    #[instrument]
    fn asm_range<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        from: Slot,
        limits: InstRangeLimits,
        to: Slot,
        output: AssemblyAddress,
    ) -> Result<()> {
        let to_output = cx.allocator.alloc();
        asm(cx, from, output)?;
        asm(cx, from, to_output)?;

        cx.push(Inst::Range {
            from: output,
            to: to_output,
            limits,
            output,
        });

        cx.allocator.free(cx.span, to_output)?;
        Ok(())
    }

    #[instrument]
    fn asm_tuple<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        items: &[Slot],
        output: AssemblyAddress,
    ) -> Result<()> {
        match items {
            &[a] => {
                asm(cx, a, output)?;

                cx.push(Inst::Tuple1 {
                    args: [output],
                    output,
                });
            }
            &[a, b] => {
                let b_output = cx.allocator.alloc();

                asm(cx, a, output)?;
                asm(cx, b, b_output)?;

                cx.push(Inst::Tuple2 {
                    args: [output, b_output],
                    output,
                });

                cx.allocator.free(cx.span, b_output)?;
            }
            &[a, b, c] => {
                let [b_output, c_output] = cx.allocator.alloc_many();

                asm(cx, a, output)?;
                asm(cx, b, b_output)?;
                asm(cx, c, c_output)?;

                cx.push(Inst::Tuple3 {
                    args: [output, b_output, c_output],
                    output,
                });

                cx.allocator.free_many(cx.span, [b_output, c_output])?;
            }
            &[a, b, c, d] => {
                let [b_output, c_output, d_output] = cx.allocator.alloc_many();

                asm(cx, a, output)?;
                asm(cx, b, b_output)?;
                asm(cx, c, c_output)?;
                asm(cx, d, d_output)?;

                cx.push(Inst::Tuple4 {
                    args: [output, b_output, c_output, d_output],
                    output,
                });

                cx.allocator
                    .free_many(cx.span, [b_output, c_output, d_output])?;
            }
            args => {
                let address = asm_array(cx, args.iter().copied())?;

                cx.push(Inst::Tuple {
                    address: address.address(),
                    count: args.len(),
                    output,
                });

                address.free(cx)?;
            }
        }

        Ok(())
    }

    #[instrument]
    fn asm_option<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        value: Option<Slot>,
        output: AssemblyAddress,
    ) -> Result<()> {
        let variant = match value {
            Some(value) => {
                asm(cx, value, output)?;
                InstVariant::Some
            }
            None => InstVariant::None,
        };

        cx.push(Inst::Variant {
            address: output,
            variant,
            output,
        });

        Ok(())
    }

    #[instrument]
    fn asm_tuple_field_access<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        lhs: Slot,
        index: usize,
        output: AssemblyAddress,
    ) -> Result<()> {
        asm(cx, lhs, output)?;

        cx.push(Inst::TupleIndexGet {
            address: output,
            index,
            output,
        });

        Ok(())
    }

    #[instrument]
    fn asm_struct_field_access<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        lhs: Slot,
        field: &str,
        output: AssemblyAddress,
    ) -> Result<()> {
        asm(cx, lhs, output)?;

        let slot = cx.q.unit.new_static_string(cx.span, field)?;

        cx.push(Inst::ObjectIndexGet {
            address: output,
            slot,
            output,
        });

        Ok(())
    }

    #[instrument]
    fn asm_assign<'hir>(cx: &mut Ctxt<'_, 'hir>, address: Slot, rhs: Slot) -> Result<()> {
        let (expr, _) = cx.scopes.expr(cx.span, address)?;

        let output = match expr.kind {
            ExprKind::Address { address } => cx.allocator.alloc_for(address),
            ExprKind::Binding { address, .. } => cx.allocator.alloc_for(address),
            _ => {
                todo!()
            }
        };

        asm(cx, rhs, output)?;
        Ok(())
    }

    #[instrument]
    fn asm_assign_struct_field<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        lhs: Slot,
        field: &str,
        rhs: Slot,
        output: AssemblyAddress,
    ) -> Result<()> {
        let rhs_output = cx.allocator.alloc();

        asm(cx, lhs, output)?;
        asm(cx, rhs, rhs_output)?;

        let slot = cx.q.unit.new_static_string(cx.span, field)?;

        cx.push(Inst::ObjectIndexSet {
            address: output,
            value: rhs_output,
            slot,
            output,
        });

        cx.allocator.free(cx.span, rhs_output)?;
        Ok(())
    }

    #[instrument]
    fn asm_assign_tuple_field<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        lhs: Slot,
        index: usize,
        rhs: Slot,
        output: AssemblyAddress,
    ) -> Result<()> {
        let rhs_output = cx.allocator.alloc();

        asm(cx, lhs, output)?;
        asm(cx, rhs, rhs_output)?;

        cx.push(Inst::TupleIndexSet {
            address: output,
            value: rhs_output,
            index,
            output,
        });

        cx.allocator.free(cx.span, rhs_output)?;
        Ok(())
    }

    #[instrument]
    fn asm_call_address<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        function: Slot,
        args: &[Slot],
        output: AssemblyAddress,
    ) -> Result<()> {
        let function_output = cx.allocator.alloc();
        asm(cx, function, function_output)?;
        let address = asm_array(cx, args.iter().copied())?;

        cx.push(Inst::CallFn {
            function: function_output,
            address: address.address(),
            count: address.count(),
            output,
        });

        address.free(cx)?;
        cx.allocator.free(cx.span, function_output)?;
        Ok(())
    }

    #[instrument]
    fn asm_call_hash<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        args: &[Slot],
        hash: Hash,
        output: AssemblyAddress,
    ) -> Result<()> {
        let array = asm_array(cx, args.iter().copied())?;

        cx.push(Inst::Call {
            hash,
            address: array.address(),
            count: array.count(),
            output,
        });

        array.free(cx)?;
        Ok(())
    }

    #[instrument]
    fn asm_call_instance<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        lhs: Slot,
        args: &[Slot],
        hash: Hash,
        output: AssemblyAddress,
    ) -> Result<()> {
        let address = cx.allocator.array_address();

        {
            let output = cx.allocator.array_address();
            asm(cx, lhs, output)?;
            cx.allocator.alloc_array_item();
        }

        for hir in args {
            let output = cx.allocator.array_address();
            asm(cx, *hir, output)?;
            cx.allocator.alloc_array_item();
        }

        cx.push(Inst::CallInstance {
            hash,
            address,
            count: args.len(),
            output,
        });

        cx.allocator.free_array(cx.span, args.len())?;
        Ok(())
    }

    #[instrument]
    fn asm_call_expr<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        expr: Slot,
        args: &[Slot],
        output: AssemblyAddress,
    ) -> Result<()> {
        asm(cx, expr, output)?;
        let array = asm_array(cx, args.iter().copied())?;

        cx.push(Inst::CallFn {
            function: output,
            address: array.address(),
            count: array.count(),
            output,
        });

        array.free(cx)?;
        Ok(())
    }

    #[instrument]
    fn asm_yield<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        expr: Option<Slot>,
        output: AssemblyAddress,
    ) -> Result<()> {
        Ok(match expr {
            Some(expr) => {
                asm(cx, expr, output)?;

                cx.push(Inst::Yield {
                    address: output,
                    output,
                });
            }
            None => {
                cx.push(Inst::YieldUnit { output });
            }
        })
    }

    #[instrument]
    fn asm_await<'hir>(cx: &mut Ctxt<'_, 'hir>, expr: Slot, output: AssemblyAddress) -> Result<()> {
        asm(cx, expr, output)?;

        cx.push(Inst::Await {
            address: output,
            output,
        });

        Ok(())
    }

    #[instrument]
    fn asm_return<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        expr: Slot,
        output: AssemblyAddress,
    ) -> Result<()> {
        asm(cx, expr, output)?;
        cx.push(Inst::Return { address: output });
        Ok(())
    }

    #[instrument]
    fn asm_try<'hir>(cx: &mut Ctxt<'_, 'hir>, expr: Slot, output: AssemblyAddress) -> Result<()> {
        asm(cx, expr, output)?;

        cx.push(Inst::Try {
            address: output,
            output,
        });

        Ok(())
    }

    #[instrument]
    fn asm_function<'hir>(cx: &mut Ctxt<'_, 'hir>, hash: Hash, output: AssemblyAddress) {
        cx.push(Inst::LoadFn { hash, output });
    }

    #[instrument]
    fn asm_closure<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        captures: &[Slot],
        hash: Hash,
        output: AssemblyAddress,
    ) -> Result<()> {
        let array = asm_array(cx, captures.iter().copied())?;

        cx.push(Inst::Closure {
            hash,
            address: array.address(),
            count: array.count(),
            output,
        });

        array.free(cx)?;
        Ok(())
    }
}

/// Assemble a block.
#[instrument]
fn block<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: &hir::Block<'hir>) -> Result<Slot> {
    cx.with_span(hir.span(), |cx| {
        let statements = iter!(cx; hir.statements, |stmt| {
            let address = match *stmt {
                hir::Stmt::Local(hir) => {
                    let expr = expr_value(cx, hir.expr)?;
                    let pat = pat(cx, hir.pat)?;
                    cx.insert_expr(ExprKind::Let { pat, expr })
                }
                hir::Stmt::Expr(hir) => expr_value(cx, hir)?,
            };

            address
        });

        let tail = if let Some(hir) = hir.tail {
            Some(expr_value(cx, hir)?)
        } else {
            None
        };

        Ok(cx.insert_expr(ExprKind::Block { statements, tail }))
    })
}

/// Compile an expression.
#[instrument]
fn expr_value<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: &hir::Expr<'hir>) -> Result<Slot> {
    expr(cx, hir, Needs::Value)
}

/// Custom needs expression.
#[instrument]
fn expr<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: &hir::Expr<'hir>, needs: Needs) -> Result<Slot> {
    return cx.with_span(hir.span, |cx| {
        let address = match hir.kind {
            hir::ExprKind::Empty => cx.insert_expr(ExprKind::Empty),
            hir::ExprKind::Path(hir) => path(cx, hir, needs)?,
            hir::ExprKind::Assign(hir) => assign(cx, hir)?,
            hir::ExprKind::Call(hir) => call(cx, hir)?,
            hir::ExprKind::FieldAccess(hir) => field_access(cx, hir)?,
            hir::ExprKind::Unary(hir) => unary(cx, hir)?,
            hir::ExprKind::Binary(hir) => binary(cx, hir)?,
            hir::ExprKind::Index(hir) => index(cx, hir)?,
            hir::ExprKind::Block(hir) => expr_block(cx, hir)?,
            hir::ExprKind::Yield(hir) => yield_(cx, hir)?,
            hir::ExprKind::Return(hir) => return_(cx, hir)?,
            hir::ExprKind::Await(hir) => await_(cx, hir)?,
            hir::ExprKind::Try(hir) => try_(cx, hir)?,
            hir::ExprKind::Closure(hir) => closure(cx, hir)?,
            hir::ExprKind::Lit(hir) => lit(cx, hir)?,
            hir::ExprKind::Object(hir) => object(cx, hir)?,
            hir::ExprKind::Tuple(hir) => tuple(cx, hir)?,
            hir::ExprKind::Vec(hir) => vec(cx, hir)?,
            hir::ExprKind::Range(hir) => range(cx, hir)?,
            hir::ExprKind::Group(hir) => expr(cx, hir, needs)?,
            hir::ExprKind::Select(..) => todo!(),
            hir::ExprKind::Loop(..) => todo!(),
            hir::ExprKind::Break(..) => todo!(),
            hir::ExprKind::Continue(..) => todo!(),
            hir::ExprKind::Let(..) => todo!(),
            hir::ExprKind::If(..) => todo!(),
            hir::ExprKind::Match(..) => todo!(),
            hir::ExprKind::MacroCall(..) => todo!(),
        };

        Ok(address)
    });

    /// Assembling of a binary expression.
    #[instrument]
    fn path<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: &'hir hir::Path, needs: Needs) -> Result<Slot> {
        let span = hir.span();
        let loc = Location::new(cx.source_id, span);

        if let Some(ast::PathKind::SelfValue) = hir.as_kind() {
            let (binding, address) = cx.scopes.lookup(loc, cx.scope, cx.q.visitor, SELF)?;
            return Ok(cx.insert_expr(ExprKind::Binding { binding, address }));
        }

        let named = cx.convert_path(hir)?;

        if let Needs::Value = needs {
            if let Some(local) = named.as_local() {
                let local = local.resolve(resolve_context!(cx.q))?;

                if let Some((binding, address)) =
                    cx.scopes.try_lookup(loc, cx.scope, cx.q.visitor, local)?
                {
                    return Ok(cx.insert_expr(ExprKind::Binding { binding, address }));
                }
            }
        }

        if let Some(meta) = cx.try_lookup_meta(span, named.item)? {
            return Ok(cx.insert_expr(ExprKind::Meta {
                meta: alloc!(cx; meta),
                needs,
                named: alloc!(cx; named),
            }));
        }

        if let (Needs::Value, Some(local)) = (needs, named.as_local()) {
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

    /// An assignment expression.
    #[instrument]
    fn assign<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: &'hir hir::ExprAssign<'hir>) -> Result<Slot> {
        let rhs = expr_value(cx, &hir.rhs)?;

        expr_value(cx, &hir.lhs)?.map(cx, |cx, kind| match kind {
            ExprKind::Address { address, .. } => Ok(ExprKind::Assign { address, rhs }),
            ExprKind::StructFieldAccess { lhs, field, .. } => {
                Ok(ExprKind::AssignStructField { lhs, field, rhs })
            }
            ExprKind::TupleFieldAccess { lhs, index } => {
                Ok(ExprKind::AssignTupleField { lhs, index, rhs })
            }
            _ => Err(cx.error(CompileErrorKind::UnsupportedAssignExpr)),
        })
    }

    /// Assemble a call expression.
    #[instrument]
    fn call<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: &hir::ExprCall<'hir>) -> Result<Slot> {
        expr_value(cx, hir.expr)?.map(cx, |cx, kind| match kind {
            ExprKind::Address { address, .. } => Ok(ExprKind::CallAddress {
                address,
                args: iter!(cx; hir.args, |hir| expr_value(cx, hir)?),
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
                        let value = const_value(cx, &value)?;
                        let (expr, _) = cx.scopes.expr(cx.span, value)?;
                        return Ok(expr.kind);
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
                    args: iter!(cx; hir.args, |hir| expr_value(cx, hir)?),
                })
            }
            ExprKind::StructFieldAccess { lhs, hash, .. } => Ok(ExprKind::CallInstance {
                lhs,
                hash,
                args: iter!(cx; hir.args, |hir| expr_value(cx, hir)?),
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
                    args: iter!(cx; hir.args, |hir| expr_value(cx, hir)?),
                })
            }
            kind => Ok(ExprKind::CallExpr {
                expr: cx.insert_expr(kind),
                args: iter!(cx; hir.args, |hir| expr_value(cx, hir)?),
            }),
        })
    }

    /// Decode a field access expression.
    #[instrument]
    fn field_access<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        hir: &hir::ExprFieldAccess<'hir>,
    ) -> Result<Slot> {
        match hir.expr_field {
            hir::ExprField::Path(path) => {
                if let Some(ident) = path.try_as_ident() {
                    let n = ident.resolve(resolve_context!(cx.q))?;
                    let field = str!(cx; n);
                    let hash = Hash::instance_fn_name(n);
                    let lhs = expr_value(cx, hir.expr)?;
                    return Ok(cx.insert_expr(ExprKind::StructFieldAccess { lhs, field, hash }));
                }

                if let Some((ident, generics)) = path.try_as_ident_generics() {
                    let n = ident.resolve(resolve_context!(cx.q))?;
                    let hash = Hash::instance_fn_name(n.as_ref());
                    let lhs = expr_value(cx, hir.expr)?;

                    return Ok(cx.insert_expr(ExprKind::StructFieldAccessGeneric {
                        lhs,
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

                let lhs = expr_value(cx, hir.expr)?;

                return Ok(cx.insert_expr(ExprKind::TupleFieldAccess { lhs, index }));
            }
        }

        Err(cx.error(CompileErrorKind::BadFieldAccess))
    }

    /// Assemble a unary expression.
    #[instrument]
    fn unary<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: &hir::ExprUnary<'hir>) -> Result<Slot> {
        // NB: special unary expressions.
        if let ast::UnOp::BorrowRef { .. } = hir.op {
            return Err(cx.error(CompileErrorKind::UnsupportedRef));
        }

        if let (ast::UnOp::Neg(..), hir::ExprKind::Lit(ast::Lit::Number(n))) =
            (hir.op, hir.expr.kind)
        {
            match n.resolve(resolve_context!(cx.q))? {
                ast::Number::Float(n) => {
                    return Ok(cx.insert_expr(ExprKind::Store {
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

                    return Ok(cx.insert_expr(ExprKind::Store {
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

        let expr = expr_value(cx, hir.expr)?;

        Ok(cx.insert_expr(ExprKind::Unary { op, expr }))
    }

    /// Assemble a binary expression.
    #[instrument]
    fn binary<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: &hir::ExprBinary<'hir>) -> Result<Slot> {
        if hir.op.is_assign() {
            let lhs = expr_value(cx, hir.lhs)?;
            let rhs = expr_value(cx, hir.rhs)?;

            return Ok(cx.insert_expr(ExprKind::BinaryAssign {
                lhs,
                op: hir.op,
                rhs,
            }));
        }

        if hir.op.is_conditional() {
            let lhs_scope = cx.scopes.push_branch(cx.span, Some(cx.scope))?;
            let lhs = cx.with_scope(lhs_scope, |cx| expr_value(cx, hir.lhs))?;
            cx.scopes.pop(cx.span, lhs_scope)?;

            let rhs_scope = cx.scopes.push_branch(cx.span, Some(cx.scope))?;
            let rhs = cx.with_scope(rhs_scope, |cx| expr_value(cx, hir.rhs))?;
            cx.scopes.pop(cx.span, rhs_scope)?;

            return Ok(cx.insert_expr(ExprKind::BinaryConditional {
                lhs,
                op: hir.op,
                rhs,
            }));
        }

        let lhs = expr_value(cx, hir.lhs)?;
        let rhs = expr(cx, hir.rhs, rhs_needs_of(&hir.op))?;

        return Ok(cx.insert_expr(ExprKind::Binary {
            lhs,
            op: hir.op,
            rhs,
        }));

        /// Get the need of the right-hand side operator from the type of the operator.
        fn rhs_needs_of(op: &ast::BinOp) -> Needs {
            match op {
                ast::BinOp::Is(..) | ast::BinOp::IsNot(..) => Needs::Type,
                _ => Needs::Value,
            }
        }
    }

    /// Assemble an index expression
    #[instrument]
    fn index<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: &hir::ExprIndex<'hir>) -> Result<Slot> {
        let target = expr_value(cx, hir.target)?;
        let index = expr_value(cx, hir.index)?;
        Ok(cx.insert_expr(ExprKind::Index { target, index }))
    }

    /// A block expression.
    #[instrument]
    fn expr_block<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: &hir::ExprBlock<'hir>) -> Result<Slot> {
        cx.with_span(hir.block.span, |cx| {
            if let hir::ExprBlockKind::Default = hir.kind {
                let scope = cx.scopes.push(cx.span, Some(cx.scope))?;
                let expr = cx.with_scope(scope, |cx| block(cx, hir.block))?;
                cx.scopes.pop(cx.span, scope)?;
                return Ok(expr);
            }

            let item = cx.q.item_for(hir.block)?;
            let meta = cx.lookup_meta(hir.block.span, item.item)?;

            let expr = match (hir.kind, &meta.kind) {
                (hir::ExprBlockKind::Async, PrivMetaKind::AsyncBlock { captures, .. }) => {
                    let captures = captures.as_ref();

                    let args = iter!(cx; captures, |ident| {
                        let (binding, address) = cx.scopes.lookup(
                            Location::new(cx.source_id, cx.span),
                            cx.scope,
                            cx.q.visitor,
                            &ident.ident,
                        )?;

                        cx.insert_expr(ExprKind::Binding {
                            binding,
                            address,
                        })
                    });

                    let hash = cx.q.pool.item_type_hash(meta.item_meta.item);

                    cx.insert_expr(ExprKind::CallHash { hash, args })
                }
                (hir::ExprBlockKind::Const, PrivMetaKind::Const { const_value: value }) => {
                    const_value(cx, value)?
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

    /// Assemble a return expression.
    #[instrument]
    fn return_<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: Option<&'hir hir::Expr<'hir>>) -> Result<Slot> {
        let kind = match hir {
            Some(hir) => ExprKind::Return {
                expr: expr_value(cx, hir)?,
            },
            None => ExprKind::Return {
                expr: cx.insert_expr(ExprKind::Store {
                    value: InstValue::Unit,
                }),
            },
        };

        Ok(cx.insert_expr(kind))
    }

    /// Assemble a yield expression.
    #[instrument]
    fn yield_<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: Option<&'hir hir::Expr>) -> Result<Slot> {
        let expr = match hir {
            Some(hir) => Some(expr_value(cx, hir)?),
            None => None,
        };

        Ok(cx.insert_expr(ExprKind::Yield { expr }))
    }

    /// Assemble an await expression.
    #[instrument]
    fn await_<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: &'hir hir::Expr) -> Result<Slot> {
        let expr = expr_value(cx, hir)?;
        Ok(cx.insert_expr(ExprKind::Await { expr }))
    }

    /// Assemble a try expression.
    #[instrument]
    fn try_<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: &'hir hir::Expr) -> Result<Slot> {
        let expr = expr_value(cx, hir)?;
        Ok(cx.insert_expr(ExprKind::Try { expr }))
    }

    /// Assemble a closure.
    #[instrument]
    fn closure<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: &hir::ExprClosure<'hir>) -> Result<Slot> {
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

                cx.insert_expr(ExprKind::Binding {
                    binding,
                    address,
                })
            });

            ExprKind::Closure { hash, captures }
        };

        Ok(cx.insert_expr(kind))
    }

    /// Construct a literal value.
    #[instrument]
    fn lit<'hir>(cx: &mut Ctxt<'_, 'hir>, ast: &'hir ast::Lit) -> Result<Slot> {
        cx.with_span(ast.span(), |cx| {
            let expr = match ast {
                ast::Lit::Bool(lit) => cx.insert_expr(ExprKind::Store {
                    value: InstValue::Bool(lit.value),
                }),
                ast::Lit::Char(lit) => {
                    let ch = lit.resolve(resolve_context!(cx.q))?;
                    cx.insert_expr(ExprKind::Store {
                        value: InstValue::Char(ch),
                    })
                }
                ast::Lit::ByteStr(lit) => {
                    let b = lit.resolve(resolve_context!(cx.q))?;
                    cx.insert_expr(ExprKind::Bytes {
                        bytes: cx
                            .arena
                            .alloc_bytes(b.as_ref())
                            .map_err(arena_error(cx.span))?,
                    })
                }
                ast::Lit::Str(lit) => {
                    let b = lit.resolve(resolve_context!(cx.q))?;
                    cx.insert_expr(ExprKind::String {
                        string: str!(cx; b.as_ref()),
                    })
                }
                ast::Lit::Byte(lit) => {
                    let b = lit.resolve(resolve_context!(cx.q))?;
                    cx.insert_expr(ExprKind::Store {
                        value: InstValue::Byte(b),
                    })
                }
                ast::Lit::Number(number) => lit_number(cx, number)?,
            };

            Ok(expr)
        })
    }

    /// An object expression
    #[instrument]
    fn object<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: &hir::ExprObject<'hir>) -> Result<Slot> {
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
                        Some(hir) => expr_value(cx, hir)?,
                        None => match assignment.key {
                            hir::ObjectKey::Path(hir) => path(cx, hir, Needs::Value)?,
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
            Some(hir) => path(cx, hir, Needs::Type)?.map(cx, |cx, kind| {
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
                let slot = cx
                    .q
                    .unit
                    .new_static_object_keys_iter(cx.span, keys.iter().map(|(s, _)| s.as_str()))?;

                cx.insert_expr(ExprKind::Struct {
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
    fn tuple<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: &hir::ExprSeq<'hir>) -> Result<Slot> {
        let items = iter!(cx; hir.items, |hir| expr_value(cx, hir)?);
        Ok(cx.insert_expr(ExprKind::Tuple { items }))
    }

    /// Assemble a vector expression.
    #[instrument]
    fn vec<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: &hir::ExprSeq<'hir>) -> Result<Slot> {
        let items = iter!(cx; hir.items, |hir| expr_value(cx, hir)?);
        Ok(cx.insert_expr(ExprKind::Vec { items }))
    }

    /// Assemble a range expression.
    #[instrument]
    fn range<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: &hir::ExprRange<'hir>) -> Result<Slot> {
        let limits = match hir.limits {
            hir::ExprRangeLimits::HalfOpen => InstRangeLimits::HalfOpen,
            hir::ExprRangeLimits::Closed => InstRangeLimits::Closed,
        };

        let from = ExprKind::Option {
            value: match hir.from {
                Some(hir) => Some(expr_value(cx, hir)?),
                None => None,
            },
        };
        let from = cx.insert_expr(from);

        let to = ExprKind::Option {
            value: match hir.to {
                Some(hir) => Some(expr_value(cx, hir)?),
                None => None,
            },
        };
        let to = cx.insert_expr(to);

        Ok(cx.insert_expr(ExprKind::Range { from, limits, to }))
    }

    /// Convert a literal number into an expression.
    #[instrument]
    fn lit_number<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: &ast::LitNumber) -> Result<Slot> {
        let number = hir.resolve(resolve_context!(cx.q))?;

        match number {
            ast::Number::Float(float) => Ok(cx.insert_expr(ExprKind::Store {
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

                Ok(cx.insert_expr(ExprKind::Store {
                    value: InstValue::Integer(n),
                }))
            }
        }
    }
}

/// Compile a constant value into an expression.
#[instrument]
fn const_value<'hir>(cx: &mut Ctxt<'_, 'hir>, value: &ConstValue) -> Result<Slot> {
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
            string: str!(cx; s.as_str()),
        },
        ConstValue::StaticString(ref s) => ExprKind::String {
            string: str!(cx; s.as_ref()),
        },
        ConstValue::Bytes(ref b) => ExprKind::Bytes {
            bytes: cx
                .arena
                .alloc_bytes(b.as_ref())
                .map_err(arena_error(cx.span))?,
        },
        ConstValue::Option(ref option) => ExprKind::Option {
            value: match option.as_deref() {
                Some(value) => Some(const_value(cx, value)?),
                None => None,
            },
        },
        ConstValue::Vec(ref vec) => {
            let args = iter!(cx; vec, |value| const_value(cx, &value)?);
            ExprKind::Vec { items: args }
        }
        ConstValue::Tuple(ref tuple) => {
            let args = iter!(cx; tuple.iter(), |value| const_value(cx, value)?);
            ExprKind::Tuple { items: args }
        }
        ConstValue::Object(ref object) => {
            let mut entries = object.iter().collect::<vec::Vec<_>>();
            entries.sort_by_key(|k| k.0);

            let exprs = iter!(cx; entries.iter(), |(_, value)| const_value(cx, value)?);

            let slot =
                cx.q.unit
                    .new_static_object_keys_iter(cx.span, entries.iter().map(|e| e.0))?;

            ExprKind::Struct {
                kind: ExprStructKind::Anonymous { slot },
                exprs,
            }
        }
    };

    Ok(cx.insert_expr(kind))
}

/// Assemble a pattern.
fn pat<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: &hir::Pat<'hir>) -> Result<Slot> {
    let mut removed = HashMap::new();
    return pat(cx, hir, &mut removed);

    #[instrument]
    fn pat<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        hir: &hir::Pat<'hir>,
        removed: &mut HashMap<Name, Span>,
    ) -> Result<Slot> {
        return cx.with_span(hir.span(), |cx| {
            let slot = match hir.kind {
                hir::PatKind::PatPath(hir) => pat_binding(cx, hir, removed)?,
                hir::PatKind::PatIgnore => cx.insert_pat(PatKind::Ignore, &[]),
                hir::PatKind::PatLit(hir) => {
                    let lit = expr_value(cx, hir)?;
                    cx.insert_pat(PatKind::Lit { lit }, &[])
                }
                hir::PatKind::PatVec(hir) => {
                    let items = iter!(cx; hir.items, |hir| pat(cx, hir, removed)?);

                    let mut outputs = Vec::new();

                    for slot in items.iter().copied() {
                        outputs.extend(cx.scopes.pat(cx.span, slot)?.0.outputs.iter().copied());
                    }

                    cx.insert_pat(
                        PatKind::Vec {
                            items,
                            is_open: hir.is_open,
                        },
                        iter!(cx; outputs),
                    )
                }
                hir::PatKind::PatTuple(hir) => pat_tuple(cx, hir, removed)?,
                hir::PatKind::PatObject(hir) => pat_object(cx, hir, removed)?,
            };

            Ok(slot)
        });
    }

    /// A path that has been evaluated.
    #[derive(Debug, Clone, Copy)]
    enum PatPath<'hir> {
        /// An identifier as a pattern.
        Name { name: &'hir str },
        /// A meta item as a pattern.
        Meta { meta: &'hir PrivMeta },
    }

    #[instrument]
    fn pat_path<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: &hir::Path<'hir>) -> Result<PatPath<'hir>> {
        let span = hir.span();

        let named = cx.convert_path(hir)?;

        if let Some(meta) = cx.try_lookup_meta(span, named.item)? {
            return Ok(PatPath::Meta {
                meta: alloc!(cx; meta),
            });
        }

        if let Some(ident) = hir.try_as_ident() {
            let name = ident.resolve(resolve_context!(cx.q))?;
            return Ok(PatPath::Name {
                name: str!(cx; name),
            });
        }

        Err(CompileError::new(
            span,
            CompileErrorKind::MissingItem {
                item: cx.q.pool.item(named.item).to_owned(),
            },
        ))
    }

    /// Assemble a tuple pattern.
    #[instrument]
    fn pat_tuple<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        hir: &hir::PatItems<'hir>,
        removed: &mut HashMap<Name, Span>,
    ) -> Result<Slot> {
        let path = match hir.path {
            Some(hir) => Some(pat_path(cx, hir)?),
            None => None,
        };

        let kind = match path {
            Some(PatPath::Meta { meta }) => {
                let (args, type_match) = match tuple_match_for(cx, meta) {
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

        let patterns = iter!(cx; hir.items, |hir| pat(cx, hir, removed)?);

        let mut outputs = Vec::new();

        for slot in patterns.iter().copied() {
            outputs.extend(cx.scopes.pat(cx.span, slot)?.0.outputs.iter().copied());
        }

        Ok(cx.insert_pat(
            PatKind::Tuple {
                kind,
                patterns,
                is_open: hir.is_open,
            },
            iter!(cx; outputs),
        ))
    }

    /// Assemble a pattern object.
    #[instrument]
    fn pat_object<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        hir: &hir::PatObject<'hir>,
        removed: &mut HashMap<Name, Span>,
    ) -> Result<Slot> {
        let path = match hir.path {
            Some(hir) => Some(pat_path(cx, hir)?),
            None => None,
        };

        let mut keys_dup = HashMap::new();
        let mut keys = Vec::with_capacity(hir.bindings.len());

        for binding in hir.bindings {
            let key = match binding.key {
                hir::ObjectKey::Path(hir) => match pat_path(cx, hir)? {
                    PatPath::Name { name } => Cow::Borrowed(name),
                    _ => {
                        return Err(cx.error(CompileErrorKind::UnsupportedPattern));
                    }
                },
                hir::ObjectKey::LitStr(lit) => lit.resolve(resolve_context!(cx.q))?,
            };

            if let Some(existing) = keys_dup.insert(key.clone().into_owned(), binding.span) {
                return Err(cx.error(CompileErrorKind::DuplicateObjectKey {
                    existing,
                    object: binding.span,
                }));
            }

            keys.push(key.into_owned());
        }

        let kind = match path {
            Some(PatPath::Meta { meta }) => {
                let (st, type_match) = match struct_match_for(cx, meta) {
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
                        let field = str!(cx; key.as_str());
                        (field, hash)
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
            if let Some(hir) = binding.pat {
                pat(cx, hir, removed)?
            } else {
                pat_object_key(cx, binding.key, removed)?
            }
        });

        let mut outputs = Vec::new();

        for slot in patterns.iter().copied() {
            outputs.extend(cx.scopes.pat(cx.span, slot)?.0.outputs.iter().copied());
        }

        Ok(cx.insert_pat(PatKind::Object { kind, patterns }, iter!(cx; outputs)))
    }

    /// Assemble a binding pattern which is *just* a variable captured from an object.
    #[instrument]
    fn pat_object_key<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        hir: &hir::ObjectKey<'hir>,
        removed: &mut HashMap<Name, Span>,
    ) -> Result<Slot> {
        let slot = match *hir {
            hir::ObjectKey::LitStr(..) => {
                return Err(cx.error(CompileErrorKind::UnsupportedPattern))
            }
            hir::ObjectKey::Path(hir) => pat_binding(cx, hir, removed)?,
        };

        Ok(slot)
    }

    /// Handle the binding of a path.
    #[instrument]
    fn pat_binding<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        hir: &hir::Path<'hir>,
        removed: &mut HashMap<Name, Span>,
    ) -> Result<Slot> {
        let slot = match pat_path(cx, hir)? {
            PatPath::Name { name } => {
                let expr = cx.insert_expr(ExprKind::Empty);

                {
                    let name = cx.scopes.name(name);
                    let replaced = cx.scopes.declare(cx.span, cx.scope, name, expr)?;

                    if replaced.is_some() {
                        if let Some(span) = removed.insert(name, cx.span) {
                            return Err(cx.error(CompileErrorKind::DuplicateBinding {
                                previous_span: span,
                            }));
                        }
                    }
                }

                cx.insert_pat(PatKind::Name { name }, iter!(cx; [expr]))
            }
            PatPath::Meta { meta } => cx.insert_pat(PatKind::Meta { meta }, &[]),
        };

        Ok(slot)
    }
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

/// Bind the pattern in the *current* scope.
#[instrument]
fn bind_pat<'hir>(cx: &mut Ctxt<'_, 'hir>, pat: Slot, expr: Slot) -> Result<BoundPat<'hir>> {
    let (&pat, _) = cx.scopes.pat(cx.span, pat)?;

    return cx.with_span(pat.span, |cx| match pat.kind {
        PatKind::Ignore => Ok(cx.bound_pat(BoundPatKind::Irrefutable)),
        PatKind::Lit { lit } => bind_pat_lit(cx, lit, expr),
        PatKind::Name { name } => Ok(cx.bound_pat(BoundPatKind::Expr { name, expr })),
        PatKind::Meta { meta } => {
            let type_match = match tuple_match_for(cx, meta) {
                Some((args, inst)) if args == 0 => inst,
                _ => return Err(cx.error(CompileErrorKind::UnsupportedPattern)),
            };

            Ok(cx.bound_pat(BoundPatKind::TypedSequence {
                type_match,
                expr,
                items: &[],
            }))
        }
        PatKind::Vec { items, is_open } => bind_pat_vec(cx, items, is_open, expr),
        PatKind::Tuple {
            kind,
            patterns,
            is_open,
        } => bind_pat_tuple(cx, kind, patterns, is_open, expr),
        PatKind::Object { kind, patterns } => bind_pat_object(cx, kind, patterns, expr),
    });

    /// Bind a literal pattern.
    #[instrument]
    fn bind_pat_lit<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        lit: Slot,
        expr: Slot,
    ) -> Result<BoundPat<'hir>> {
        {
            let (&lit, _) = cx.scopes.expr(cx.span, lit)?;
            let (&expr, _) = cx.scopes.expr(cx.span, expr)?;

            // Match irrefutable patterns.
            match (lit.kind, expr.kind) {
                (ExprKind::Store { value: a }, ExprKind::Store { value: b }) if a == b => {
                    return Ok(cx.bound_pat(BoundPatKind::Irrefutable));
                }
                (ExprKind::String { string: a }, ExprKind::String { string: b }) if a == b => {
                    return Ok(cx.bound_pat(BoundPatKind::Irrefutable));
                }
                (ExprKind::Bytes { bytes: a }, ExprKind::Bytes { bytes: b }) if a == b => {
                    return Ok(cx.bound_pat(BoundPatKind::Irrefutable));
                }
                _ => {}
            }
        }

        Ok(cx.bound_pat(BoundPatKind::Lit { lit, expr }))
    }

    /// Bind a vector pattern.
    #[instrument]
    fn bind_pat_vec<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        patterns: &'hir [Slot],
        is_open: bool,
        expr: Slot,
    ) -> Result<BoundPat<'hir>> {
        // Try a simpler form of pattern matching through syntactical reassignment.
        match cx.scopes.expr(cx.span, expr)?.0.kind {
            ExprKind::Vec { items: expr_items }
                if expr_items.len() == patterns.len()
                    || expr_items.len() >= patterns.len() && is_open =>
            {
                let items = iter!(cx; patterns.iter().copied().zip(expr_items.iter().copied()), |(pat, expr)| {
                    bind_pat(cx, pat, expr)?
                });

                return Ok(cx.bound_pat(BoundPatKind::IrrefutableSequence { items }));
            }
            _ => {}
        }

        let address = cx.allocator.array_address();

        cx.allocator.alloc_array_items(patterns.len());

        // NB we bind the arguments in reverse to allow for higher elements
        // in the array to be freed up.
        let items = iter!(cx; patterns.iter().rev(), |pat| {
            cx.allocator.free_array_item(cx.span)?;

            let expr = todo!();

            bind_pat(
                cx,
                *pat,
                expr,
            )?
        });

        Ok(cx.bound_pat(BoundPatKind::Vec {
            expr,
            is_open,
            items,
        }))
    }

    /// Bind a tuple pattern.
    #[instrument]
    fn bind_pat_tuple<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        kind: PatTupleKind,
        patterns: &'hir [Slot],
        is_open: bool,
        expr: Slot,
    ) -> Result<BoundPat<'hir>> {
        match kind {
            PatTupleKind::Typed { type_match } => {
                let items = iter!(cx; patterns.iter().enumerate().rev(), |(index, pat)| {
                    let expr = cx.insert_expr(ExprKind::TupleFieldAccess {
                        lhs: expr,
                        index,
                    });

                    bind_pat(cx, *pat, expr)?
                });

                Ok(cx.bound_pat(BoundPatKind::TypedSequence {
                    type_match,
                    expr,
                    items,
                }))
            }
            PatTupleKind::Anonymous => {
                // Try a simpler form of pattern matching through syntactical
                // reassignment.
                match cx.scopes.expr(cx.span, expr)?.0.kind {
                    ExprKind::Tuple { items: tuple_items }
                        if tuple_items.len() == patterns.len()
                            || is_open && tuple_items.len() >= patterns.len() =>
                    {
                        let items = iter!(cx; patterns.iter().zip(tuple_items), |(pat, expr)| {
                            bind_pat(cx, *pat, *expr)?
                        });

                        return Ok(cx.bound_pat(BoundPatKind::IrrefutableSequence { items }));
                    }
                    _ => {}
                }

                cx.allocator.alloc_array_items(patterns.len());

                // NB we bind the arguments in reverse to allow for higher elements
                // in the array to be freed up for subsequent bindings.
                let items = iter!(cx; patterns.iter().rev(), |pat| {
                    cx.allocator.free_array_item(cx.span)?;
                    let expr = todo!();
                    bind_pat(cx, *pat, expr)?
                });

                Ok(cx.bound_pat(BoundPatKind::AnonymousTuple {
                    expr,
                    is_open,
                    items,
                }))
            }
        }
    }

    /// Set up binding for pattern objects.
    #[instrument]
    fn bind_pat_object<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        kind: PatObjectKind<'hir>,
        patterns: &'hir [Slot],
        expr: Slot,
    ) -> Result<BoundPat<'hir>> {
        match kind {
            PatObjectKind::Typed { type_match, keys } => {
                let items = iter!(cx; keys.iter().zip(patterns), |(&(field, hash), pat)| {
                    let expr = cx.insert_expr(ExprKind::StructFieldAccess {
                        lhs: expr,
                        field,
                        hash,
                    });

                    bind_pat(cx, *pat, expr)?
                });

                Ok(cx.bound_pat(BoundPatKind::TypedSequence {
                    type_match,
                    expr,
                    items,
                }))
            }
            PatObjectKind::Anonymous { slot, is_open } => {
                // Try a simpler form of pattern matching through syntactical
                // reassignment.
                match cx.scopes.expr(cx.span, expr)?.0.kind {
                    ExprKind::Struct {
                        kind: ExprStructKind::Anonymous { slot: expr_slot },
                        exprs,
                    } if object_keys_match(&cx.q.unit, expr_slot, slot, is_open)
                        .unwrap_or_default() =>
                    {
                        let items = iter!(cx; patterns.iter().zip(exprs), |(pat, expr)| {
                            bind_pat(cx, *pat, *expr)?
                        });

                        return Ok(cx.bound_pat(BoundPatKind::IrrefutableSequence { items }));
                    }
                    _ => {}
                }

                let address = cx.allocator.array_index();

                cx.allocator.alloc_array_items(patterns.len());

                // NB we bind the arguments in reverse to allow for higher elements
                // in the array to be freed up for subsequent bindings.
                let items = iter!(cx; patterns.iter().rev(), |pat| {
                    cx.allocator.free_array_item(cx.span)?;
                    let expr = todo!();
                    bind_pat(cx, *pat, expr)?
                });

                Ok(cx.bound_pat(BoundPatKind::AnonymousObject {
                    expr,
                    slot,
                    is_open,
                    items,
                }))
            }
        }
    }

    /// Test if object keys match.
    fn object_keys_match(
        unit: &UnitBuilder,
        from_slot: usize,
        to_slot: usize,
        is_open: bool,
    ) -> Option<bool> {
        let from_keys = unit.get_static_object_keys(from_slot)?;
        let to_keys = unit.get_static_object_keys(to_slot)?;

        let mut from_keys = from_keys.iter().cloned().collect::<HashSet<_>>();

        for key in to_keys {
            if !from_keys.remove(key) {
                return Some(false);
            }
        }

        Some(is_open || from_keys.is_empty())
    }
}

/// Coerce an aerna allocation error.
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

#[derive(Debug, Clone, Copy)]
enum TypeMatch {
    BuiltIn {
        type_check: TypeCheck,
    },
    Type {
        type_hash: Hash,
    },
    Variant {
        variant_hash: Hash,
        enum_hash: Hash,
        index: usize,
    },
}

/// Construct the appropriate match instruction for the given [PrivMeta].
#[instrument]
fn tuple_match_for<'hir>(cx: &mut Ctxt<'_, 'hir>, meta: &PrivMeta) -> Option<(usize, TypeMatch)> {
    match &meta.kind {
        PrivMetaKind::Struct {
            type_hash,
            variant: PrivVariantMeta::Unit,
            ..
        } => Some((
            0,
            TypeMatch::Type {
                type_hash: *type_hash,
            },
        )),
        PrivMetaKind::Struct {
            type_hash,
            variant: PrivVariantMeta::Tuple(tuple),
            ..
        } => Some((
            tuple.args,
            TypeMatch::Type {
                type_hash: *type_hash,
            },
        )),
        PrivMetaKind::Variant {
            enum_hash,
            type_hash,
            index,
            variant,
            ..
        } => {
            let args = match variant {
                PrivVariantMeta::Tuple(tuple) => tuple.args,
                PrivVariantMeta::Unit => 0,
                _ => return None,
            };

            let struct_match = if let Some(type_check) = cx.context.type_check_for(*type_hash) {
                TypeMatch::BuiltIn { type_check }
            } else {
                TypeMatch::Variant {
                    enum_hash: *enum_hash,
                    variant_hash: *type_hash,
                    index: *index,
                }
            };

            Some((args, struct_match))
        }
        meta => {
            tracing::trace!(?meta, "no match");
            None
        }
    }
}

/// Construct the appropriate match instruction for the given [PrivMeta].
#[instrument]
fn struct_match_for<'a>(
    cx: &mut Ctxt<'_, '_>,
    meta: &'a PrivMeta,
) -> Option<(&'a PrivStructMeta, TypeMatch)> {
    Some(match &meta.kind {
        PrivMetaKind::Struct {
            type_hash,
            variant: PrivVariantMeta::Struct(st),
            ..
        } => (
            st,
            TypeMatch::Type {
                type_hash: *type_hash,
            },
        ),
        PrivMetaKind::Variant {
            type_hash,
            enum_hash,
            index,
            variant: PrivVariantMeta::Struct(st),
            ..
        } => {
            let type_check = if let Some(type_check) = cx.context.type_check_for(*type_hash) {
                TypeMatch::BuiltIn { type_check }
            } else {
                TypeMatch::Variant {
                    variant_hash: *type_hash,
                    enum_hash: *enum_hash,
                    index: *index,
                }
            };

            (st, type_check)
        }
        _ => {
            return None;
        }
    })
}
