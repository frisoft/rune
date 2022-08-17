use std::borrow::Cow;
use std::cell::Cell;
use std::fmt;
use std::mem;
use std::num::NonZeroUsize;
use std::ops::Neg as _;
use std::vec;

use num::ToPrimitive;
use rune_macros::__instrument_hir as instrument;

use crate::arena::{Arena, ArenaAllocError, ArenaWriteSliceOutOfBounds};
use crate::ast::{self, Span, Spanned};
use crate::collections::{BTreeSet, HashMap, HashSet, VecDeque};
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
    Address, AssemblyInst as Inst, ConstValue, FormatSpec, InstAssignOp, InstOp, InstRangeLimits,
    InstTarget, InstValue, InstVariant, PanicReason, Protocol, TypeCheck,
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

#[derive(Debug, Clone, Copy, Default)]
enum CtxtState {
    #[default]
    Default,
    Unreachable {
        reported: bool,
    },
}

#[derive(Clone)]
enum SlottedValue<T, Id> {
    Value(T),
    Moved(Id),
}

#[derive(Clone)]
struct Slotted<T, Id> {
    storage: Vec<SlottedValue<T, Id>>,
}

impl<T, Id> Slotted<T, Id>
where
    Id: Index,
{
    fn new() -> Self {
        Self {
            storage: Vec::new(),
        }
    }

    /// Move one value from one slot to another.
    fn move_value(&mut self, from: Id, to: Id) {
        if let Some(from_slot) = self.storage.get_mut(from.index()) {
            *from_slot = SlottedValue::Moved(to);
        }
    }

    /// Follow the given id.
    fn follow(&self, id: Id) -> Option<Id>
    where
        Id: Copy,
    {
        let mut id = match self.storage.get(id.index())? {
            SlottedValue::Moved(id) => *id,
            _ => return None,
        };

        while let SlottedValue::Moved(new_id) = self.storage.get(id.index())? {
            id = *new_id;
        }

        Some(id)
    }

    /// Get a slotted value.
    fn get(&self, id: Id) -> Option<&T>
    where
        Id: Copy,
    {
        let mut current = id;

        loop {
            match self.storage.get(current.index())? {
                SlottedValue::Moved(id) => {
                    current = *id;
                }
                SlottedValue::Value(value) => {
                    return Some(value);
                }
            }
        }
    }

    /// Get a mutable value.
    fn get_mut(&mut self, id: Id) -> Option<&mut T>
    where
        Id: Copy,
    {
        let id = self.follow(id).unwrap_or(id);

        match self.storage.get_mut(id.index())? {
            SlottedValue::Value(value) => Some(value),
            SlottedValue::Moved(_) => None,
        }
    }

    /// Get the next id that will be inserted.
    fn next_id(&self) -> Option<Id> {
        Id::new(self.storage.len())
    }

    /// Insert the given value.
    fn insert(&mut self, value: T) {
        self.storage.push(SlottedValue::Value(value));
    }
}

impl<T, Id> IntoIterator for Slotted<T, Id> {
    type Item = T;
    type IntoIter = IntoIter<T, Id>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            it: self.storage.into_iter(),
        }
    }
}

struct IntoIter<T, Id> {
    it: vec::IntoIter<SlottedValue<T, Id>>,
}

impl<T, Id> Iterator for IntoIter<T, Id> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let SlottedValue::Value(value) = self.it.next()? {
                return Some(value);
            }
        }
    }
}

trait Index: Sized {
    /// Construct a new index.
    fn new(value: usize) -> Option<Self>;

    /// Get the usize index of an index.
    fn index(&self) -> usize;
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
    scopes: Scopes,
    /// State of code generation.
    state: CtxtState,
    /// A memory allocator.
    allocator: Allocator,
    /// Set of useless slots.
    useless: BTreeSet<ExprId>,
    /// All defined patterns and the expressions they are to be bound to.
    patterns: Slotted<(Pat<'hir>, UsedExprId), PatId>,
    /// Keeping track of every slot.
    expressions: Slotted<Expr<'hir>, ExprId>,
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
            useless: BTreeSet::new(),
            patterns: Slotted::new(),
            expressions: Slotted::new(),
        }
    }

    /// Access the scopes built by the context.
    pub(crate) fn into_allocator(self) -> Allocator {
        self.allocator
    }

    /// Pass an address for a slot.
    fn slot_pass(&mut self, slot: ExprId, address: AssemblyAddress) -> Result<ExprOutput> {
        Ok(self.expr_mut(slot)?.pass(address))
    }

    /// Restore an address for a slot.
    fn slot_restore(&mut self, slot: ExprId, previous: ExprOutput) -> Result<()> {
        self.expr_mut(slot)?.restore(previous);
        Ok(())
    }

    /// Allocate a new output address slotted by the given expression address.
    fn expr_output(&mut self, slot: ExprId) -> Result<AssemblyAddress> {
        let expr = self.expr_mut(slot)?;

        if let ExprOutput::Allocated(address) | ExprOutput::Passed(address) = expr.address {
            tracing::trace!(?slot, ?address, "using existing address");
            return Ok(address);
        }

        if matches!(expr.address, ExprOutput::Freed) {
            return Err(self.msg(format_args!("trying to use freed address on slot {slot:?}")));
        }

        let address = self.allocator.alloc();
        tracing::trace!(?slot, ?address, "allocating address");
        self.expr_mut(slot)?.address = ExprOutput::Allocated(address);
        Ok(address)
    }

    /// Free the implicit address associated with the given slot.
    fn free_expr(&mut self, slot: ExprId) -> Result<()> {
        let storage = self.expr_mut(slot)?;

        // NB: We make freeing optional instead of a hard error, since we might
        // call this function even if an address has not been allocated for the
        // slot.
        if let Some(address) = storage.address.free() {
            tracing::trace!(?storage.slot, ?address, "freeing address");
            self.allocator.free(self.span, address)?;
        }

        Ok(())
    }

    /// Load slot.
    fn expr(&self, id: ExprId) -> Result<&Expr<'hir>> {
        if let Some(expr) = self.expressions.get(id) {
            return Ok(expr);
        }

        Err(CompileError::msg(
            self.span,
            format_args!("failed to look up expression by id {id:?}"),
        ))
    }

    /// Load slot mutably.
    fn expr_mut(&mut self, id: ExprId) -> Result<&mut Expr<'hir>> {
        if let Some(expr) = self.expressions.get_mut(id) {
            return Ok(expr);
        }

        Err(CompileError::msg(
            self.span,
            format_args!("failed to look up expression by id {id:?}"),
        ))
    }

    /// Declare a variable.
    fn declare(&mut self, name: Name, id: ExprId) -> Result<Option<ExprId>> {
        let id = self.expressions.follow(id).unwrap_or(id);
        self.scopes.declare(self.span, self.scope, name, id)
    }

    /// Add a slot user with custom use kind.
    fn add_expr_user(&mut self, used_id: UsedExprId, user: ExprUser) -> Result<()> {
        let span = self.span;
        self.expr_mut(used_id.id())?
            .insert_use(span, user, used_id.use_kind())?;
        // We can now remove this slot from the set of useless slots.
        self.useless.remove(&used_id.id());
        Ok(())
    }

    /// Take a single slot user.
    fn take_expr_user(&mut self, expr: UsedExprId, user: ExprUser) -> Result<UseSummary> {
        let expr_id = self.expressions.follow(expr.id()).unwrap_or(expr.id());

        let span = self.span;
        let slot_mut = self.expr_mut(expr_id)?;

        let sealed = slot_mut.seal();
        slot_mut.remove_use(span, user)?;

        let summary = UseSummary {
            current: slot_mut.uses.len(),
            branch: slot_mut.branch,
            total: sealed.total,
            pending: slot_mut.pending,
        };

        tracing::trace!(?expr, ?user, ?summary, "took slot user");
        Ok(summary)
    }

    /// Remove a slot user.
    fn remove_expr_user(&mut self, used_id: UsedExprId, user: ExprUser) -> Result<()> {
        let span = self.span;
        let storage = self.expr_mut(used_id.id())?;
        storage.remove_use(span, user)?;

        if storage.uses.is_empty() {
            self.useless.insert(used_id.id());
        }

        Ok(())
    }

    /// Retain all expressions referenced by the given expression.
    fn retain_expr_kind(&mut self, kind: ExprKind<'hir>, user: ExprUser) -> Result<()> {
        walk_expr(self, kind, |cx, used_id| cx.add_expr_user(used_id, user))
    }

    /// Release all expressions referenced by the given expression.
    fn release_expr_kind(&mut self, kind: ExprKind<'hir>, user: ExprUser) -> Result<()> {
        walk_expr(self, kind, |cx, used_id| cx.remove_expr_user(used_id, user))
    }

    /// Retain all expressions associated with a pattern.
    fn retain_pat_kind(&mut self, kind: PatKind<'hir>, user: ExprUser) -> Result<()> {
        walk_pat(self, kind, |cx, used_id| cx.add_expr_user(used_id, user))
    }

    /// Release all expressions referenced associated with a pattern.
    fn release_pat_kind(&mut self, kind: PatKind<'hir>, user: ExprUser) -> Result<()> {
        walk_pat(self, kind, |cx, used_id| cx.remove_expr_user(used_id, user))
    }

    /// Get the id of the next expression that will be inserted.
    fn next_expr_id(&mut self) -> Result<ExprId> {
        match self.expressions.next_id() {
            Some(id) => Ok(id),
            None => Err(CompileError::msg(self.span, "ran out of expression ids")),
        }
    }

    /// Insert a expression.
    fn insert_expr(&mut self, kind: ExprKind<'hir>) -> Result<UsedExprId> {
        let id = self.next_expr_id()?;
        self.retain_expr_kind(kind, ExprUser::Expr(id))?;
        // Mark current statement inserted as potentially useless to sweep it up later.
        self.useless.insert(id);

        self.expressions.insert(Expr {
            span: self.span,
            slot: id,
            kind,
            uses: BTreeSet::new(),
            address: ExprOutput::Empty,
            branch: false,
            pending: true,
            sealed: Cell::new(None),
        });

        Ok(UsedExprId::new(id))
    }

    /// Insert an expression with an assembly address.
    fn insert_expr_with_address(
        &mut self,
        kind: ExprKind<'hir>,
        address: AssemblyAddress,
    ) -> Result<UsedExprId> {
        let used_id = self.insert_expr(kind)?;
        self.expr_mut(used_id.id())?.address = ExprOutput::Allocated(address);
        Ok(used_id)
    }

    /// Insert a pattern.
    fn insert_pat(&mut self, kind: PatKind<'hir>, expr: UsedExprId) -> Result<PatId> {
        let id = match self.patterns.next_id() {
            Some(id) => id,
            None => {
                return Err(CompileError::msg(
                    self.span,
                    "ran out of pattern identifiers",
                ))
            }
        };

        let pat = Pat::new(self.span, kind);
        self.retain_pat_kind(pat.kind, ExprUser::Pat(id))?;
        self.add_expr_user(expr, ExprUser::Pat(id))?;
        self.patterns.insert((pat, expr));
        Ok(id)
    }

    /// Build a nested pattern.
    fn build_pat(&self, kind: PatKind<'hir>) -> Pat<'hir> {
        Pat::new(self.span, kind)
    }

    /// Get a reference to a pattern by id.
    fn pat(&self, pat: PatId) -> Result<&Pat<'hir>> {
        match self.patterns.get(pat) {
            Some((pat, _)) => Ok(pat),
            None => {
                return Err(CompileError::msg(
                    self.span,
                    format_args!("missing pattern by id {pat:?}"),
                ))
            }
        }
    }

    /// Get a reference to the expression that is associated with a pattern.
    fn pat_expr(&self, pat: PatId) -> Result<UsedExprId> {
        match self.patterns.get(pat) {
            Some(&(_, expr)) => Ok(expr),
            None => {
                return Err(CompileError::msg(
                    self.span,
                    format_args!("missing pattern by id {pat:?}"),
                ))
            }
        }
    }

    /// Get a mutable reference to a pattern by id.
    fn pat_mut(&mut self, pat: PatId) -> Result<&mut Pat<'hir>> {
        match self.patterns.get_mut(pat) {
            Some((pat, _)) => Ok(pat),
            None => {
                return Err(CompileError::msg(
                    self.span,
                    format_args!("missing pattern by id {pat:?}"),
                ))
            }
        }
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

    /// Construct a compile error associated with the current scope.
    fn msg<M>(&self, message: M) -> CompileError
    where
        M: fmt::Display,
    {
        CompileError::msg(self.span, message)
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

    /// Allocate an array of addresses.
    fn array<I>(&mut self, this: ExprId, expressions: I) -> Result<AssemblyAddress>
    where
        I: IntoIterator<Item = UsedExprId>,
    {
        let address = self.allocator.array_address();

        for id in expressions {
            let output = self.allocator.array_address();
            self.array_address(id, Some(this), output)?;
            self.allocator.alloc_array_item();
        }

        Ok(address)
    }

    /// Get all addresses associated with a normal expression that has multiple inputs and a single output.
    fn addresses<const N: usize>(
        &mut self,
        this: ExprId,
        expressions: [UsedExprId; N],
    ) -> Result<([AssemblyAddress; N], AssemblyAddress)> {
        let outputs = self.expression_addresses(Some(ExprUser::Expr(this)), expressions)?;
        let output = self.expr_output(this)?;
        Ok((outputs, output))
    }

    /// Allocate a collection of slots as addresses.
    fn expression_addresses<const N: usize>(
        &mut self,
        this: Option<ExprUser>,
        expressions: [UsedExprId; N],
    ) -> Result<[AssemblyAddress; N]> {
        let mut inputs = [mem::MaybeUninit::uninit(); N];

        for (expr, input) in expressions.into_iter().zip(&mut inputs) {
            let (address, used) = self.address(expr, this)?;
            input.write((address, used, expr));
        }

        // SAFETY: we just initialized the array above.
        let inputs = unsafe { array_assume_init(inputs) };
        let mut outputs = [mem::MaybeUninit::uninit(); N];

        for (o, (address, used, expr)) in outputs.iter_mut().zip(inputs) {
            if used.is_last() {
                self.free_expr(expr.id())?;
            }

            o.write(address);
        }

        // SAFETY: we just initialized the array above.
        let outputs = unsafe { array_assume_init(outputs) };
        Ok(outputs)
    }

    /// Helper function to assemble and allocate a single address. If `output`
    /// is specified, this function will ensure that the output of the
    /// expression ends up in it and `output` is taken.
    fn address(
        &mut self,
        used_id: UsedExprId,
        user: Option<ExprUser>,
    ) -> Result<(AssemblyAddress, UseSummary)> {
        let summary = if let Some(user) = user {
            self.take_expr_user(used_id, user)?
        } else {
            self.expr_mut(used_id.id())?.use_summary()
        };

        if summary.is_pending() {
            asm(self, used_id.id())?;
        }

        Ok((self.expr_output(used_id.id())?, summary))
    }

    /// Ensure that the given expression is available at the given address.
    fn array_address(
        &mut self,
        used_id: UsedExprId,
        user: Option<ExprId>,
        output: AssemblyAddress,
    ) -> Result<()> {
        match self.expr(used_id.id())?.kind {
            ExprKind::Address => {}
            _ => {
                let used = if let Some(user) = user {
                    self.take_expr_user(used_id, ExprUser::Expr(user))?
                } else {
                    self.expr_mut(used_id.id())?.use_summary()
                };

                if used.is_pending() {
                    if used.is_only() {
                        asm_to_output(self, used_id.id(), output)?;
                        return Ok(());
                    }

                    asm(self, used_id.id())?;
                }
            }
        }

        let address = self.expr_output(used_id.id())?;
        self.push_with_comment(
            Inst::Copy { address, output },
            format_args!("copy into {output:?}"),
        );
        Ok(())
    }
}

/// Helper function to initialized an array of [mem::MaybeUninit].
pub unsafe fn array_assume_init<T, const N: usize>(array: [mem::MaybeUninit<T>; N]) -> [T; N] {
    let ret = (&array as *const _ as *const [T; N]).read();
    ret
}

/// Assemble an async block.
#[instrument]
pub(crate) fn closure_from_block<'hir>(
    cx: &mut Ctxt<'_, 'hir>,
    _: &hir::Block<'hir>,
    _: &[CaptureMeta],
) -> Result<()> {
    todo!()
}

/// Assemble the body of a closure function.
#[instrument]
pub(crate) fn closure_from_expr_closure<'hir>(
    cx: &mut Ctxt<'_, 'hir>,
    _: &hir::ExprClosure<'hir>,
    _: &[CaptureMeta],
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
    let mut arguments = Vec::new();
    let scope = cx.scopes.push(cx.span, None)?;

    let slot = cx.with_scope(scope, |cx| {
        let mut first = true;

        for arg in hir.args {
            // Allocate the first couple of slots.
            let address = cx.allocator.alloc();

            match *arg {
                hir::FnArg::SelfValue(span) => {
                    if !instance_fn || !first {
                        return Err(CompileError::new(span, CompileErrorKind::UnsupportedSelf));
                    }

                    let name = cx.scopes.name(SELF);
                    let expr = cx.insert_expr_with_address(ExprKind::Empty, address)?;
                    cx.declare(name, expr.id())?;
                }
                hir::FnArg::Pat(hir) => {
                    let expr = cx.insert_expr_with_address(ExprKind::Empty, address)?;
                    arguments.push(pat(cx, hir, expr)?);
                }
            }

            first = false;
        }

        block(cx, hir.body)
    })?;

    cx.scopes.pop(span, scope)?;

    // Bind all patterns.
    bind_pats(cx)?;

    for pat in arguments {
        let panic_label = cx.new_label("argument_panic");

        if let PatOutcome::Refutable = asm_pat(cx, pat, panic_label)? {
            let end = cx.new_label("argument_end");
            cx.push(Inst::Jump { label: end });
            cx.label(panic_label)?;
            cx.push(Inst::Panic {
                reason: PanicReason::UnmatchedPattern,
            });
            cx.label(end)?;
        }

        let pat_expr = cx.pat_expr(pat)?;
        let [_] = cx.expression_addresses(Some(ExprUser::Pat(pat)), [pat_expr])?;
    }

    let [address] = cx.expression_addresses(None, [slot])?;
    cx.push(Inst::Return { address });
    Ok(())
}

/// Bind all patterns available in the context.
#[instrument]
fn bind_pats(cx: &mut Ctxt<'_, '_>) -> Result<()> {
    let patterns = cx.patterns.clone();

    for (index, (pat, slot)) in patterns.into_iter().enumerate() {
        tracing::trace!("binding pattern {pat:?}");

        let kind = bind_pat(cx, pat.span, pat.kind, slot)?;

        if let Some(pat) = PatId::new(index) {
            let previous = mem::replace(&mut cx.pat_mut(pat)?.kind, kind);
            cx.release_pat_kind(previous, ExprUser::Pat(pat))?;
            cx.retain_pat_kind(kind, ExprUser::Pat(pat))?;
        }
    }

    return Ok(());

    /// Bind the pattern in the *current* scope.
    #[instrument]
    fn bind_pat<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        span: Span,
        kind: PatKind<'hir>,
        expr: UsedExprId,
    ) -> Result<PatKind<'hir>> {
        return cx.with_span(span, |cx| match kind {
            PatKind::Ignore => Ok(PatKind::Irrefutable),
            PatKind::Lit { lit } => bind_pat_lit(cx, lit, expr),
            PatKind::Ghost { ghost_expr } => {
                // Here we make the ghost expression reference the bound
                // expression so it can be used.
                let snapshot = cx.expr_mut(ghost_expr.id())?.take_uses();
                cx.expr_mut(expr.id())?.import_uses(snapshot);
                cx.expressions.move_value(ghost_expr.id(), expr.id());
                Ok(PatKind::Irrefutable)
            }
            PatKind::Meta { meta } => {
                let type_match = match tuple_match_for(cx, meta) {
                    Some((args, inst)) if args == 0 => inst,
                    _ => return Err(cx.error(CompileErrorKind::UnsupportedPattern)),
                };

                Ok(PatKind::TypedSequence {
                    type_match,
                    expr,
                    items: &[],
                })
            }
            PatKind::Vec { items, is_open } => bind_pat_vec(cx, items, is_open, expr),
            PatKind::Tuple {
                kind,
                items,
                is_open,
            } => bind_pat_tuple(cx, kind, items, is_open, expr),
            PatKind::Object { kind, items } => bind_pat_object(cx, kind, items, expr),
            _ => Err(cx.msg("trying to bind pattern which has already been bound")),
        });
    }

    /// Bind a literal pattern.
    #[instrument]
    fn bind_pat_lit<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        lit: UsedExprId,
        expr: UsedExprId,
    ) -> Result<PatKind<'hir>> {
        // Match irrefutable patterns.
        match (cx.expr(lit.id())?.kind, cx.expr(expr.id())?.kind) {
            (ExprKind::Store { value: a }, ExprKind::Store { value: b }) if a == b => {
                return Ok(PatKind::Irrefutable);
            }
            (ExprKind::String { string: a }, ExprKind::String { string: b }) if a == b => {
                return Ok(PatKind::Irrefutable);
            }
            (ExprKind::Bytes { bytes: a }, ExprKind::Bytes { bytes: b }) if a == b => {
                return Ok(PatKind::Irrefutable);
            }
            _ => {}
        }

        Ok(PatKind::BoundLit { lit, expr })
    }

    /// Bind a vector pattern.
    #[instrument]
    fn bind_pat_vec<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        items: &'hir [Pat<'hir>],
        is_open: bool,
        expr: UsedExprId,
    ) -> Result<PatKind<'hir>> {
        // Try a simpler form of pattern matching through syntactical reassignment.
        match cx.expr(expr.id())?.kind {
            ExprKind::Vec { items: expr_items }
                if expr_items.len() == items.len()
                    || expr_items.len() >= items.len() && is_open =>
            {
                let items = iter!(cx; items.iter().copied().zip(expr_items.iter().copied()), |(pat, expr)| {
                    let kind = bind_pat(cx, pat.span, pat.kind, expr)?;
                    cx.insert_pat(kind, expr)?
                });

                return Ok(PatKind::IrrefutableSequence { items });
            }
            _ => {}
        }

        let address = cx.allocator.array_address();
        cx.allocator.alloc_array_items(items.len());

        // NB we bind the arguments in reverse to allow for higher elements
        // in the array to be freed up.

        let items = iter!(cx; items.iter().copied().rev(), |pat| {
            cx.allocator.free_array_item(cx.span)?;
            let expr =
                cx.insert_expr_with_address(ExprKind::Address, cx.allocator.array_address())?;
            let kind = bind_pat(cx, pat.span, pat.kind, expr)?;
            cx.insert_pat(kind, expr)?
        });

        Ok(PatKind::BoundVec {
            address,
            expr,
            is_open,
            items,
        })
    }

    /// Bind a tuple pattern.
    #[instrument]
    fn bind_pat_tuple<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        kind: PatTupleKind,
        items: &'hir [Pat<'hir>],
        is_open: bool,
        expr: UsedExprId,
    ) -> Result<PatKind<'hir>> {
        match kind {
            PatTupleKind::Typed { type_match } => {
                let items = iter!(cx; items.iter().copied().enumerate().rev(), |(index, pat)| {
                    let expr = cx.insert_expr(ExprKind::TupleFieldAccess { lhs: expr, index })?;
                    let kind = bind_pat(cx, pat.span, pat.kind, expr)?;
                    cx.insert_pat(kind, expr)?
                });

                Ok(PatKind::TypedSequence {
                    type_match,
                    expr,
                    items,
                })
            }
            PatTupleKind::Anonymous => {
                // Try a simpler form of pattern matching through syntactical
                // reassignment.
                match cx.expr(expr.id())?.kind {
                    ExprKind::Tuple { items: tuple_items }
                        if tuple_items.len() == items.len()
                            || is_open && tuple_items.len() >= items.len() =>
                    {
                        let items = iter!(cx; items.iter().copied().zip(tuple_items), |(pat, expr)| {
                            let kind = bind_pat(cx, pat.span, pat.kind, *expr)?;
                            cx.insert_pat(kind, *expr)?
                        });

                        return Ok(PatKind::IrrefutableSequence { items });
                    }
                    _ => {}
                }

                let address = cx.allocator.array_address();
                cx.allocator.alloc_array_items(items.len());

                // NB we bind the arguments in reverse to allow for higher elements
                // in the array to be freed up for subsequent bindings.
                let items = iter!(cx; items.iter().rev().copied(), |pat| {
                    cx.allocator.free_array_item(cx.span)?;
                    let expr = cx.insert_expr_with_address(
                        ExprKind::Address,
                        cx.allocator.array_address(),
                    )?;
                    let kind = bind_pat(cx, pat.span, pat.kind, expr)?;
                    cx.insert_pat(kind, expr)?
                });

                Ok(PatKind::AnonymousTuple {
                    expr,
                    is_open,
                    items,
                    address,
                })
            }
        }
    }

    /// Set up binding for pattern objects.
    #[instrument]
    fn bind_pat_object<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        kind: PatObjectKind<'hir>,
        items: &'hir [Pat<'hir>],
        expr: UsedExprId,
    ) -> Result<PatKind<'hir>> {
        match kind {
            PatObjectKind::Typed { type_match, keys } => {
                let items = iter!(cx; keys.iter().zip(items.iter().copied()), |(&(field, hash), pat)| {
                    let expr = cx.insert_expr(ExprKind::StructFieldAccess {
                        lhs: expr,
                        field,
                        hash,
                    })?;

                    let kind = bind_pat(cx, pat.span, pat.kind, expr)?;
                    cx.insert_pat(kind, expr)?
                });

                Ok(PatKind::TypedSequence {
                    type_match,
                    expr,
                    items,
                })
            }
            PatObjectKind::Anonymous { slot, is_open } => {
                // Try a simpler form of pattern matching through syntactical
                // reassignment.
                match cx.expr(expr.id())?.kind {
                    ExprKind::Struct {
                        kind: ExprStructKind::Anonymous { slot: expr_slot },
                        exprs,
                    } if object_keys_match(&cx.q.unit, expr_slot, slot, is_open)
                        .unwrap_or_default() =>
                    {
                        let items = iter!(cx; items.iter().copied().zip(exprs.iter().copied()), |(pat, expr)| {
                            let kind = bind_pat(cx, pat.span, pat.kind, expr)?;
                            cx.insert_pat(kind, expr)?
                        });

                        return Ok(PatKind::IrrefutableSequence { items });
                    }
                    _ => {}
                }

                let address = cx.allocator.array_address();
                cx.allocator.alloc_array_items(items.len());

                // NB we bind the arguments in reverse to allow for higher elements
                // in the array to be freed up for subsequent bindings.
                let items = iter!(cx; items.iter().rev().copied(), |pat| {
                    cx.allocator.free_array_item(cx.span)?;
                    let expr = cx.insert_expr_with_address(
                        ExprKind::Address,
                        cx.allocator.array_address(),
                    )?;
                    let kind = bind_pat(cx, pat.span, pat.kind, expr)?;
                    cx.insert_pat(kind, expr)?
                });

                Ok(PatKind::AnonymousObject {
                    address,
                    expr,
                    slot,
                    is_open,
                    items,
                })
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

/// Debug a slot to stdout.
fn debug_stdout(cx: &mut Ctxt<'_, '_>, slot: ExprId) -> Result<()> {
    let out = std::io::stdout();
    let mut out = out.lock();

    let mut task = Task {
        visited: HashSet::new(),
        queue: VecDeque::from_iter([Job::Slot(slot)]),
    };

    while let Some(job) = task.queue.pop_front() {
        match job {
            Job::Pat(pat) => {
                debug_pat(&mut out, cx, pat, &mut task)?;
            }
            Job::Slot(slot) => {
                debug_expr(&mut out, cx, slot, &mut task)?;
            }
        }
    }

    return Ok(());

    struct Task {
        visited: HashSet<ExprId>,
        queue: VecDeque<Job>,
    }

    impl Task {
        fn expr_id(&mut self, cx: &mut Ctxt<'_, '_>, id: ExprId) -> Result<String> {
            let id = cx.expressions.follow(id).unwrap_or(id);

            if self.visited.insert(id) {
                self.queue.push_back(Job::Slot(id));
            }

            Ok(format!("${}", id.index()))
        }

        fn used_expr_id(&mut self, cx: &mut Ctxt<'_, '_>, id_use: UsedExprId) -> Result<String> {
            let id = cx.expressions.follow(id_use.id()).unwrap_or(id_use.id());

            if self.visited.insert(id) {
                self.queue.push_back(Job::Slot(id));
            }

            match id_use.kind {
                UsedExprIdKind::Default => Ok(format!("${}", id.index())),
                UsedExprIdKind::Binding { binding, use_kind } => {
                    let name = cx.scopes.name_to_string(cx.span, binding.name)?;
                    Ok(format!("${} {{{name:?}, {use_kind:?}}}", id.index()))
                }
            }
        }

        fn pat_id(&mut self, _: &mut Ctxt<'_, '_>, pat: PatId) -> Result<String> {
            self.queue.push_back(Job::Pat(pat));
            Ok(format!("Pat${}", pat.index()))
        }

        fn nested_pat(&mut self, _: &mut Ctxt<'_, '_>, pat: Pat<'_>) -> Result<String> {
            let kind = pat.kind;
            Ok(format!("{kind:?}"))
        }
    }

    enum Job {
        Pat(PatId),
        Slot(ExprId),
    }

    /// Debug a slot.
    fn debug_expr<'hir, O>(
        o: &mut O,
        cx: &mut Ctxt<'_, 'hir>,
        id: ExprId,
        task: &mut Task,
    ) -> Result<()>
    where
        O: std::io::Write,
    {
        let expr = cx.expr(id)?;
        let span = expr.span;
        let err = write_err(span);

        if let Some(to) = cx.expressions.follow(id) {
            writeln!(o, "${} => ${};", id.index(), to.index()).map_err(err)?;
            return Ok(());
        }

        let summary = expr.use_temporary_summary();
        let uses = format_uses(expr, span)?;
        let kind = expr.kind;

        macro_rules! field {
            ($pad:literal, lit, $var:expr) => {
                writeln!(o, "{}{} = {:?},", $pad, stringify!($var), $var).map_err(err)?;
            };

            ($pad:literal, binding, $var:expr) => {{
                let name = cx.scopes.name_to_string(span, $var.name)?;
                writeln!(o, "{}{} = {name:?},", $pad, stringify!($var)).map_err(err)?;
            }};

            ($pad:literal, condition, $var:expr) => {
                match $var {
                    LoopCondition::Forever => {
                        writeln!(o, "{}{} = Forever,", $pad, stringify!($var)).map_err(err)?;
                    }
                    LoopCondition::Condition { pat } => {
                        let pat = task.pat_id(cx, pat)?;

                        writeln!(
                            o,
                            "{}{} = Condition {{ pat = {pat} }},",
                            $pad,
                            stringify!($var)
                        )
                        .map_err(err)?;
                    }
                    LoopCondition::Iterator { iter, pat } => {
                        let iter = task.used_expr_id(cx, iter)?;
                        let pat = task.pat_id(cx, pat)?;

                        writeln!(
                            o,
                            "{}{} = Iterator {{ iter = {iter}, pat = {pat} }},",
                            $pad,
                            stringify!($var)
                        )
                        .map_err(err)?;
                    }
                }
            };

            ($pad:literal, $fn:ident, $var:expr) => {{
                let name = task.$fn(cx, $var)?;
                writeln!(o, "{}{} = {name},", $pad, stringify!($var)).map_err(err)?;
            }};

            ($pad:literal, [array $fn:ident], $var:expr) => {
                let mut it = IntoIterator::into_iter($var);

                let first = it.next();

                if let Some(&value) = first {
                    writeln!(o, "{}{} = [", $pad, stringify!($var)).map_err(err)?;
                    let name = task.$fn(cx, value)?;
                    writeln!(o, "{}  {name},", $pad).map_err(err)?;

                    for &value in it {
                        let name = task.$fn(cx, value)?;
                        writeln!(o, "{}  {name},", $pad).map_err(err)?;
                    }

                    writeln!(o, "{}],", $pad).map_err(err)?;
                } else {
                    writeln!(o, "{}{} = [],", $pad, stringify!($var)).map_err(err)?;
                }
            };

            ($pad:literal, [option $fn:ident], $var:expr) => {
                if let Some(value) = $var {
                    let name = task.$fn(cx, value)?;
                    writeln!(o, "{}{} = Some({name}),", $pad, stringify!($var)).map_err(err)?;
                } else {
                    writeln!(o, "{}{} = None,", $pad, stringify!($var)).map_err(err)?;
                }
            };
        }

        macro_rules! variant {
            ($name:ident) => {{
                writeln!(o, "${} = {} summary = {summary:?}, uses = {uses};", id.index(), stringify!($name)).map_err(err)?;
            }};

            ($name:ident { $($what:tt $field:ident),* }) => {{
                writeln!(o, "${} = {} summary = {summary:?}, uses = {uses} {{", id.index(), stringify!($name)).map_err(err)?;
                $(field!("  ", $what, $field);)*
                writeln!(o, "}};").map_err(err)?;
            }};
        }

        macro_rules! matches {
            ($expr:expr, { $($name:ident $({ $($what:tt $field:ident),* $(,)? })?),* $(,)? }) => {{
                match $expr {
                    $(
                        ExprKind::$name { $($($field),*)* } => {
                            variant!($name $({ $($what $field),* })*)
                        }
                    )*
                }
            }};
        }

        matches! {
            kind, {
                Empty,
                Address,
                TupleFieldAccess { used_expr_id lhs, lit index },
                StructFieldAccess { used_expr_id lhs, lit field, lit hash },
                StructFieldAccessGeneric { used_expr_id lhs, lit hash, lit generics },
                Assign { used_expr_id lhs, used_expr_id rhs },
                AssignStructField { used_expr_id lhs, lit field, used_expr_id rhs },
                AssignTupleField { used_expr_id lhs, lit index, used_expr_id rhs },
                Block { [array used_expr_id] statements, [option used_expr_id] tail },
                Let { pat_id pat },
                Store { lit value },
                Bytes { lit bytes },
                String { lit string },
                Unary { lit op, used_expr_id expr },
                BinaryAssign { used_expr_id lhs, lit op, used_expr_id rhs },
                BinaryConditional { used_expr_id lhs, lit op, used_expr_id rhs },
                Binary { used_expr_id lhs, lit op, used_expr_id rhs },
                Index { used_expr_id target, used_expr_id index },
                Meta { lit meta, lit needs, lit named },
                Struct { lit kind, [array used_expr_id] exprs },
                Tuple { [array used_expr_id] items },
                Vec { [array used_expr_id] items },
                Range { used_expr_id from, lit limits, used_expr_id to },
                Option { [option used_expr_id] value },
                CallHash { lit hash, [array used_expr_id] args },
                CallInstance { used_expr_id lhs, lit hash, [array used_expr_id] args },
                CallExpr { used_expr_id expr, [array used_expr_id] args },
                Yield { [option used_expr_id] expr },
                Await { used_expr_id expr },
                Return { used_expr_id expr },
                Try { used_expr_id expr },
                Function { lit hash },
                Closure { lit hash, [array used_expr_id] captures },
                Loop { condition condition, used_expr_id body, lit start, lit end },
                Break { [option used_expr_id] value, lit label, expr_id loop_slot },
                Continue { lit label },
                StringConcat { [array used_expr_id] exprs },
                Format { lit spec, used_expr_id expr },
            }
        }

        Ok(())
    }

    /// Debug a patter.
    fn debug_pat<'hir, O>(
        o: &mut O,
        cx: &mut Ctxt<'_, 'hir>,
        pat: PatId,
        task: &mut Task,
    ) -> Result<()>
    where
        O: std::io::Write,
    {
        let id = pat.index();
        let expr = cx.pat_expr(pat)?;
        let pat = cx.pat(pat)?;
        let span = pat.span;
        let kind = pat.kind;
        let err = write_err(span);
        let expr = task.used_expr_id(cx, expr)?;

        macro_rules! field {
            ($pad:literal, lit, $var:expr) => {
                writeln!(o, "{}{} = {:?},", $pad, stringify!($var), $var).map_err(err)?;
            };

            ($pad:literal, $fn:ident, $var:expr) => {{
                let name = task.$fn(cx, $var)?;
                writeln!(o, "{}{} = {name},", $pad, stringify!($var)).map_err(err)?;
            }};

            ($pad:literal, [array $fn:ident], $var:expr) => {
                let mut it = IntoIterator::into_iter($var);

                let first = it.next();

                if let Some(&value) = first {
                    writeln!(o, "{}{} = [", $pad, stringify!($var)).map_err(err)?;
                    let name = task.$fn(cx, value)?;
                    writeln!(o, "{}  {name},", $pad).map_err(err)?;

                    for &value in it {
                        let name = task.$fn(cx, value)?;
                        writeln!(o, "{}  {name},", $pad).map_err(err)?;
                    }

                    writeln!(o, "{}],", $pad).map_err(err)?;
                } else {
                    writeln!(o, "{}{} = [],", $pad, stringify!($var)).map_err(err)?;
                }
            };
        }

        macro_rules! variant {
            ($name:ident) => {{
                writeln!(o, "Pat${id} = {name} <= {expr};", name = stringify!($name)).map_err(err)?;
            }};

            ($name:ident { $($what:tt $field:ident),* }) => {{
                writeln!(o, "Pat${id} = {name} <= {expr} {{", name = stringify!($name)).map_err(err)?;
                $(field!("  ", $what, $field);)*
                writeln!(o, "}};").map_err(err)?;
            }};
        }

        macro_rules! matches {
            ($expr:expr, { $($name:ident $({ $($what:tt $field:ident),* $(,)? })?),* $(,)? }) => {{
                match $expr {
                    $(
                        PatKind::$name { $($($field),*)* } => {
                            variant!($name $({ $($what $field),* })*)
                        }
                    )*
                }
            }};
        }

        matches! {
            kind, {
                Ignore,
                Lit { used_expr_id lit },
                Ghost { used_expr_id ghost_expr },
                Meta { lit meta },
                Vec {
                    [array nested_pat] items,
                    lit is_open,
                },
                Tuple {
                    lit kind,
                    [array nested_pat] items,
                    lit is_open,
                },
                Object {
                    lit kind,
                    [array nested_pat] items,
                },
                Irrefutable,
                IrrefutableSequence { [array pat_id] items },
                BoundLit { used_expr_id lit, used_expr_id expr },
                BoundVec {
                    lit address,
                    used_expr_id expr,
                    lit is_open,
                    [array pat_id] items,
                },
                AnonymousTuple {
                    lit address,
                    used_expr_id expr,
                    lit is_open,
                    [array pat_id] items,
                },
                AnonymousObject {
                    lit address,
                    used_expr_id expr,
                    lit slot,
                    lit is_open,
                    [array pat_id] items,
                },
                TypedSequence {
                    lit type_match,
                    used_expr_id expr,
                    [array pat_id] items,
                },
            }
        }

        Ok(())
    }

    fn format_uses(expr: &Expr<'_>, span: Span) -> Result<String> {
        use std::fmt::Write as _;

        let mut users = String::new();
        users.push('{');

        let mut first = true;

        let mut it = expr.uses.iter().peekable();

        while let Some(user) = it.next() {
            if mem::take(&mut first) {
                users.push(' ');
            }

            match user {
                ExprUser::Pat(id) => {
                    write!(users, "Pat${}", id.index()).map_err(write_err(span))?;
                }
                ExprUser::Expr(id) => {
                    write!(users, "${}", id.index()).map_err(write_err(span))?;
                }
            }

            if it.peek().is_some() {
                write!(users, ", ").map_err(write_err(span))?;
            }
        }

        if !first {
            users.push(' ');
        }

        users.push('}');
        Ok(users)
    }

    fn write_err<E>(span: Span) -> impl FnOnce(E) -> CompileError + Copy
    where
        E: fmt::Display,
    {
        move |e| CompileError::msg(span, e)
    }
}

/// The outcome of a pattern being applied.
#[derive(Debug)]
enum PatOutcome {
    Irrefutable,
    Refutable,
}

impl PatOutcome {
    /// Combine this outcome with another.
    fn combine(self, other: Self) -> Self {
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
    /// An expression who's inherent address refers to a static address.
    Address,
    /// An assignment to a binding.
    Assign {
        /// Address to assign to.
        lhs: UsedExprId,
        /// The expression to assign.
        rhs: UsedExprId,
    },
    /// A tuple field access.
    TupleFieldAccess { lhs: UsedExprId, index: usize },
    /// A struct field access where the index is the slot used.
    StructFieldAccess {
        lhs: UsedExprId,
        field: &'hir str,
        hash: Hash,
    },
    StructFieldAccessGeneric {
        lhs: UsedExprId,
        hash: Hash,
        generics: Option<(Span, &'hir [hir::Expr<'hir>])>,
    },
    /// An assignment to a struct field.
    AssignStructField {
        lhs: UsedExprId,
        field: &'hir str,
        rhs: UsedExprId,
    },
    /// An assignment to a tuple field.
    AssignTupleField {
        lhs: UsedExprId,
        index: usize,
        rhs: UsedExprId,
    },
    /// A block with a sequence of statements.
    Block {
        /// Statements in the block.
        statements: &'hir [UsedExprId],
        /// The tail of the block expression.
        tail: Option<UsedExprId>,
    },
    Let {
        /// The assembled pattern.
        pat: PatId,
    },
    /// A literal value.
    Store { value: InstValue },
    /// A byte blob.
    Bytes { bytes: &'hir [u8] },
    /// A string.
    String { string: &'hir str },
    /// A unary expression.
    Unary { op: ExprUnOp, expr: UsedExprId },
    /// A binary assign operation.
    BinaryAssign {
        /// The left-hand side of a binary operation.
        lhs: UsedExprId,
        /// The operator.
        op: ast::BinOp,
        /// The right-hand side of a binary operation.
        rhs: UsedExprId,
    },
    /// A binary conditional operation.
    BinaryConditional {
        /// The left-hand side of a binary operation.
        lhs: UsedExprId,
        /// The operator.
        op: ast::BinOp,
        /// The right-hand side of a binary operation.
        rhs: UsedExprId,
    },
    /// A binary expression.
    Binary {
        /// The left-hand side of a binary operation.
        lhs: UsedExprId,
        /// The operator.
        op: ast::BinOp,
        /// The right-hand side of a binary operation.
        rhs: UsedExprId,
    },
    /// The `<target>[<value>]` operation.
    Index {
        target: UsedExprId,
        index: UsedExprId,
    },
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
        exprs: &'hir [UsedExprId],
    },
    /// An anonymous tuple.
    Tuple { items: &'hir [UsedExprId] },
    /// Allocate a vector.
    Vec { items: &'hir [UsedExprId] },
    /// A range expression.
    Range {
        from: UsedExprId,
        limits: InstRangeLimits,
        to: UsedExprId,
    },
    /// Allocate an optional value.
    Option {
        /// The value to allocate.
        value: Option<UsedExprId>,
    },
    /// Call the given hash.
    CallHash {
        hash: Hash,
        args: &'hir [UsedExprId],
    },
    /// Call the given instance fn.
    CallInstance {
        lhs: UsedExprId,
        hash: Hash,
        args: &'hir [UsedExprId],
    },
    /// Call the given expression.
    CallExpr {
        expr: UsedExprId,
        args: &'hir [UsedExprId],
    },
    /// Yield the given value.
    Yield {
        /// Yield the given expression.
        expr: Option<UsedExprId>,
    },
    /// Perform an await operation.
    Await {
        /// The expression to await.
        expr: UsedExprId,
    },
    /// Return a kind.
    Return { expr: UsedExprId },
    /// Perform a try operation.
    Try {
        /// The expression to try.
        expr: UsedExprId,
    },
    /// Load a function address.
    Function { hash: Hash },
    /// Load a closure.
    Closure {
        /// The hash of the closure function to load.
        hash: Hash,
        /// Captures to this closure.
        captures: &'hir [UsedExprId],
    },
    Loop {
        /// The condition to advance the loop.
        condition: LoopCondition,
        /// The body of the loop.
        body: UsedExprId,
        /// The start label to use.
        start: Label,
        /// The end label.
        end: Label,
    },
    /// A break expression.
    Break {
        /// The value to break with.
        value: Option<UsedExprId>,
        /// End label to jump to.
        label: Label,
        /// The loop slot to break to.
        loop_slot: ExprId,
    },
    /// A continue expression.
    Continue {
        /// The label to jump to.
        label: Label,
    },
    /// A concatenation of a sequence of expressions.
    StringConcat { exprs: &'hir [UsedExprId] },
    /// A value format.
    Format {
        spec: &'hir FormatSpec,
        expr: UsedExprId,
    },
}

/// A loop condition.
#[derive(Debug, Clone, Copy)]
enum LoopCondition {
    /// A forever loop condition.
    Forever,
    /// A pattern condition.
    Condition { pat: PatId },
    /// An iterator condition.
    Iterator { iter: UsedExprId, pat: PatId },
}

/// An expression that can be assembled.
#[must_use]
#[derive(Debug, Clone, Copy)]
struct Pat<'hir> {
    /// The span of the assembled expression.
    span: Span,
    /// The kind of the expression.
    kind: PatKind<'hir>,
}

impl<'hir> Pat<'hir> {
    /// Construct a new pattern.
    const fn new(span: Span, kind: PatKind<'hir>) -> Self {
        Self { span, kind }
    }
}

/// The kind of a pattern.
#[derive(Debug, Clone, Copy)]
enum PatKind<'hir> {
    /// An ignore pattern.
    Ignore,
    /// A literal value.
    Lit { lit: UsedExprId },
    /// An unbound ghost expression which is empty until it is bound.
    Ghost { ghost_expr: UsedExprId },
    /// A meta binding.
    Meta { meta: &'hir PrivMeta },
    /// A vector pattern.
    Vec {
        items: &'hir [Pat<'hir>],
        is_open: bool,
    },
    /// A tuple pattern.
    Tuple {
        kind: PatTupleKind,
        items: &'hir [Pat<'hir>],
        is_open: bool,
    },
    /// An object pattern.
    Object {
        kind: PatObjectKind<'hir>,
        items: &'hir [Pat<'hir>],
    },
    /// Bound irrefutable pattern.
    Irrefutable,
    /// An irrefutable sequence.
    IrrefutableSequence { items: &'hir [PatId] },
    /// A bound literal pattern.
    BoundLit { lit: UsedExprId, expr: UsedExprId },
    /// A bound vector.
    BoundVec {
        address: AssemblyAddress,
        expr: UsedExprId,
        is_open: bool,
        items: &'hir [PatId],
    },
    /// A bound anonymous tuple.
    AnonymousTuple {
        address: AssemblyAddress,
        expr: UsedExprId,
        is_open: bool,
        items: &'hir [PatId],
    },
    /// A bound anonymous object.
    AnonymousObject {
        address: AssemblyAddress,
        expr: UsedExprId,
        slot: usize,
        is_open: bool,
        items: &'hir [PatId],
    },
    /// A bound typed sequence.
    TypedSequence {
        type_match: TypeMatch,
        expr: UsedExprId,
        items: &'hir [PatId],
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
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
struct ExprId(NonZeroUsize);

impl Index for ExprId {
    /// Construct a new slot.
    fn new(value: usize) -> Option<Self> {
        Some(Self(NonZeroUsize::new(value.wrapping_add(1))?))
    }

    /// Get the index that the slot corresponds to.
    fn index(&self) -> usize {
        self.0.get().wrapping_sub(1)
    }
}

impl fmt::Debug for ExprId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("ExprId").field(&self.index()).finish()
    }
}

/// A pattern identifier.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
struct PatId(NonZeroUsize);

impl Index for PatId {
    /// Construct a new pattern identifier.
    fn new(value: usize) -> Option<Self> {
        Some(Self(NonZeroUsize::new(value.wrapping_add(1))?))
    }

    /// Get the index that the pattern identifier corresponds to.
    fn index(&self) -> usize {
        self.0.get().wrapping_sub(1)
    }
}

impl fmt::Debug for PatId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("PatId").field(&self.index()).finish()
    }
}

/// Scopes to use when compiling.
#[derive(Default)]
struct Scopes {
    /// Stack of scopes.
    scopes: slab::Slab<Scope>,
    /// The collection of known names in the scope, so that we can generate the
    /// [BindingName] identifier.
    names_by_id: Vec<Box<str>>,
    /// Names cache.
    names_by_name: HashMap<Box<str>, Name>,
}

impl Scopes {
    /// Construct an empty scope.
    fn new() -> Self {
        Self {
            scopes: slab::Slab::new(),
            names_by_id: Vec::new(),
            names_by_name: HashMap::new(),
        }
    }

    /// Declare a variable with an already known address.
    #[tracing::instrument(skip_all)]
    fn declare(
        &mut self,
        span: Span,
        scope: ScopeId,
        name: Name,
        slot: ExprId,
    ) -> Result<Option<ExprId>> {
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
        loop_slot: ExprId,
    ) -> Result<ScopeId> {
        self.push_inner(
            span,
            parent,
            ControlFlow::Loop(LoopControlFlow {
                label,
                start,
                end,
                loop_slot,
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
    ) -> Result<Option<UsedExprId>> {
        let name = self.name(string);
        tracing::trace!(string = string, name = ?name);

        let mut current = Some(scope);
        let mut use_kind = UseKind::Same;

        while let Some((id, scope)) = current
            .take()
            .and_then(|s| Some((s, self.scopes.get(s.0)?)))
        {
            if let Some(slot) = scope.lookup(name) {
                tracing::trace!("found: {name:?} => {slot:?}");
                let binding = Binding { scope: id, name };
                return Ok(Some(UsedExprId::binding(slot, binding, use_kind)));
            }

            if !matches!(scope.control_flow, ControlFlow::None) {
                use_kind = UseKind::Branch;
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
    ) -> Result<UsedExprId> {
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

impl fmt::Debug for Scopes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        return f
            .debug_struct("Scopes")
            .field("scopes", &ScopesIter(&self.scopes))
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
    loop_slot: ExprId,
}

/// The kind of lookup that was performed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum UseKind {
    /// The variable is found in the same scope as this one.
    Same,
    /// The variable was found outside of a branch.
    Branch,
}

#[derive(Debug, Clone, Default)]
struct Scope {
    parent: Option<ScopeId>,
    names: HashMap<Name, ExprId>,
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
    fn lookup<'data>(&self, name: Name) -> Option<ExprId> {
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

/// Use summary for a slot.
#[derive(Debug, Clone, Copy)]
struct UseSummary {
    /// How many users this value has.
    current: usize,
    /// How many branches that are using this value.
    branch: bool,
    /// The total number of users of an expression.
    total: usize,
    /// If the value is pending construction.
    pending: bool,
}

impl UseSummary {
    /// Test if this is the last user.
    fn is_last(&self) -> bool {
        self.current == 0
    }

    /// If the expression is waiting to be built.
    fn is_pending(&self) -> bool {
        self.pending
    }

    /// Test if expression is still alive.
    fn is_alive(&self) -> bool {
        self.current > 0
    }

    /// If the value is branched.
    fn is_branched(&self) -> bool {
        self.branch
    }

    /// Test if this expression only has one non-branch user which allows for folding it into the current expression.
    fn is_only(&self) -> bool {
        !self.branch && self.total == 1
    }
}

/// Use summary for a slot.
#[derive(Debug, Clone, Copy)]
struct SlotUse {
    /// How many users this value has.
    summary: UseSummary,
    /// The kind of use which is in place.
    use_kind: UseKind,
}

impl SlotUse {
    /// Construct a new slot use.
    const fn new(summary: UseSummary, use_kind: UseKind) -> Self {
        Self { summary, use_kind }
    }

    /// Test if this is the last user.
    #[inline(always)]
    fn is_last(&self) -> bool {
        self.summary.is_last()
    }

    /// If the expression is waiting to be built.
    #[inline(always)]
    fn is_pending(&self) -> bool {
        self.summary.is_pending()
    }
}

/// Data that is calculated once an expression is sealed.
#[derive(Debug, Clone, Copy)]
struct ExprSealed {
    total: usize,
}

/// The output of an expression.
#[derive(Debug, Clone, Copy)]
enum ExprOutput {
    /// Address is empty.
    Empty,
    /// Address is freed and must not be used again.
    Freed,
    /// An allocated address that should be freed here.
    Allocated(AssemblyAddress),
    /// A passed-in address.
    Passed(AssemblyAddress),
}

impl ExprOutput {
    /// Free the current slot address effectively setting it to empty and return
    /// the assembly address that should be freed.
    fn free(&mut self) -> Option<AssemblyAddress> {
        if matches!(self, ExprOutput::Passed(..)) {
            return None;
        }

        if let ExprOutput::Allocated(address) = mem::replace(self, ExprOutput::Freed) {
            Some(address)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum ExprUser {
    /// Pattern user.
    Pat(PatId),
    /// Expression user.
    Expr(ExprId),
}

#[derive(Debug, Clone)]
struct Expr<'hir> {
    /// The span associated with the slot.
    span: Span,
    /// The output address of the expression.
    slot: ExprId,
    /// The kind of the expression.
    kind: ExprKind<'hir>,
    /// The downstream users of this slot.
    uses: BTreeSet<ExprUser>,
    /// The implicit address of the slot.
    address: ExprOutput,
    /// If this expression is used in a branch context which might conditionally
    /// execute. Such use requires that the expression is assembled *before* the
    /// branch.
    branch: bool,
    /// If the slot is pending to be built.
    pending: bool,
    /// Is calculated when the first `take_user` is called.
    sealed: Cell<Option<ExprSealed>>,
}

impl<'hir> Expr<'hir> {
    /// Replace the current slot address temporarily with another address. This
    /// implies that the address is "passed" and shouldn't be freed.
    fn pass(&mut self, address: AssemblyAddress) -> ExprOutput {
        tracing::trace!(?self.slot, ?address, "passing in address");
        mem::replace(&mut self.address, ExprOutput::Passed(address))
    }

    /// Restore a previously passed address to a freed state.
    fn restore(&mut self, previous: ExprOutput) {
        tracing::trace!(?self.slot, ?previous, "restoring passed in address");
        self.address = previous;
    }

    /// Insert a single user of this expression.
    fn insert_use(&mut self, span: Span, user: ExprUser, use_kind: UseKind) -> Result<()> {
        let this = self.slot;
        tracing::trace!(?this, ?user, ?use_kind, "inserting use");

        if self.sealed.get().is_some() {
            return Err(CompileError::msg(
                span,
                format_args!("cannot add user {user:?} to slot {this:?} since it's sealed"),
            ));
        }

        if !self.uses.insert(user) {
            return Err(CompileError::msg(
                span,
                format_args!("cannot add user {user:?} to slot {this:?} since it already exists"),
            ));
        }

        if let UseKind::Branch = use_kind {
            self.branch = true;
        }

        Ok(())
    }

    /// Remove a single user of this expression.
    fn remove_use(&mut self, span: Span, user: ExprUser) -> Result<()> {
        let this = self.slot;
        tracing::trace!(?this, ?user, "removing use");

        if !self.uses.remove(&user) {
            return Err(CompileError::msg(
                span,
                format_args!("{user:?} is not a user of slot {this:?}"),
            ));
        };

        Ok(())
    }

    /// Seal the current slot. This is called automatically when a summary
    /// called for or use is taken.
    fn seal(&self) -> ExprSealed {
        if let Some(value) = self.sealed.get() {
            return value;
        }

        let value = ExprSealed {
            total: self.uses.len(),
        };

        self.sealed.set(Some(value));
        value
    }

    /// Get a use summary for the current slot.
    fn use_summary(&self) -> UseSummary {
        let sealed = self.seal();

        UseSummary {
            current: self.uses.len(),
            branch: self.branch,
            total: sealed.total,
            pending: self.pending,
        }
    }

    /// Get a current non-sealed summary.
    fn use_temporary_summary(&self) -> UseSummary {
        UseSummary {
            current: self.uses.len(),
            branch: self.branch,
            total: self.uses.len(),
            pending: self.pending,
        }
    }

    /// Take and export uses for this expression and seal it.
    fn take_uses(&mut self) -> ExprUseSnapshot {
        let uses = mem::take(&mut self.uses);
        let branch = mem::take(&mut self.branch);

        let _ = self.seal();

        ExprUseSnapshot { uses, branch }
    }

    /// Import uses for an expression.
    fn import_uses(&mut self, snapshot: ExprUseSnapshot) {
        self.branch |= snapshot.branch;

        for user in snapshot.uses {
            self.uses.insert(user);
        }
    }
}

/// A snapshot from the uses of an expression.
#[derive(Debug, Clone)]
struct ExprUseSnapshot {
    uses: BTreeSet<ExprUser>,
    branch: bool,
}

/// Memory allocator.
#[derive(Debug, Clone)]
pub(crate) struct Allocator {
    slots: slab::Slab<()>,
    /// The bottom of the allocated elements.
    bottom: usize,
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
            bottom: 0,
            count: 0,
            array_index: 0,
            array_count: 0,
        }
    }

    /// Translate an assembly address into a stack address.
    #[tracing::instrument(skip_all)]
    pub(crate) fn translate(&self, span: Span, address: AssemblyAddress) -> Result<Address> {
        fn check<T>(span: Span, v: Option<T>, msg: &'static str) -> Result<T> {
            match v {
                Some(v) => Ok(v),
                None => Err(CompileError::msg(span, msg)),
            }
        }

        let slot = match address {
            AssemblyAddress::Bottom(slot) => slot,
            AssemblyAddress::Slot(slot) => {
                check(span, self.bottom.checked_add(slot), "slot overflow")?
            }
            AssemblyAddress::Array(index) => check(
                span,
                self.count.checked_add(index),
                "array index out of bound",
            )?,
        };

        Ok(Address(check(
            span,
            u32::try_from(slot).ok(),
            "slot out of bound",
        )?))
    }

    /// Get the next address that will be generated without consuming it.
    fn next_address(&self) -> AssemblyAddress {
        let slot = self.slots.vacant_key();
        AssemblyAddress::Slot(slot)
    }

    /// Allocate a new bottom assembly address.
    fn alloc_bottom(&mut self) -> AssemblyAddress {
        let address = AssemblyAddress::Bottom(self.bottom);
        self.bottom += 1;
        tracing::trace!(bottom = self.bottom, address = ?address);
        address
    }

    /// Allocate a new assembly address.
    fn alloc(&mut self) -> AssemblyAddress {
        let slot = self.slots.insert(());
        let address = AssemblyAddress::Slot(slot);
        self.count = self.slots.len().max(self.count);
        tracing::trace!(count = self.count, address = ?address);
        address
    }

    /// Free an assembly address.
    fn free(&mut self, span: Span, address: AssemblyAddress) -> Result<()> {
        if let AssemblyAddress::Slot(slot) = address {
            if self.slots.try_remove(slot).is_none() {
                return Err(CompileError::msg(
                    span,
                    format_args!("missing slot to free `{slot}`"),
                ));
            }
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
        self.count
            .saturating_add(self.array_count)
            .saturating_add(self.bottom)
    }
}

/// Test if the expression at the given slot has side effects.
fn has_side_effects(cx: &mut Ctxt<'_, '_>, used_id: UsedExprId) -> Result<bool> {
    match cx.expr(used_id.id())?.kind {
        ExprKind::Empty => Ok(false),
        ExprKind::Address { .. } => Ok(true),
        ExprKind::TupleFieldAccess { .. } => Ok(true),
        ExprKind::StructFieldAccess { .. } => Ok(true),
        ExprKind::StructFieldAccessGeneric { .. } => Ok(true),
        ExprKind::Assign { rhs, .. } => has_side_effects(cx, rhs),
        ExprKind::AssignStructField { .. } => Ok(true),
        ExprKind::AssignTupleField { .. } => Ok(true),
        ExprKind::Block { statements, tail } => {
            for &slot in statements {
                if has_side_effects(cx, slot)? {
                    return Ok(false);
                }
            }

            match tail {
                Some(slot) => has_side_effects(cx, slot),
                None => Ok(false),
            }
        }
        ExprKind::Let { .. } => Ok(true),
        ExprKind::Store { .. } => Ok(false),
        ExprKind::Bytes { .. } => Ok(false),
        ExprKind::String { .. } => Ok(false),
        ExprKind::Unary { .. } => {
            // Validate the unary expression.
            Ok(true)
        }
        ExprKind::BinaryAssign { .. } => Ok(true),
        ExprKind::BinaryConditional { .. } => Ok(true),
        ExprKind::Binary { .. } => Ok(true),
        ExprKind::Index { .. } => Ok(true),
        ExprKind::Meta { .. } => Ok(true),
        ExprKind::Struct { .. } => Ok(true),
        ExprKind::Tuple { items } => {
            for &slot in items {
                if has_side_effects(cx, slot)? {
                    return Ok(true);
                }
            }

            Ok(false)
        }
        ExprKind::Vec { items } => {
            for &slot in items {
                if has_side_effects(cx, slot)? {
                    return Ok(true);
                }
            }

            Ok(false)
        }
        ExprKind::Range { from, to, .. } => {
            if has_side_effects(cx, from)? {
                return Ok(true);
            }

            if has_side_effects(cx, to)? {
                return Ok(true);
            }

            Ok(false)
        }
        ExprKind::Option { value } => match value {
            Some(slot) => has_side_effects(cx, slot),
            None => Ok(false),
        },
        ExprKind::CallHash { .. } => Ok(true),
        ExprKind::CallInstance { .. } => Ok(true),
        ExprKind::CallExpr { .. } => Ok(true),
        ExprKind::Yield { .. } => Ok(true),
        ExprKind::Await { .. } => Ok(true),
        ExprKind::Return { .. } => Ok(true),
        ExprKind::Try { .. } => Ok(true),
        ExprKind::Function { .. } => Ok(true),
        ExprKind::Closure { .. } => Ok(true),
        ExprKind::Loop { .. } => Ok(true),
        ExprKind::Break { .. } => Ok(true),
        ExprKind::Continue { .. } => Ok(true),
        ExprKind::StringConcat { exprs, .. } => {
            for &slot in exprs {
                if has_side_effects(cx, slot)? {
                    return Ok(true);
                }
            }

            Ok(false)
        }
        ExprKind::Format { expr, .. } => has_side_effects(cx, expr),
    }
}

/// Assemble a value with a custom output address.
#[instrument]
fn asm_to_output<'hir>(
    cx: &mut Ctxt<'_, 'hir>,
    this: ExprId,
    output: AssemblyAddress,
) -> Result<ExprOutcome> {
    let previous = cx.slot_pass(this, output)?;
    let result = asm(cx, this);
    cx.slot_restore(this, previous)?;
    result
}

/// Assemble the given address.
#[instrument]
fn asm<'hir>(cx: &mut Ctxt<'_, 'hir>, this: ExprId) -> Result<ExprOutcome> {
    let slot_mut = cx.expr_mut(this)?;

    if !slot_mut.pending {
        return Err(CompileError::msg(
            slot_mut.span,
            format_args!("expression {this:?} is being built more than once"),
        ));
    }

    slot_mut.pending = false;

    let span = slot_mut.span;
    let kind = slot_mut.kind;

    return cx.with_span(span, |cx| {
        match kind {
            ExprKind::Empty => {}
            ExprKind::Address => {
                return Err(cx.msg("cannot assemble address expressions"));
            }
            ExprKind::Block { statements, tail } => {
                asm_block(cx, this, statements, tail)?;
            }
            ExprKind::Let { pat } => asm_let(cx, this, pat)?,
            ExprKind::Store { value } => {
                asm_store(cx, this, value)?;
            }
            ExprKind::Bytes { bytes } => {
                asm_bytes(cx, this, bytes)?;
            }
            ExprKind::String { string } => {
                asm_string(cx, this, string)?;
            }
            ExprKind::Unary { op, expr } => {
                asm_unary(cx, this, op, expr)?;
            }
            ExprKind::BinaryAssign { lhs, op, rhs } => {
                asm_binary_assign(cx, this, lhs, op, rhs)?;
            }
            ExprKind::BinaryConditional { lhs, op, rhs } => {
                asm_binary_conditional(cx, this, lhs, op, rhs)?;
            }
            ExprKind::Binary { lhs, op, rhs } => {
                asm_binary(cx, this, lhs, op, rhs)?;
            }
            ExprKind::Index { target, index } => {
                asm_index(cx, this, target, index)?;
            }
            ExprKind::Meta { meta, needs, named } => {
                asm_meta(cx, this, meta, needs, named)?;
            }
            ExprKind::Struct { kind, exprs } => {
                asm_struct(cx, this, kind, exprs)?;
            }
            ExprKind::Vec { items } => {
                asm_vec(cx, this, items)?;
            }
            ExprKind::Range { from, limits, to } => {
                asm_range(cx, this, from, limits, to)?;
            }
            ExprKind::Tuple { items } => {
                asm_tuple(cx, this, items)?;
            }
            ExprKind::Option { value } => {
                asm_option(cx, this, value)?;
            }
            ExprKind::TupleFieldAccess { lhs, index } => {
                asm_tuple_field_access(cx, this, lhs, index)?;
            }
            ExprKind::StructFieldAccess { lhs, field, .. } => {
                asm_struct_field_access(cx, this, lhs, field)?;
            }
            ExprKind::StructFieldAccessGeneric { .. } => {
                return Err(cx.error(CompileErrorKind::ExpectedExpr));
            }
            ExprKind::Assign { lhs, rhs } => {
                asm_assign(cx, this, lhs, rhs)?;
            }
            ExprKind::AssignStructField { lhs, field, rhs } => {
                asm_assign_struct_field(cx, this, lhs, field, rhs)?;
            }
            ExprKind::AssignTupleField { lhs, index, rhs } => {
                asm_assign_tuple_field(cx, this, lhs, index, rhs)?;
            }
            ExprKind::CallHash { hash, args } => {
                asm_call_hash(cx, this, args, hash)?;
            }
            ExprKind::CallInstance { lhs, hash, args } => {
                asm_call_instance(cx, this, lhs, args, hash)?;
            }
            ExprKind::CallExpr { expr, args } => {
                asm_call_expr(cx, this, expr, args)?;
            }
            ExprKind::Yield { expr } => {
                asm_yield(cx, this, expr)?;
            }
            ExprKind::Await { expr } => {
                asm_await(cx, this, expr)?;
            }
            ExprKind::Return { expr } => {
                asm_return(cx, this, expr)?;
                return Ok(ExprOutcome::Unreachable);
            }
            ExprKind::Try { expr } => {
                asm_try(cx, this, expr)?;
            }
            ExprKind::Function { hash } => {
                asm_function(cx, this, hash)?;
            }
            ExprKind::Closure { hash, captures } => {
                asm_closure(cx, this, captures, hash)?;
            }
            ExprKind::Loop {
                condition,
                body,
                start,
                end,
            } => {
                asm_loop(cx, condition, body, start, end)?;
            }
            ExprKind::Break {
                value,
                label,
                loop_slot,
            } => {
                asm_break(cx, value, label, loop_slot)?;
            }
            ExprKind::Continue { label } => {
                asm_continue(cx, label)?;
            }
            ExprKind::StringConcat { exprs } => {
                asm_string_concat(cx, this, exprs)?;
            }
            ExprKind::Format { spec, expr } => {
                asm_format(cx, this, spec, expr)?;
            }
        }

        Ok(ExprOutcome::Output)
    });

    #[instrument]
    fn asm_block<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        this: ExprId,
        statements: &[UsedExprId],
        tail: Option<UsedExprId>,
    ) -> Result<()> {
        for &id in statements {
            let _ = cx.addresses(this, [id])?;
        }

        if let Some(tail) = tail {
            let ([tail], output) = cx.addresses(this, [tail])?;

            // If tail and output did not end up being allocated to the same
            // address we need to perform a copy - this means that the address
            // used by the block has been requested and allocated somewhere else
            // (which should be unusual).
            if tail != output {
                cx.push_with_comment(
                    Inst::Copy {
                        address: tail,
                        output,
                    },
                    format_args!("copy block {this:?} output"),
                );
            }
        }

        Ok(())
    }

    #[instrument]
    fn asm_let<'hir>(cx: &mut Ctxt<'_, 'hir>, this: ExprId, pat: PatId) -> Result<()> {
        let panic_label = cx.new_label("let_panic");

        if let PatOutcome::Refutable = asm_pat(cx, pat, panic_label)? {
            let end = cx.new_label("pat_end");
            cx.push(Inst::Jump { label: end });
            cx.label(panic_label)?;
            cx.push(Inst::Panic {
                reason: PanicReason::UnmatchedPattern,
            });
            cx.label(end)?;
        }

        let pat_expr = cx.pat_expr(pat)?;
        cx.remove_expr_user(pat_expr, ExprUser::Expr(this))?;
        Ok(())
    }

    #[instrument]
    fn asm_store<'hir>(cx: &mut Ctxt<'_, 'hir>, this: ExprId, value: InstValue) -> Result<()> {
        let ([], output) = cx.addresses(this, [])?;
        cx.push(Inst::Store { value, output });
        Ok(())
    }

    #[instrument]
    fn asm_bytes<'hir>(cx: &mut Ctxt<'_, 'hir>, this: ExprId, bytes: &[u8]) -> Result<()> {
        let slot = cx.q.unit.new_static_bytes(cx.span, bytes)?;
        let ([], output) = cx.addresses(this, [])?;
        cx.push(Inst::Bytes { slot, output });
        Ok(())
    }

    #[instrument]
    fn asm_string<'hir>(cx: &mut Ctxt<'_, 'hir>, this: ExprId, string: &str) -> Result<()> {
        let slot = cx.q.unit.new_static_string(cx.span, string)?;
        let ([], output) = cx.addresses(this, [])?;
        cx.push(Inst::String { slot, output });
        Ok(())
    }

    #[instrument]
    fn asm_unary<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        this: ExprId,
        op: ExprUnOp,
        expr: UsedExprId,
    ) -> Result<()> {
        let ([address], output) = cx.addresses(this, [expr])?;

        match op {
            ExprUnOp::Neg => {
                cx.push(Inst::Neg { address, output });
            }
            ExprUnOp::Not => {
                cx.push(Inst::Not { address, output });
            }
        }

        Ok(())
    }

    #[instrument]
    fn asm_binary_assign<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        this: ExprId,
        lhs: UsedExprId,
        op: ast::BinOp,
        rhs: UsedExprId,
    ) -> Result<()> {
        let ([lhs, rhs], output) = cx.addresses(this, [lhs, rhs])?;

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
            rhs,
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
        this: ExprId,
        lhs: UsedExprId,
        op: ast::BinOp,
        rhs: UsedExprId,
    ) -> Result<()> {
        let end_label = cx.new_label("conditional_end");

        let lhs = cx.with_state_checkpoint(|cx| {
            let ([lhs], _) = cx.addresses(this, [lhs])?;
            Ok(lhs)
        })?;

        match op {
            ast::BinOp::And(..) => {
                cx.push(Inst::JumpIfNot {
                    address: lhs,
                    label: end_label,
                });
            }
            ast::BinOp::Or(..) => {
                cx.push(Inst::JumpIf {
                    address: lhs,
                    label: end_label,
                });
            }
            op => {
                return Err(cx.error(CompileErrorKind::UnsupportedBinaryOp { op }));
            }
        }

        let rhs = cx.with_state_checkpoint(|cx| {
            let ([rhs], _) = cx.addresses(this, [rhs])?;
            Ok(rhs)
        })?;

        cx.label(end_label)?;
        Ok(())
    }

    /// Assembling of a binary expression.
    #[instrument]
    fn asm_binary<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        this: ExprId,
        lhs: UsedExprId,
        op: ast::BinOp,
        rhs: UsedExprId,
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

        let ([a, b], output) = cx.addresses(this, [lhs, rhs])?;
        cx.push(Inst::Op { op, a, b, output });
        Ok(())
    }

    #[instrument]
    fn asm_index<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        this: ExprId,
        target: UsedExprId,
        index: UsedExprId,
    ) -> Result<()> {
        let ([address, index], output) = cx.addresses(this, [target, index])?;

        cx.push(Inst::IndexGet {
            address,
            index,
            output,
        });

        Ok(())
    }

    /// Compile an item.
    #[instrument]
    fn asm_meta<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        this: ExprId,
        meta: &PrivMeta,
        needs: Needs,
        named: &'hir Named<'hir>,
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

                    let ([], output) = cx.addresses(this, [])?;

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

                    let ([], output) = cx.addresses(this, [])?;

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

                    let ([], output) = cx.addresses(this, [])?;

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

                    let ([], output) = cx.addresses(this, [])?;

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

                    let ([], output) = cx.addresses(this, [])?;

                    cx.push_with_comment(Inst::LoadFn { hash, output }, meta.info(cx.q.pool));
                    ExprOutcome::Output
                }
                PrivMetaKind::Const {
                    const_value: value, ..
                } => {
                    named.assert_not_generic()?;
                    let expr = const_value(cx, value)?;
                    let ([], output) = cx.addresses(this, [])?;
                    asm_to_output(cx, expr.id(), output)?
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

            let ([], output) = cx.addresses(this, [])?;

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
        this: ExprId,
        kind: ExprStructKind,
        exprs: &[UsedExprId],
    ) -> Result<()> {
        let ([], output) = cx.addresses(this, [])?;
        let address = cx.array(this, exprs.iter().copied())?;

        match kind {
            ExprStructKind::Anonymous { slot } => {
                cx.push(Inst::Object {
                    address,
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
                    address,
                    slot,
                    output,
                });
            }
            ExprStructKind::StructVariant { hash, slot } => {
                cx.push(Inst::StructVariant {
                    hash,
                    address,
                    slot,
                    output,
                });
            }
        }

        Ok(())
    }

    #[instrument]
    fn asm_vec<'hir>(cx: &mut Ctxt<'_, 'hir>, this: ExprId, items: &[UsedExprId]) -> Result<()> {
        let ([], output) = cx.addresses(this, [])?;
        let address = cx.array(this, items.iter().copied())?;

        cx.push(Inst::Vec {
            address,
            count: items.len(),
            output,
        });

        Ok(())
    }

    #[instrument]
    fn asm_range<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        this: ExprId,
        from: UsedExprId,
        limits: InstRangeLimits,
        to: UsedExprId,
    ) -> Result<()> {
        let ([from, to], output) = cx.addresses(this, [from, to])?;

        cx.push(Inst::Range {
            from,
            to,
            limits,
            output,
        });

        Ok(())
    }

    #[instrument]
    fn asm_tuple<'hir>(cx: &mut Ctxt<'_, 'hir>, this: ExprId, items: &[UsedExprId]) -> Result<()> {
        match items {
            &[a] => {
                let (args, output) = cx.addresses(this, [a])?;
                cx.push(Inst::Tuple1 { args, output });
            }
            &[a, b] => {
                let (args, output) = cx.addresses(this, [a, b])?;
                cx.push(Inst::Tuple2 { args, output });
            }
            &[a, b, c] => {
                let (args, output) = cx.addresses(this, [a, b, c])?;
                cx.push(Inst::Tuple3 { args: args, output });
            }
            &[a, b, c, d] => {
                let (args, output) = cx.addresses(this, [a, b, c, d])?;
                cx.push(Inst::Tuple4 { args, output });
            }
            args => {
                let ([], output) = cx.addresses(this, [])?;
                let address = cx.array(this, args.iter().copied())?;

                cx.push(Inst::Tuple {
                    address,
                    count: args.len(),
                    output,
                });
            }
        }

        Ok(())
    }

    #[instrument]
    fn asm_option<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        this: ExprId,
        value: Option<UsedExprId>,
    ) -> Result<()> {
        match value {
            Some(value) => {
                let ([address], output) = cx.addresses(this, [value])?;

                cx.push(Inst::Variant {
                    address,
                    variant: InstVariant::Some,
                    output,
                });
            }
            None => {
                let ([], output) = cx.addresses(this, [])?;

                cx.push(Inst::Variant {
                    address: output,
                    variant: InstVariant::None,
                    output,
                });
            }
        }

        Ok(())
    }

    #[instrument]
    fn asm_tuple_field_access<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        this: ExprId,
        lhs: UsedExprId,
        index: usize,
    ) -> Result<()> {
        let ([address], output) = cx.addresses(this, [lhs])?;

        cx.push(Inst::TupleIndexGet {
            address,
            index,
            output,
        });

        Ok(())
    }

    #[instrument]
    fn asm_struct_field_access<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        this: ExprId,
        lhs: UsedExprId,
        field: &str,
    ) -> Result<()> {
        let ([address], output) = cx.addresses(this, [lhs])?;
        let slot = cx.q.unit.new_static_string(cx.span, field)?;

        cx.push(Inst::ObjectIndexGet {
            address,
            slot,
            output,
        });

        Ok(())
    }

    #[instrument]
    fn asm_assign<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        _this: ExprId,
        _lhs: UsedExprId,
        _rhs: UsedExprId,
    ) -> Result<()> {
        todo!()
    }

    #[instrument]
    fn asm_assign_struct_field<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        this: ExprId,
        lhs: UsedExprId,
        field: &str,
        rhs: UsedExprId,
    ) -> Result<()> {
        let ([address, value], output) = cx.addresses(this, [lhs, rhs])?;
        let slot = cx.q.unit.new_static_string(cx.span, field)?;

        cx.push(Inst::ObjectIndexSet {
            address,
            value,
            slot,
            output,
        });

        Ok(())
    }

    #[instrument]
    fn asm_assign_tuple_field<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        this: ExprId,
        lhs: UsedExprId,
        index: usize,
        rhs: UsedExprId,
    ) -> Result<()> {
        let ([address, value], output) = cx.addresses(this, [lhs, rhs])?;

        cx.push(Inst::TupleIndexSet {
            address,
            value,
            index,
            output,
        });

        Ok(())
    }

    #[instrument]
    fn asm_call_hash<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        this: ExprId,
        args: &[UsedExprId],
        hash: Hash,
    ) -> Result<()> {
        let ([], output) = cx.addresses(this, [])?;
        let address = cx.array(this, args.iter().copied())?;

        cx.push(Inst::Call {
            hash,
            address,
            count: args.len(),
            output,
        });

        Ok(())
    }

    #[instrument]
    fn asm_call_instance<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        this: ExprId,
        lhs: UsedExprId,
        args: &[UsedExprId],
        hash: Hash,
    ) -> Result<()> {
        let ([], output) = cx.addresses(this, [])?;
        let address = cx.allocator.array_address();

        {
            let output = cx.allocator.array_address();
            asm_to_output(cx, lhs.id(), output)?;
            cx.allocator.alloc_array_item();
        }

        for &used_id in args {
            let output = cx.allocator.array_address();
            asm_to_output(cx, used_id.id(), output)?;
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
        this: ExprId,
        expr: UsedExprId,
        args: &[UsedExprId],
    ) -> Result<()> {
        let ([function], output) = cx.addresses(this, [expr])?;
        let array = cx.array(this, args.iter().copied())?;

        cx.push(Inst::CallFn {
            function,
            address: array,
            count: args.len(),
            output,
        });

        Ok(())
    }

    #[instrument]
    fn asm_yield<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        this: ExprId,
        expr: Option<UsedExprId>,
    ) -> Result<()> {
        Ok(match expr {
            Some(expr) => {
                let ([address], output) = cx.addresses(this, [expr])?;

                cx.push(Inst::Yield { address, output });
            }
            None => {
                let ([], output) = cx.addresses(this, [])?;
                cx.push(Inst::YieldUnit { output });
            }
        })
    }

    #[instrument]
    fn asm_await<'hir>(cx: &mut Ctxt<'_, 'hir>, this: ExprId, expr: UsedExprId) -> Result<()> {
        let ([address], output) = cx.addresses(this, [expr])?;

        cx.push(Inst::Await { address, output });

        Ok(())
    }

    #[instrument]
    fn asm_return<'hir>(cx: &mut Ctxt<'_, 'hir>, this: ExprId, expr: UsedExprId) -> Result<()> {
        let ([address], _) = cx.addresses(this, [expr])?;
        cx.push(Inst::Return { address });
        Ok(())
    }

    #[instrument]
    fn asm_try<'hir>(cx: &mut Ctxt<'_, 'hir>, this: ExprId, expr: UsedExprId) -> Result<()> {
        let ([address], output) = cx.addresses(this, [expr])?;

        cx.push(Inst::Try { address, output });

        Ok(())
    }

    #[instrument]
    fn asm_function<'hir>(cx: &mut Ctxt<'_, 'hir>, this: ExprId, hash: Hash) -> Result<()> {
        let ([], output) = cx.addresses(this, [])?;
        cx.push(Inst::LoadFn { hash, output });
        Ok(())
    }

    #[instrument]
    fn asm_closure<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        this: ExprId,
        captures: &[UsedExprId],
        hash: Hash,
    ) -> Result<()> {
        let ([], output) = cx.addresses(this, [])?;
        let address = cx.array(this, captures.iter().copied())?;

        cx.push(Inst::Closure {
            hash,
            address,
            count: captures.len(),
            output,
        });

        Ok(())
    }

    /// Assemble a loop.
    #[instrument]
    fn asm_loop<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        condition: LoopCondition,
        body: UsedExprId,
        start: Label,
        end: Label,
    ) -> Result<ExprOutcome> {
        let cleanup = match condition {
            LoopCondition::Forever => {
                cx.label(start)?;
                None
            }
            LoopCondition::Condition { pat } => {
                cx.label(start)?;
                asm_pat(cx, pat, end)?;
                None
            }
            LoopCondition::Iterator { iter, pat } => {
                let value = cx.pat_expr(pat)?;
                let [iter_address, value_address] = cx.expression_addresses(None, [iter, value])?;

                cx.push(Inst::CallInstance {
                    hash: *Protocol::INTO_ITER,
                    address: iter_address,
                    count: 0,
                    output: iter_address,
                });

                cx.label(start)?;

                cx.push(Inst::IterNext {
                    address: iter_address,
                    label: end,
                    output: value_address,
                });

                asm_pat(cx, pat, end)?;
                Some((iter, value))
            }
        };

        let outcome = asm(cx, body.id())?;

        cx.push(Inst::Jump { label: start });
        cx.label(end)?;

        if let Some((iter, value)) = cleanup {
            // cx.free_expr([iter, value])?;
            // TODO: do something about iter and value.
        }

        Ok(outcome)
    }

    /// Assemble a loop break.
    #[instrument]
    fn asm_break<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        value: Option<UsedExprId>,
        label: Label,
        loop_slot: ExprId,
    ) -> Result<ExprOutcome> {
        if let Some(expr) = value {
            let loop_summary = cx.expr_mut(loop_slot)?.use_summary();

            // Only assemble loop output *if* the loop is still alive.
            if loop_summary.is_alive() {
                let output = cx.expr_output(loop_slot)?;
                asm_to_output(cx, expr.id(), output)?;
            } else {
                asm(cx, expr.id())?;
            }
        }

        cx.push(Inst::Jump { label });
        Ok(ExprOutcome::Empty)
    }

    /// Assemble a loop continue.
    #[instrument]
    fn asm_continue<'hir>(cx: &mut Ctxt<'_, 'hir>, label: Label) -> Result<ExprOutcome> {
        cx.push(Inst::Jump { label });
        Ok(ExprOutcome::Empty)
    }

    /// Compile a string concat expression.
    #[instrument]
    fn asm_string_concat<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        this: ExprId,
        exprs: &[UsedExprId],
    ) -> Result<ExprOutcome> {
        let ([], output) = cx.addresses(this, [])?;
        let address = cx.array(this, exprs.iter().copied())?;

        cx.push(Inst::StringConcat {
            address,
            count: exprs.len(),
            size_hint: 0,
            output,
        });

        Ok(ExprOutcome::Output)
    }

    /// Compile a format expression.
    #[instrument]
    fn asm_format<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        this: ExprId,
        spec: &FormatSpec,
        expr: UsedExprId,
    ) -> Result<ExprOutcome> {
        let ([address], output) = cx.addresses(this, [expr])?;

        cx.push(Inst::Format {
            address,
            spec: *spec,
            output,
        });

        Ok(ExprOutcome::Output)
    }
}

#[instrument]
fn asm_pat<'hir>(cx: &mut Ctxt<'_, 'hir>, pat: PatId, label: Label) -> Result<PatOutcome> {
    let pat_expr = cx.pat_expr(pat)?;
    let pat = cx.pat(pat)?;
    let span = pat.span;
    let kind = pat.kind;

    cx.with_span(span, |cx| match kind {
        PatKind::Irrefutable => {
            // Assemble value eagerly which is used by a branched context.
            let summary = cx.expr_mut(pat_expr.id())?.use_summary();

            if summary.is_pending() {
                if summary.is_branched() || has_side_effects(cx, pat_expr)? {
                    asm(cx, pat_expr.id())?;
                }
            }

            Ok(PatOutcome::Irrefutable)
        }
        PatKind::IrrefutableSequence { items } => {
            let mut outcome = PatOutcome::Irrefutable;

            for pat in items {
                outcome = outcome.combine(asm_pat(cx, *pat, label)?);
            }

            Ok(outcome)
        }
        PatKind::BoundLit { lit, expr } => {
            let [expr] = cx.expression_addresses(None, [expr])?;

            match cx.expr(lit.id())?.kind {
                ExprKind::Store { value } => {
                    cx.push(Inst::MatchValue {
                        address: expr,
                        value,
                        label,
                    });
                }
                ExprKind::String { string } => {
                    let slot = cx.q.unit.new_static_string(cx.span, string)?;
                    cx.push(Inst::MatchString {
                        address: expr,
                        slot,
                        label,
                    });
                }
                ExprKind::Bytes { bytes } => {
                    let slot = cx.q.unit.new_static_bytes(cx.span, bytes)?;
                    cx.push(Inst::MatchBytes {
                        address: expr,
                        slot,
                        label,
                    });
                }
                _ => {
                    return Err(cx.error(CompileErrorKind::UnsupportedPattern));
                }
            }

            Ok(PatOutcome::Refutable)
        }
        PatKind::BoundVec {
            address,
            expr,
            is_open,
            items,
        } => {
            let [expr] = cx.expression_addresses(None, [expr])?;

            cx.push(Inst::MatchSequence {
                address: expr,
                type_check: TypeCheck::Vec,
                len: items.len(),
                exact: !is_open,
                label,
                output: address,
            });

            for &pat in items {
                asm_pat(cx, pat, label)?;
            }

            Ok(PatOutcome::Refutable)
        }
        PatKind::AnonymousTuple {
            address,
            expr,
            is_open,
            items,
        } => {
            let [expr] = cx.expression_addresses(None, [expr])?;

            cx.push(Inst::MatchSequence {
                address: expr,
                type_check: TypeCheck::Tuple,
                len: items.len(),
                exact: !is_open,
                label,
                output: address,
            });

            for &pat in items {
                asm_pat(cx, pat, label)?;
            }

            Ok(PatOutcome::Refutable)
        }
        PatKind::AnonymousObject {
            address,
            expr,
            slot,
            is_open,
            items,
        } => {
            let [expr] = cx.expression_addresses(None, [expr])?;

            cx.push(Inst::MatchObject {
                address: expr,
                slot,
                exact: !is_open,
                label,
                output: address,
            });

            for &pat in items {
                asm_pat(cx, pat, label)?;
            }

            Ok(PatOutcome::Refutable)
        }
        PatKind::TypedSequence {
            type_match,
            expr,
            items,
        } => {
            let [expr] = cx.expression_addresses(None, [expr])?;

            match type_match {
                TypeMatch::BuiltIn { type_check } => cx.push(Inst::MatchBuiltIn {
                    address: expr,
                    type_check,
                    label,
                }),
                TypeMatch::Type { type_hash } => cx.push(Inst::MatchType {
                    address: expr,
                    type_hash,
                    label,
                }),
                TypeMatch::Variant {
                    variant_hash,
                    enum_hash,
                    index,
                } => {
                    let output = cx.allocator.next_address();

                    cx.push(Inst::MatchVariant {
                        address: expr,
                        variant_hash,
                        enum_hash,
                        index,
                        label,
                        output,
                    });
                }
            }

            for &pat in items {
                asm_pat(cx, pat, label)?;
            }

            Ok(PatOutcome::Refutable)
        }
        _ => Err(cx.msg("trying to assemble pattern which hasn't been bound")),
    })
}

/// Assemble a block.
#[instrument]
fn block<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: &hir::Block<'hir>) -> Result<UsedExprId> {
    cx.with_span(hir.span(), |cx| {
        let statements = iter!(cx; hir.statements, |stmt| {
            match *stmt {
                hir::Stmt::Local(hir) => {
                    let expr = expr_value(cx, hir.expr)?;
                    let pat = pat(cx, hir.pat, expr)?;
                    cx.insert_expr(ExprKind::Let { pat })?
                }
                hir::Stmt::Expr(hir) => expr_value(cx, hir)?,
            }
        });

        let tail = if let Some(hir) = hir.tail {
            Some(expr_value(cx, hir)?)
        } else {
            None
        };

        cx.insert_expr(ExprKind::Block { statements, tail })
    })
}

/// Compile an expression.
#[instrument]
fn expr_value<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: &hir::Expr<'hir>) -> Result<UsedExprId> {
    expr(cx, hir, Needs::Value)
}

/// Custom needs expression.
#[instrument]
fn expr<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: &hir::Expr<'hir>, needs: Needs) -> Result<UsedExprId> {
    return cx.with_span(hir.span, |cx| {
        let address = match hir.kind {
            hir::ExprKind::Empty => cx.insert_expr(ExprKind::Empty)?,
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
            hir::ExprKind::Loop(hir) => loop_(cx, hir)?,
            hir::ExprKind::Break(hir) => break_(cx, hir)?,
            hir::ExprKind::Continue(hir) => continue_(cx, hir)?,
            hir::ExprKind::Let(..) => todo!(),
            hir::ExprKind::If(..) => todo!(),
            hir::ExprKind::Match(..) => todo!(),
            hir::ExprKind::MacroCall(hir) => macro_call(cx, hir)?,
        };

        Ok(address)
    });

    /// Assembling of a binary expression.
    #[instrument]
    fn path<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        hir: &'hir hir::Path,
        needs: Needs,
    ) -> Result<UsedExprId> {
        let span = hir.span();
        let loc = Location::new(cx.source_id, span);

        if let Some(ast::PathKind::SelfValue) = hir.as_kind() {
            return cx.scopes.lookup(loc, cx.scope, cx.q.visitor, SELF);
        }

        let named = cx.convert_path(hir)?;

        if let Needs::Value = needs {
            if let Some(local) = named.as_local() {
                let local = local.resolve(resolve_context!(cx.q))?;

                if let Some(expr) = cx.scopes.try_lookup(loc, cx.scope, cx.q.visitor, local)? {
                    return Ok(expr);
                }
            }
        }

        if let Some(meta) = cx.try_lookup_meta(span, named.item)? {
            return Ok(cx.insert_expr(ExprKind::Meta {
                meta: alloc!(cx; meta),
                needs,
                named: alloc!(cx; named),
            })?);
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
    fn assign<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        hir: &'hir hir::ExprAssign<'hir>,
    ) -> Result<UsedExprId> {
        let rhs = expr_value(cx, &hir.rhs)?;
        let lhs = expr_value(cx, &hir.lhs)?;

        let kind = match cx.expr(lhs.id())?.kind {
            ExprKind::StructFieldAccess { lhs, field, .. } => {
                ExprKind::AssignStructField { lhs, field, rhs }
            }
            ExprKind::TupleFieldAccess { lhs, index } => {
                ExprKind::AssignTupleField { lhs, index, rhs }
            }
            _ => ExprKind::Assign { lhs, rhs },
        };

        cx.insert_expr(kind)
    }

    /// Assemble a call expression.
    #[instrument]
    fn call<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: &hir::ExprCall<'hir>) -> Result<UsedExprId> {
        let this = expr_value(cx, hir.expr)?;

        let this = this.map_expr(cx, |cx, kind| match kind {
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
                        let value = const_value(cx, &value)?;
                        // NB: It is valid to coerce an expr_const into a kind
                        // because it belongs to the constant scope which requires
                        // no cleanup.
                        return Ok(cx.expr(value.id())?.kind);
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
                expr: cx.insert_expr(kind)?,
                args: iter!(cx; hir.args, |hir| expr_value(cx, hir)?),
            }),
        })?;

        Ok(this)
    }

    /// Decode a field access expression.
    #[instrument]
    fn field_access<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        hir: &hir::ExprFieldAccess<'hir>,
    ) -> Result<UsedExprId> {
        match hir.expr_field {
            hir::ExprField::Path(path) => {
                if let Some(ident) = path.try_as_ident() {
                    let n = ident.resolve(resolve_context!(cx.q))?;
                    let field = str!(cx; n);
                    let hash = Hash::instance_fn_name(n);
                    let lhs = expr_value(cx, hir.expr)?;
                    return cx.insert_expr(ExprKind::StructFieldAccess { lhs, field, hash });
                }

                if let Some((ident, generics)) = path.try_as_ident_generics() {
                    let n = ident.resolve(resolve_context!(cx.q))?;
                    let hash = Hash::instance_fn_name(n.as_ref());
                    let lhs = expr_value(cx, hir.expr)?;

                    return cx.insert_expr(ExprKind::StructFieldAccessGeneric {
                        lhs,
                        hash,
                        generics,
                    });
                }
            }
            hir::ExprField::LitNumber(field) => {
                let span = field.span();

                let number = field.resolve(resolve_context!(cx.q))?;
                let index = number.as_tuple_index().ok_or_else(|| {
                    CompileError::new(span, CompileErrorKind::UnsupportedTupleIndex { number })
                })?;

                let lhs = expr_value(cx, hir.expr)?;
                return cx.insert_expr(ExprKind::TupleFieldAccess { lhs, index });
            }
        }

        Err(cx.error(CompileErrorKind::BadFieldAccess))
    }

    /// Assemble a unary expression.
    #[instrument]
    fn unary<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: &hir::ExprUnary<'hir>) -> Result<UsedExprId> {
        // NB: special unary expressions.
        if let ast::UnOp::BorrowRef { .. } = hir.op {
            return Err(cx.error(CompileErrorKind::UnsupportedRef));
        }

        if let (ast::UnOp::Neg(..), hir::ExprKind::Lit(ast::Lit::Number(n))) =
            (hir.op, hir.expr.kind)
        {
            match n.resolve(resolve_context!(cx.q))? {
                ast::Number::Float(n) => {
                    return cx.insert_expr(ExprKind::Store {
                        value: InstValue::Float(-n),
                    });
                }
                ast::Number::Integer(int) => {
                    let n = match int.neg().to_i64() {
                        Some(n) => n,
                        None => {
                            return Err(cx.error(ParseErrorKind::BadNumberOutOfBounds));
                        }
                    };

                    return cx.insert_expr(ExprKind::Store {
                        value: InstValue::Integer(n),
                    });
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
        cx.insert_expr(ExprKind::Unary { op, expr })
    }

    /// Assemble a binary expression.
    #[instrument]
    fn binary<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: &hir::ExprBinary<'hir>) -> Result<UsedExprId> {
        if hir.op.is_assign() {
            let lhs = expr_value(cx, hir.lhs)?;
            let rhs = expr_value(cx, hir.rhs)?;

            return cx.insert_expr(ExprKind::BinaryAssign {
                lhs,
                op: hir.op,
                rhs,
            });
        }

        if hir.op.is_conditional() {
            let lhs_scope = cx.scopes.push_branch(cx.span, Some(cx.scope))?;
            let lhs = cx.with_scope(lhs_scope, |cx| expr_value(cx, hir.lhs))?;
            cx.scopes.pop(cx.span, lhs_scope)?;

            let rhs_scope = cx.scopes.push_branch(cx.span, Some(cx.scope))?;
            let rhs = cx.with_scope(rhs_scope, |cx| expr_value(cx, hir.rhs))?;
            cx.scopes.pop(cx.span, rhs_scope)?;

            return cx.insert_expr(ExprKind::BinaryConditional {
                lhs,
                op: hir.op,
                rhs,
            });
        }

        let lhs = expr_value(cx, hir.lhs)?;
        let rhs = expr(cx, hir.rhs, rhs_needs_of(&hir.op))?;

        return cx.insert_expr(ExprKind::Binary {
            lhs,
            op: hir.op,
            rhs,
        });

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
    fn index<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: &hir::ExprIndex<'hir>) -> Result<UsedExprId> {
        let target = expr_value(cx, hir.target)?;
        let index = expr_value(cx, hir.index)?;

        cx.insert_expr(ExprKind::Index { target, index })
    }

    /// A block expression.
    #[instrument]
    fn expr_block<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: &hir::ExprBlock<'hir>) -> Result<UsedExprId> {
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
                        cx.scopes.lookup(
                            Location::new(cx.source_id, cx.span),
                            cx.scope,
                            cx.q.visitor,
                            &ident.ident,
                        )?
                    });

                    let hash = cx.q.pool.item_type_hash(meta.item_meta.item);
                    cx.insert_expr(ExprKind::CallHash { hash, args })?
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
    fn return_<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        hir: Option<&'hir hir::Expr<'hir>>,
    ) -> Result<UsedExprId> {
        let kind = match hir {
            Some(hir) => ExprKind::Return {
                expr: expr_value(cx, hir)?,
            },
            None => ExprKind::Return {
                expr: cx.insert_expr(ExprKind::Store {
                    value: InstValue::Unit,
                })?,
            },
        };

        cx.insert_expr(kind)
    }

    /// Assemble a yield expression.
    #[instrument]
    fn yield_<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: Option<&'hir hir::Expr>) -> Result<UsedExprId> {
        let expr = match hir {
            Some(hir) => Some(expr_value(cx, hir)?),
            None => None,
        };

        cx.insert_expr(ExprKind::Yield { expr })
    }

    /// Assemble an await expression.
    #[instrument]
    fn await_<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: &'hir hir::Expr) -> Result<UsedExprId> {
        let expr = expr_value(cx, hir)?;
        cx.insert_expr(ExprKind::Await { expr })
    }

    /// Assemble a try expression.
    #[instrument]
    fn try_<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: &'hir hir::Expr) -> Result<UsedExprId> {
        let expr = expr_value(cx, hir)?;
        cx.insert_expr(ExprKind::Try { expr })
    }

    /// Assemble a closure.
    #[instrument]
    fn closure<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: &hir::ExprClosure<'hir>) -> Result<UsedExprId> {
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

                cx.scopes.lookup(
                    Location::new(cx.source_id, cx.span),
                    cx.scope,
                    cx.q.visitor,
                    &capture.ident,
                )?
            });

            ExprKind::Closure { hash, captures }
        };

        cx.insert_expr(kind)
    }

    /// Construct a literal value.
    #[instrument]
    fn lit<'hir>(cx: &mut Ctxt<'_, 'hir>, ast: &'hir ast::Lit) -> Result<UsedExprId> {
        cx.with_span(ast.span(), |cx| {
            let expr = match ast {
                ast::Lit::Bool(lit) => cx.insert_expr(ExprKind::Store {
                    value: InstValue::Bool(lit.value),
                })?,
                ast::Lit::Char(lit) => {
                    let ch = lit.resolve(resolve_context!(cx.q))?;
                    cx.insert_expr(ExprKind::Store {
                        value: InstValue::Char(ch),
                    })?
                }
                ast::Lit::ByteStr(lit) => {
                    let b = lit.resolve(resolve_context!(cx.q))?;
                    cx.insert_expr(ExprKind::Bytes {
                        bytes: cx
                            .arena
                            .alloc_bytes(b.as_ref())
                            .map_err(arena_error(cx.span))?,
                    })?
                }
                ast::Lit::Str(lit) => {
                    let b = lit.resolve(resolve_context!(cx.q))?;
                    cx.insert_expr(ExprKind::String {
                        string: str!(cx; b.as_ref()),
                    })?
                }
                ast::Lit::Byte(lit) => {
                    let b = lit.resolve(resolve_context!(cx.q))?;
                    cx.insert_expr(ExprKind::Store {
                        value: InstValue::Byte(b),
                    })?
                }
                ast::Lit::Number(number) => lit_number(cx, number)?,
            };

            Ok(expr)
        })
    }

    /// An object expression
    #[instrument]
    fn object<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: &hir::ExprObject<'hir>) -> Result<UsedExprId> {
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
            Some(hir) => path(cx, hir, Needs::Type)?.map_expr(cx, |cx, kind| {
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
                })?
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
    fn tuple<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: &hir::ExprSeq<'hir>) -> Result<UsedExprId> {
        let items = iter!(cx; hir.items, |hir| expr_value(cx, hir)?);
        Ok(cx.insert_expr(ExprKind::Tuple { items })?)
    }

    /// Assemble a vector expression.
    #[instrument]
    fn vec<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: &hir::ExprSeq<'hir>) -> Result<UsedExprId> {
        let items = iter!(cx; hir.items, |hir| expr_value(cx, hir)?);
        Ok(cx.insert_expr(ExprKind::Vec { items })?)
    }

    /// Assemble a range expression.
    #[instrument]
    fn range<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: &hir::ExprRange<'hir>) -> Result<UsedExprId> {
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
        let from = cx.insert_expr(from)?;

        let to = ExprKind::Option {
            value: match hir.to {
                Some(hir) => Some(expr_value(cx, hir)?),
                None => None,
            },
        };
        let to = cx.insert_expr(to)?;

        Ok(cx.insert_expr(ExprKind::Range { from, limits, to })?)
    }

    /// Convert a literal number into an expression.
    #[instrument]
    fn lit_number<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: &ast::LitNumber) -> Result<UsedExprId> {
        let number = hir.resolve(resolve_context!(cx.q))?;

        match number {
            ast::Number::Float(float) => Ok(cx.insert_expr(ExprKind::Store {
                value: InstValue::Float(float),
            })?),
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
                })?)
            }
        }
    }

    /// Convert a loop.
    #[instrument]
    fn loop_<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: &hir::ExprLoop<'hir>) -> Result<UsedExprId> {
        let label = match hir.label {
            Some(label) => Some(cx.scopes.name(label.resolve(resolve_context!(cx.q))?)),
            None => None,
        };

        let start = cx.new_label("loop_start");
        let end = cx.new_label("loop_end");
        let loop_slot = cx.insert_expr(ExprKind::Empty)?;

        let scope =
            cx.scopes
                .push_loop(cx.span, Some(cx.scope), label, start, end, loop_slot.id())?;

        let condition = match hir.condition {
            hir::LoopCondition::Forever => LoopCondition::Forever,
            hir::LoopCondition::Condition { condition } => {
                let pat = match *condition {
                    hir::Condition::Expr(hir) => {
                        let lit = cx.insert_expr(ExprKind::Store {
                            value: InstValue::Bool(true),
                        })?;
                        let expr = expr_value(cx, hir)?;
                        cx.insert_pat(PatKind::Lit { lit }, expr)?
                    }
                    hir::Condition::ExprLet(hir) => {
                        let expr = expr_value(cx, hir.expr)?;
                        cx.with_scope(scope, |cx| pat(cx, hir.pat, expr))?
                    }
                };

                LoopCondition::Condition { pat }
            }
            hir::LoopCondition::Iterator { binding, iter } => {
                let iter = expr_value(cx, iter)?;
                let value = cx.insert_expr(ExprKind::Empty)?;
                let pat = cx.with_scope(scope, |cx| pat(cx, binding, value))?;
                LoopCondition::Iterator { iter, pat }
            }
        };

        let body = cx.with_scope(scope, |cx| block(cx, hir.body))?;

        cx.scopes.pop(cx.span, scope)?;

        loop_slot.map_expr(cx, |_, _| {
            Ok(ExprKind::Loop {
                condition,
                body,
                start,
                end,
            })
        })
    }

    /// Convert a break.
    #[instrument]
    fn break_<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        hir: &hir::ExprBreakValue<'hir>,
    ) -> Result<UsedExprId> {
        let kind = match *hir {
            hir::ExprBreakValue::None => {
                let (label, loop_slot) = if let Some(ControlFlow::Loop(control_flow)) = cx
                    .scopes
                    .find_ancestor(cx.scope, |flow| matches!(flow, ControlFlow::Loop { .. }))
                {
                    (control_flow.end, control_flow.loop_slot)
                } else {
                    return Err(cx.error(CompileErrorKind::BreakOutsideOfLoop));
                };

                ExprKind::Break {
                    value: None,
                    label,
                    loop_slot,
                }
            }
            hir::ExprBreakValue::Expr(hir) => {
                let (label, loop_slot) = if let Some(ControlFlow::Loop(control_flow)) = cx
                    .scopes
                    .find_ancestor(cx.scope, |flow| matches!(flow, ControlFlow::Loop { .. }))
                {
                    (control_flow.end, control_flow.loop_slot)
                } else {
                    return Err(cx.error(CompileErrorKind::BreakOutsideOfLoop));
                };

                let expr = expr_value(cx, hir)?;

                ExprKind::Break {
                    value: Some(expr),
                    label,
                    loop_slot,
                }
            }
            hir::ExprBreakValue::Label(ast) => {
                let expected = cx.scopes.name(ast.resolve(resolve_context!(cx.q))?);

                let (label, loop_slot) = if let Some(ControlFlow::Loop(control_flow)) =
                    cx.scopes.find_ancestor(
                        cx.scope,
                        |flow| matches!(flow, ControlFlow::Loop(l) if l.label == Some(expected)),
                    ) {
                    (control_flow.end, control_flow.loop_slot)
                } else {
                    let name = cx.scopes.name_to_string(cx.span, expected)?;
                    return Err(cx.error(CompileErrorKind::MissingLabel { label: name.into() }));
                };

                ExprKind::Break {
                    value: None,
                    label,
                    loop_slot,
                }
            }
        };

        cx.insert_expr(kind)
    }

    /// Convert a continue.
    #[instrument]
    fn continue_<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: Option<&ast::Label>) -> Result<UsedExprId> {
        let kind = match hir {
            None => {
                let label = if let Some(ControlFlow::Loop(control_flow)) = cx
                    .scopes
                    .find_ancestor(cx.scope, |flow| matches!(flow, ControlFlow::Loop { .. }))
                {
                    control_flow.start
                } else {
                    return Err(cx.error(CompileErrorKind::BreakOutsideOfLoop));
                };

                ExprKind::Continue { label }
            }
            Some(ast) => {
                let expected = cx.scopes.name(ast.resolve(resolve_context!(cx.q))?);

                let label = if let Some(ControlFlow::Loop(control_flow)) = cx.scopes.find_ancestor(
                    cx.scope,
                    |flow| matches!(flow, ControlFlow::Loop(l) if l.label == Some(expected)),
                ) {
                    control_flow.start
                } else {
                    let name = cx.scopes.name_to_string(cx.span, expected)?;
                    return Err(cx.error(CompileErrorKind::MissingLabel { label: name.into() }));
                };

                ExprKind::Continue { label }
            }
        };

        cx.insert_expr(kind)
    }

    /// Assemble a macro call.
    #[instrument]
    fn macro_call<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: &hir::MacroCall<'hir>) -> Result<UsedExprId> {
        let id = match hir {
            hir::MacroCall::Template(hir) => builtin_template(cx, hir)?,
            hir::MacroCall::Format(hir) => builtin_format(cx, hir)?,
            hir::MacroCall::File(hir) => {
                let current = hir.value.resolve(resolve_context!(cx.q))?;
                let string = cx
                    .arena
                    .alloc_str(current.as_ref())
                    .map_err(arena_error(cx.span))?;
                cx.insert_expr(ExprKind::String { string })?
            }
            hir::MacroCall::Line(hir) => lit_number(cx, &hir.value)?,
        };

        Ok(id)
    }

    /// Assemble #[builtin] template!(...) macro.
    #[instrument]
    fn builtin_template<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        hir: &hir::BuiltInTemplate<'hir>,
    ) -> Result<UsedExprId> {
        // Template represents a single literal string.
        if hir
            .exprs
            .iter()
            .all(|hir| matches!(hir.kind, hir::ExprKind::Lit(ast::Lit::Str(..))))
        {
            let mut string = String::new();

            if hir.from_literal {
                cx.q.diagnostics.template_without_expansions(
                    cx.source_id,
                    cx.span,
                    Some(cx.context()),
                );
            }

            for hir in hir.exprs {
                if let hir::ExprKind::Lit(ast::Lit::Str(s)) = hir.kind {
                    string += s.resolve(resolve_context!(cx.q))?.as_ref();
                }
            }

            let string = cx.arena.alloc_str(&string).map_err(arena_error(cx.span))?;
            return cx.insert_expr(ExprKind::String { string });
        }

        let exprs = iter!(cx; hir.exprs, |hir| expr_value(cx, hir)?);
        cx.insert_expr(ExprKind::StringConcat { exprs })
    }

    /// Assemble #[builtin] format!(...) macro.
    #[instrument]
    fn builtin_format<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        hir: &hir::BuiltInFormat<'hir>,
    ) -> Result<UsedExprId> {
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

        let spec = FormatSpec::new(flags, fill, align, width, precision, format_type);
        let expr = expr_value(cx, hir.value)?;

        cx.insert_expr(ExprKind::Format {
            spec: alloc!(cx; spec),
            expr,
        })
    }
}

/// An expression id that might've been looked up.
#[derive(Debug, Clone, Copy)]
struct UsedExprId {
    id: ExprId,
    kind: UsedExprIdKind,
}

impl UsedExprId {
    /// Construct a used expression id.
    fn new(id: ExprId) -> Self {
        Self {
            id,
            kind: UsedExprIdKind::Default,
        }
    }

    /// Get the underlying expression id.
    const fn id(&self) -> ExprId {
        self.id
    }

    /// Construct a used expression id from a binding.
    fn binding(id: ExprId, binding: Binding, use_kind: UseKind) -> Self {
        Self {
            id,
            kind: UsedExprIdKind::Binding { binding, use_kind },
        }
    }

    /// Get the use kind of the expression.
    fn use_kind(&self) -> UseKind {
        match self.kind {
            UsedExprIdKind::Default => UseKind::Same,
            UsedExprIdKind::Binding { use_kind, .. } => use_kind,
        }
    }

    /// Map the current expression into some other expression.
    fn map_expr<'hir, M>(self, cx: &mut Ctxt<'_, 'hir>, map: M) -> Result<UsedExprId>
    where
        M: FnOnce(&mut Ctxt<'_, 'hir>, ExprKind<'hir>) -> Result<ExprKind<'hir>>,
    {
        let kind = map(cx, cx.expr(self.id)?.kind)?;
        let replaced = mem::replace(&mut cx.expr_mut(self.id)?.kind, kind);
        cx.release_expr_kind(replaced, ExprUser::Expr(self.id))?;
        cx.retain_expr_kind(kind, ExprUser::Expr(self.id))?;
        Ok(self)
    }
}

#[derive(Debug, Clone, Copy)]
enum UsedExprIdKind {
    /// Default use which has no particular use implications.
    Default,
    /// Use from a binding through the specified `use_kind`.
    Binding { binding: Binding, use_kind: UseKind },
}

/// Compile a constant value into an expression.
#[instrument]
fn const_value<'hir>(cx: &mut Ctxt<'_, 'hir>, value: &ConstValue) -> Result<UsedExprId> {
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

    cx.insert_expr(kind)
}

/// Assemble a pattern.
fn pat<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: &hir::Pat<'hir>, expr: UsedExprId) -> Result<PatId> {
    let mut removed = HashMap::new();
    let nested_pat = pat(cx, hir, &mut removed)?;
    return cx.insert_pat(nested_pat.kind, expr);

    #[instrument]
    fn pat<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        hir: &hir::Pat<'hir>,
        removed: &mut HashMap<Name, Span>,
    ) -> Result<Pat<'hir>> {
        return cx.with_span(hir.span(), |cx| {
            let kind = match hir.kind {
                hir::PatKind::PatPath(hir) => pat_binding(cx, hir, removed)?,
                hir::PatKind::PatIgnore => cx.build_pat(PatKind::Ignore),
                hir::PatKind::PatLit(hir) => {
                    let lit = expr_value(cx, hir)?;
                    cx.build_pat(PatKind::Lit { lit })
                }
                hir::PatKind::PatVec(hir) => {
                    let items = iter!(cx; hir.items, |hir| pat(cx, hir, removed)?);

                    cx.build_pat(PatKind::Vec {
                        items,
                        is_open: hir.is_open,
                    })
                }
                hir::PatKind::PatTuple(hir) => pat_tuple(cx, hir, removed)?,
                hir::PatKind::PatObject(hir) => pat_object(cx, hir, removed)?,
            };

            Ok(kind)
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
    ) -> Result<Pat<'hir>> {
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

        let items = iter!(cx; hir.items, |hir| pat(cx, hir, removed)?);

        Ok(cx.build_pat(PatKind::Tuple {
            kind,
            items,
            is_open: hir.is_open,
        }))
    }

    /// Assemble a pattern object.
    #[instrument]
    fn pat_object<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        hir: &hir::PatObject<'hir>,
        removed: &mut HashMap<Name, Span>,
    ) -> Result<Pat<'hir>> {
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

        let items = iter!(cx; hir.bindings, |binding| {
            if let Some(hir) = binding.pat {
                pat(cx, hir, removed)?
            } else {
                pat_object_key(cx, binding.key, removed)?
            }
        });

        Ok(cx.build_pat(PatKind::Object { kind, items }))
    }

    /// Assemble a binding pattern which is *just* a variable captured from an object.
    #[instrument]
    fn pat_object_key<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        hir: &hir::ObjectKey<'hir>,
        removed: &mut HashMap<Name, Span>,
    ) -> Result<Pat<'hir>> {
        let pat = match *hir {
            hir::ObjectKey::LitStr(..) => {
                return Err(cx.error(CompileErrorKind::UnsupportedPattern))
            }
            hir::ObjectKey::Path(hir) => pat_binding(cx, hir, removed)?,
        };

        Ok(pat)
    }

    /// Handle the binding of a path.
    #[instrument]
    fn pat_binding<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        hir: &hir::Path<'hir>,
        removed: &mut HashMap<Name, Span>,
    ) -> Result<Pat<'hir>> {
        let kind = match pat_path(cx, hir)? {
            PatPath::Name { name } => {
                let ghost_expr = cx.insert_expr(ExprKind::Empty)?;

                let name = cx.scopes.name(name);
                let replaced = cx.declare(name, ghost_expr.id())?;

                if replaced.is_some() {
                    if let Some(span) = removed.insert(name, cx.span) {
                        return Err(cx.error(CompileErrorKind::DuplicateBinding {
                            previous_span: span,
                        }));
                    }
                }

                PatKind::Ghost { ghost_expr }
            }
            PatPath::Meta { meta } => PatKind::Meta { meta },
        };

        Ok(cx.build_pat(kind))
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

/// Walk over an expression and visit each expression that it touches.
fn walk_expr<T>(cx: &mut Ctxt<'_, '_>, kind: ExprKind<'_>, mut op: T) -> Result<()>
where
    T: FnMut(&mut Ctxt<'_, '_>, UsedExprId) -> Result<()>,
{
    match kind {
        ExprKind::Empty => {}
        ExprKind::Address => {}
        ExprKind::Assign { lhs, rhs } => {
            op(cx, lhs)?;
            op(cx, rhs)?;
        }
        ExprKind::TupleFieldAccess { lhs, .. } => {
            op(cx, lhs)?;
        }
        ExprKind::StructFieldAccess { lhs, .. } => {
            op(cx, lhs)?;
        }
        ExprKind::StructFieldAccessGeneric { lhs, .. } => {
            op(cx, lhs)?;
        }
        ExprKind::AssignStructField { lhs, rhs, .. } => {
            op(cx, lhs)?;
            op(cx, rhs)?;
        }
        ExprKind::AssignTupleField { lhs, rhs, .. } => {
            op(cx, lhs)?;
            op(cx, rhs)?;
        }
        ExprKind::Block {
            statements, tail, ..
        } => {
            for &slot in statements {
                op(cx, slot)?;
            }

            if let Some(slot) = tail {
                op(cx, slot)?;
            }
        }
        ExprKind::Let { pat } => {
            let expr = cx.pat_expr(pat)?;
            op(cx, expr)?;
        }
        ExprKind::Store { .. } => {}
        ExprKind::Bytes { .. } => {}
        ExprKind::String { .. } => {}
        ExprKind::Unary { expr, .. } => {
            op(cx, expr)?;
        }
        ExprKind::BinaryAssign { lhs, rhs, .. } => {
            op(cx, lhs)?;
            op(cx, rhs)?;
        }
        ExprKind::BinaryConditional { lhs, rhs, .. } => {
            op(cx, lhs)?;
            op(cx, rhs)?;
        }
        ExprKind::Binary { lhs, rhs, .. } => {
            op(cx, lhs)?;
            op(cx, rhs)?;
        }
        ExprKind::Index { target, .. } => {
            op(cx, target)?;
        }
        ExprKind::Meta { .. } => {}
        ExprKind::Struct { exprs, .. } => {
            for &slot in exprs {
                op(cx, slot)?;
            }
        }
        ExprKind::Tuple { items } => {
            for &slot in items {
                op(cx, slot)?;
            }
        }
        ExprKind::Vec { items } => {
            for &slot in items {
                op(cx, slot)?;
            }
        }
        ExprKind::Range { from, to, .. } => {
            op(cx, from)?;
            op(cx, to)?;
        }
        ExprKind::Option { value } => {
            if let Some(slot) = value {
                op(cx, slot)?;
            }
        }
        ExprKind::CallHash { args, .. } => {
            for &slot in args {
                op(cx, slot)?;
            }
        }
        ExprKind::CallInstance { lhs, args, .. } => {
            op(cx, lhs)?;

            for &slot in args {
                op(cx, slot)?;
            }
        }
        ExprKind::CallExpr { expr, args } => {
            op(cx, expr)?;

            for &slot in args {
                op(cx, slot)?;
            }
        }
        ExprKind::Yield { expr } => {
            if let Some(slot) = expr {
                op(cx, slot)?;
            }
        }
        ExprKind::Await { expr } => {
            op(cx, expr)?;
        }
        ExprKind::Return { expr } => {
            op(cx, expr)?;
        }
        ExprKind::Try { expr } => {
            op(cx, expr)?;
        }
        ExprKind::Function { .. } => {}
        ExprKind::Closure { captures, .. } => {
            for &slot in captures {
                op(cx, slot)?;
            }
        }
        ExprKind::Loop { body, .. } => {
            op(cx, body)?;
        }
        ExprKind::Break { value, .. } => {
            if let Some(slot) = value {
                op(cx, slot)?;
            }
        }
        ExprKind::Continue { .. } => {}
        ExprKind::StringConcat { exprs } => {
            for &slot in exprs {
                op(cx, slot)?;
            }
        }
        ExprKind::Format { expr, .. } => {
            op(cx, expr)?;
        }
    }

    Ok(())
}

/// Walk over a pattern and visit each expression that it touches.
fn walk_pat<T>(cx: &mut Ctxt<'_, '_>, kind: PatKind<'_>, mut op: T) -> Result<()>
where
    T: FnMut(&mut Ctxt<'_, '_>, UsedExprId) -> Result<()>,
{
    match kind {
        PatKind::Ignore => {}
        PatKind::Lit { lit } => {
            op(cx, lit)?;
        }
        PatKind::Ghost { ghost_expr } => {}
        PatKind::Meta { meta } => {}
        PatKind::Vec { items, is_open } => {}
        PatKind::Tuple {
            kind,
            items,
            is_open,
        } => {}
        PatKind::Object { kind, items } => {}
        PatKind::Irrefutable => {}
        PatKind::IrrefutableSequence { items } => {}
        PatKind::BoundLit { lit, expr } => {
            op(cx, lit)?;
            op(cx, expr)?;
        }
        PatKind::BoundVec {
            address,
            expr,
            is_open,
            items,
        } => {
            op(cx, expr)?;
        }
        PatKind::AnonymousTuple {
            address,
            expr,
            is_open,
            items,
        } => {
            op(cx, expr)?;
        }
        PatKind::AnonymousObject {
            address,
            expr,
            slot,
            is_open,
            items,
        } => {
            op(cx, expr)?;
        }
        PatKind::TypedSequence {
            type_match,
            expr,
            items,
        } => {
            op(cx, expr)?;
        }
    }

    Ok(())
}
