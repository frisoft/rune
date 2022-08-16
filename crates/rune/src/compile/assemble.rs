use std::borrow::Cow;
use std::fmt;
use std::mem;
use std::num::NonZeroUsize;
use std::ops::{Deref, Neg as _};
use std::vec;

use num::ToPrimitive;
use rune_macros::__instrument_hir as instrument;

use crate::arena::{Arena, ArenaAllocError, ArenaWriteSliceOutOfBounds};
use crate::ast::{self, Span, Spanned};
use crate::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
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
    /// Set of useless slots.
    useless: BTreeSet<Slot>,
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
        }
    }

    /// Access the scopes built by the context.
    pub(crate) fn into_allocator(self) -> Allocator {
        self.allocator
    }

    /// Allocate a new output address slotted by the given expression address.
    fn address_for(&mut self, slot: Slot) -> Result<AssemblyAddress> {
        if let Some(address) = self.slot(slot)?.address {
            return Ok(address);
        }

        Err(CompileError::msg(
            self.span,
            format_args!("no inherent address allocated for slot {slot:?}"),
        ))
    }

    /// Allocate an address for an expression.
    fn alloc_for(&mut self, expr: Slot) -> Result<AssemblyAddress> {
        self.scopes.alloc_for(self.span, expr, &mut self.allocator)
    }

    /// Free the inherent allocation associated with the slot.
    fn free_for(&mut self, slot: Slot) -> Result<()> {
        self.scopes.free_for(self.span, slot, &mut self.allocator)
    }

    /// Access slot storage for the given slot.
    fn slot(&self, slot: Slot) -> Result<&SlotStorage<'hir>> {
        self.scopes.slot(self.span, slot)
    }

    /// Access slot storage mutably for the given slot.
    fn slot_mut(&mut self, slot: Slot) -> Result<&mut SlotStorage<'hir>> {
        self.scopes.slot_mut(self.span, slot)
    }

    /// Declare a variable.
    fn declare(&mut self, name: Name, address: Slot) -> Result<Option<Slot>> {
        self.scopes.declare(self.span, self.scope, name, address)
    }

    /// Get the identifier for the next slot that will be inserted.
    fn next_slot(&self) -> Result<Slot> {
        self.scopes.next_slot(self.span)
    }

    /// Add a slot user with custom use kind.
    fn add_slot_user_with(&mut self, slot: Slot, user: Slot, use_kind: UseKind) -> Result<()> {
        let span = self.span;
        self.slot_mut(slot)?.add_user(span, user, use_kind)?;
        // We can now remove this slot from the set of useless slots.
        self.useless.remove(&slot);
        Ok(())
    }

    /// Take a single slot user.
    fn take_slot_use(&mut self, slot: Slot, user: Slot) -> Result<SlotUse> {
        let span = self.span;
        self.slot_mut(slot)?.take_use(span, user)
    }

    /// Remove a slot user.
    fn remove_slot_user(&mut self, slot: Slot, user: Slot) -> Result<()> {
        let span = self.span;
        let storage = self.slot_mut(slot)?;
        storage.remove_user(span, user)?;

        if storage.uses.is_empty() {
            self.useless.insert(slot);
        }

        Ok(())
    }

    /// Retain all slots referenced by the given expression.
    fn retain_expr_kind(&mut self, kind: ExprKind<'hir>, this: Slot) -> Result<()> {
        walk_expr_slots(kind, |slot, use_kind| {
            self.add_slot_user_with(slot, this, use_kind)
        })
    }

    /// Release all slots referenced by the given expression.
    fn release_expr_kind(&mut self, kind: ExprKind<'hir>, this: Slot) -> Result<()> {
        walk_expr_slots(kind, |slot, _| self.remove_slot_user(slot, this))
    }

    /// Insert a expression.
    fn insert_expr(&mut self, kind: ExprKind<'hir>) -> Result<Slot> {
        let this = self.next_slot()?;
        self.retain_expr_kind(kind, this)?;
        // Mark current statement inserted as potentially useless to sweep it up later.
        self.useless.insert(this);
        self.scopes.insert_expr(self.span, kind)
    }

    /// Insert an expression with an assembly address.
    fn insert_expr_with_address(
        &mut self,
        kind: ExprKind<'hir>,
        address: AssemblyAddress,
    ) -> Result<Slot> {
        let slot = self.insert_expr(kind)?;
        self.slot_mut(slot)?.address = Some(address);
        Ok(slot)
    }

    /// Free an expression.
    fn free_expr(&mut self, this: Slot) -> Result<()> {
        let kind = self.slot(this)?.kind;
        self.release_expr_kind(kind, this)?;
        Ok(())
    }

    /// Insert a pattern.
    fn insert_pat(&mut self, kind: PatKind<'hir>) -> Pat<'hir> {
        Pat::new(self.span, kind)
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

    /// Allocate an array of addresses.
    fn array<I>(&mut self, this: Slot, expressions: I) -> Result<UsedAddress>
    where
        I: IntoIterator<Item = Slot>,
    {
        let address = self.allocator.array_address();
        let mut count = 0;

        for slot in expressions {
            let output = self.allocator.array_address();

            let used = self.take_slot_use(slot, this)?;

            // If only, assembled directly onto our desired output.
            if used.is_only() {
                asm(self, slot, output)?;
            } else if used.is_pending() {
                let slot_output = self.alloc_for(slot)?;
                asm(self, slot, slot_output)?;
                self.push(Inst::Copy {
                    address: slot_output,
                    output,
                });
            } else {
                let slot_output = self.alloc_for(slot)?;
                self.push(Inst::Copy {
                    address: slot_output,
                    output,
                });
            }

            if used.is_last() {
                self.free_for(slot)?;
            }

            self.allocator.alloc_array_item();
            count += 1;
        }

        Ok(UsedAddress::array(address, count))
    }

    /// Allocate a collection of slots as addresses.
    fn addresses<const N: usize, T>(
        &mut self,
        user: Slot,
        slots: [Slot; N],
        tmp: T,
    ) -> Result<[UsedAddress; N]>
    where
        T: IntoIterator<Item = AssemblyAddress>,
    {
        let mut tmp = tmp.into_iter();
        let mut out = [mem::MaybeUninit::<UsedAddress>::uninit(); N];

        for (slot, out) in slots.into_iter().zip(&mut out) {
            let address = follow_slot(self, &mut tmp, slot, user)?;
            out.write(address);
        }

        // SAFETY: we just initialized the array above.
        let out = unsafe { array_assume_init(out) };
        return Ok(out);

        pub unsafe fn array_assume_init<T, const N: usize>(
            array: [mem::MaybeUninit<T>; N],
        ) -> [T; N] {
            let ret = (&array as *const _ as *const [T; N]).read();
            ret
        }

        /// Recursively follow bindings to addresses if we are the only user.
        fn follow_slot(
            cx: &mut Ctxt<'_, '_>,
            tmp: &mut impl Iterator<Item = AssemblyAddress>,
            slot: Slot,
            user: Slot,
        ) -> Result<UsedAddress> {
            let kind = cx.slot(slot)?.kind;
            let used = cx.take_slot_use(slot, user)?;

            let address = match kind {
                ExprKind::Address => {
                    let address = cx.alloc_for(slot)?;
                    UsedAddress::new(address, slot, used)
                }
                ExprKind::Binding {
                    slot: followed_slot,
                    ..
                } if used.is_only() => return follow_slot(cx, tmp, followed_slot, slot),
                _ => {
                    let address = if let Some(address) = tmp.next() {
                        if used.is_pending() {
                            asm(cx, slot, address)?;
                        } else {
                            let slot_address = cx.alloc_for(slot)?;

                            cx.push(Inst::Copy {
                                address: slot_address,
                                output: address,
                            });

                            if used.is_last() {
                                cx.free_for(slot)?;
                            }
                        }

                        address
                    } else {
                        let address = cx.alloc_for(slot)?;

                        if used.is_pending() {
                            asm(cx, slot, address)?;
                        }

                        address
                    };

                    UsedAddress::new(address, slot, used)
                }
            };

            Ok(address)
        }
    }

    /// Free a collection of allocated addresses.
    fn free_iter<const N: usize>(&mut self, allocs: [UsedAddress; N]) -> Result<()> {
        for alloc in allocs {
            match alloc.kind {
                UsedAddressKind::Slot(slot, used) => {
                    if used.is_last() {
                        self.free_for(slot)?;
                    }
                }
                UsedAddressKind::Array(count) => {
                    self.allocator.free_array(self.span, count)?;
                }
            }
        }

        Ok(())
    }
}

/// An address that might have been allocated..
#[must_use]
#[derive(Debug, Clone, Copy)]
struct UsedAddress {
    address: AssemblyAddress,
    kind: UsedAddressKind,
}

impl UsedAddress {
    /// A temporary address.
    const fn new(address: AssemblyAddress, slot: Slot, used: SlotUse) -> Self {
        Self {
            address,
            kind: UsedAddressKind::Slot(slot, used),
        }
    }

    /// The base of an array of allocated addresses.
    const fn array(address: AssemblyAddress, count: usize) -> Self {
        Self {
            address,
            kind: UsedAddressKind::Array(count),
        }
    }

    /// Get the number of elements in this sequence of addresses.
    fn count(&self) -> usize {
        match self.kind {
            UsedAddressKind::Slot(..) => 1,
            UsedAddressKind::Array(count) => count,
        }
    }
}

impl Deref for UsedAddress {
    type Target = AssemblyAddress;

    fn deref(&self) -> &Self::Target {
        &self.address
    }
}

/// The kind of a [UsedAddress].
#[derive(Debug, Clone, Copy)]
enum UsedAddressKind {
    Slot(Slot, SlotUse),
    Array(usize),
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

    let scope = cx.scopes.push(cx.span, None)?;

    let address = cx.with_scope(scope, |cx| {
        let mut first = true;

        let mut patterns = Vec::new();

        for (index, arg) in hir.args.iter().enumerate() {
            match *arg {
                hir::FnArg::SelfValue(span) => {
                    if !instance_fn || !first {
                        return Err(CompileError::new(span, CompileErrorKind::UnsupportedSelf));
                    }

                    let name = cx.scopes.name(SELF);
                    let address = AssemblyAddress::Bottom(index);
                    let slot = cx.insert_expr_with_address(ExprKind::Empty, address)?;
                    cx.declare(name, slot)?;
                }
                hir::FnArg::Pat(hir) => {
                    let address = AssemblyAddress::Bottom(index);
                    let slot = cx.insert_expr_with_address(ExprKind::Empty, address)?;
                    patterns.push((hir, slot));
                }
            }

            first = false;
        }

        block(cx, hir.body)
    })?;

    // Reserve the argument slots.
    cx.allocator.reserve(hir.span, hir.args.len())?;
    cx.scopes.pop(span, scope)?;
    debug_stdout(cx, address)?;

    let output = cx.alloc_for(address)?;
    asm(cx, address, output)?;
    cx.push(Inst::Return { address: output });
    Ok(())
}

/// Debug a slot to stdout.
fn debug_stdout(cx: &mut Ctxt<'_, '_>, slot: Slot) -> Result<()> {
    let out = std::io::stdout();
    let mut out = out.lock();

    let mut task = Task {
        patterns: 0,
        visited: HashSet::new(),
        queue: VecDeque::from_iter([Job::Slot(slot)]),
    };

    while let Some(job) = task.queue.pop_front() {
        match job {
            Job::Pat(index, pat) => {
                debug_pat(&mut out, index, pat, &mut task)?;
            }
            Job::Slot(slot) => {
                debug_slot(&mut out, cx, slot, &mut task)?;
            }
        }
    }

    return Ok(());

    struct Task<'hir> {
        patterns: usize,
        visited: HashSet<Slot>,
        queue: VecDeque<Job<'hir>>,
    }

    impl<'hir> Task<'hir> {
        fn slot(&mut self, slot: Slot) -> String {
            if self.visited.insert(slot) {
                self.queue.push_back(Job::Slot(slot));
            }

            format!("${}", slot.index())
        }

        fn pat(&mut self, pat: Pat<'hir>) -> String {
            let index = self.patterns;
            self.queue.push_back(Job::Pat(index, pat));
            self.patterns += 1;
            format!("pat${}", index)
        }
    }

    enum Job<'hir> {
        Pat(usize, Pat<'hir>),
        Slot(Slot),
    }

    /// Debug a slot.
    fn debug_slot<'hir, O>(
        o: &mut O,
        cx: &mut Ctxt<'_, 'hir>,
        slot: Slot,
        task: &mut Task<'hir>,
    ) -> Result<()>
    where
        O: std::io::Write,
    {
        let storage = cx.scopes.slot(cx.span, slot)?;
        let span = storage.span;
        let err = write_err(span);

        let uses = format_uses(storage, span)?;

        let same = storage.same;
        let branches = storage.branches;
        let kind = storage.kind;

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
                    LoopCondition::Condition { expr, pat } => {
                        let expr = task.slot(expr);
                        let pat = task.pat(pat);
                        writeln!(
                            o,
                            "{}{} = Condition {{ expr = {expr}, pat = {pat} }},",
                            $pad,
                            stringify!($var)
                        )
                        .map_err(err)?;
                    }
                    LoopCondition::Iterator { iter, binding } => {
                        let iter = task.slot(iter);
                        let binding = task.pat(binding);
                        writeln!(
                            o,
                            "{}{} = Iterator {{ iter = {iter}, binding = {binding} }},",
                            $pad,
                            stringify!($var)
                        )
                        .map_err(err)?;
                    }
                }
            };

            ($pad:literal, $fn:ident, $var:expr) => {{
                let name = task.$fn($var);
                writeln!(o, "{}{} = {name},", $pad, stringify!($var)).map_err(err)?;
            }};

            ($pad:literal, [array $fn:ident], $var:expr) => {
                let mut it = IntoIterator::into_iter($var);

                let first = it.next();

                if let Some(&value) = first {
                    writeln!(o, "{}{} = [", $pad, stringify!($var)).map_err(err)?;
                    let name = task.$fn(value);
                    writeln!(o, "{}  {name},", $pad).map_err(err)?;

                    for &value in it {
                        let name = task.$fn(value);
                        writeln!(o, "{}  {name},", $pad).map_err(err)?;
                    }

                    writeln!(o, "{}],", $pad).map_err(err)?;
                } else {
                    writeln!(o, "{}{} = [],", $pad, stringify!($var)).map_err(err)?;
                }
            };

            ($pad:literal, [option $fn:ident], $var:expr) => {
                if let Some(value) = $var {
                    let name = task.$fn(value);
                    writeln!(o, "{}{} = Some({name}),", $pad, stringify!($var)).map_err(err)?;
                } else {
                    writeln!(o, "{}{} = None,", $pad, stringify!($var)).map_err(err)?;
                }
            };
        }

        macro_rules! variant {
            ($name:ident) => {{
                writeln!(o, "${} = {} {{ same = {same:?}, branches = {branches:?}, uses = {uses} }},", slot.index(), stringify!($name)).map_err(err)?;
            }};

            ($name:ident { $($what:tt $field:ident),* }) => {{
                writeln!(o, "${} = {} {{ same = {same:?}, branches = {branches:?}, uses = {uses} }} {{", slot.index(), stringify!($name)).map_err(err)?;
                $(field!("  ", $what, $field);)*
                writeln!(o, "}}").map_err(err)?;
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
                Binding { binding binding, slot slot, lit use_kind },
                TupleFieldAccess { slot lhs, lit index },
                StructFieldAccess { slot lhs, lit field, lit hash },
                StructFieldAccessGeneric { slot lhs, lit hash, lit generics },
                Assign { binding binding, slot lhs, lit use_kind, slot rhs },
                AssignStructField { slot lhs, lit field, slot rhs },
                AssignTupleField { slot lhs, lit index, slot rhs },
                Block { [array slot] statements, [option slot] tail },
                Let { pat pat, slot expr },
                Store { lit value },
                Bytes { lit bytes },
                String { lit string },
                Unary { lit op, slot expr },
                BinaryAssign { slot lhs, lit op, slot rhs },
                BinaryConditional { slot lhs, lit op, slot rhs },
                Binary { slot lhs, lit op, slot rhs },
                Index { slot target, slot index },
                Meta { lit meta, lit needs, lit named },
                Struct { lit kind, [array slot] exprs },
                Tuple { [array slot] items },
                Vec { [array slot] items },
                Range { slot from, lit limits, slot to },
                Option { [option slot] value },
                CallAddress { slot address, [array slot] args },
                CallHash { lit hash, [array slot] args },
                CallInstance { slot lhs, lit hash, [array slot] args },
                CallExpr { slot expr, [array slot] args },
                Yield { [option slot] expr },
                Await { slot expr },
                Return { slot expr },
                Try { slot expr },
                Function { lit hash },
                Closure { lit hash, [array slot] captures },
                Loop { condition condition, slot body, lit start, lit end },
                Break { [option slot] value, lit label, slot loop_slot },
                Continue { lit label },
                StringConcat { [array slot] exprs },
                Format { lit spec, slot expr },
            }
        }

        Ok(())
    }

    /// Debug a patter.
    fn debug_pat<'hir, O>(
        o: &mut O,
        index: usize,
        pat: Pat<'hir>,
        task: &mut Task<'hir>,
    ) -> Result<()>
    where
        O: std::io::Write,
    {
        let err = write_err(pat.span);

        macro_rules! field {
            ($pad:literal, lit, $var:expr) => {
                writeln!(o, "{}{} = {:?},", $pad, stringify!($var), $var).map_err(err)?;
            };

            ($pad:literal, $fn:ident, $var:expr) => {{
                let name = task.$fn($var);
                writeln!(o, "{}{} = {name},", $pad, stringify!($var)).map_err(err)?;
            }};

            ($pad:literal, [array $fn:ident], $var:expr) => {
                let mut it = IntoIterator::into_iter($var);

                let first = it.next();

                if let Some(&value) = first {
                    writeln!(o, "{}{} = [", $pad, stringify!($var)).map_err(err)?;
                    let name = task.$fn(value);
                    writeln!(o, "{}  {name},", $pad).map_err(err)?;

                    for &value in it {
                        let name = task.$fn(value);
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
                writeln!(o, "pat${index} = {name},", name = stringify!($name)).map_err(err)?;
            }};

            ($name:ident { $($what:tt $field:ident),* }) => {{
                writeln!(o, "pat${index} = {name} {{", name = stringify!($name)).map_err(err)?;
                $(field!("  ", $what, $field);)*
                writeln!(o, "}}").map_err(err)?;
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
            pat.kind, {
                Ignore,
                Lit { slot lit },
                Name { lit name, slot name_expr },
                Meta { lit meta },
                Vec {
                    [array pat] items,
                    lit is_open,
                },
                Tuple {
                    lit kind,
                    [array pat] patterns,
                    lit is_open,
                },
                Object {
                    lit kind,
                    [array pat] patterns,
                },
            }
        }

        Ok(())
    }

    fn format_uses(storage: &SlotStorage<'_>, span: Span) -> Result<String> {
        use std::fmt::Write as _;

        let mut users = String::new();
        users.push('{');

        let mut first = true;

        let mut it = storage.uses.iter().peekable();

        while let Some((slot, use_kind)) = it.next() {
            if mem::take(&mut first) {
                users.push(' ');
            }

            write!(users, "${} = {use_kind:?}", slot.index()).map_err(write_err(span))?;

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
    /// An expression who's inherent address refers to a static address.
    Address,
    /// An expression referencing a binding.
    Binding {
        binding: Binding,
        slot: Slot,
        use_kind: UseKind,
    },
    /// An assignment to a binding.
    Assign {
        /// The binding being assigned to.
        binding: Binding,
        /// Address to assign to.
        lhs: Slot,
        /// The lookup being assigned to.
        use_kind: UseKind,
        /// The expression to assign.
        rhs: Slot,
    },
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
        pat: Pat<'hir>,
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
    Loop {
        /// The condition to advance the loop.
        condition: LoopCondition<'hir>,
        /// The body of the loop.
        body: Slot,
        /// The start label to use.
        start: Label,
        /// The end label.
        end: Label,
    },
    /// A break expression.
    Break {
        /// The value to break with.
        value: Option<Slot>,
        /// End label to jump to.
        label: Label,
        /// The loop slot to break to.
        loop_slot: Slot,
    },
    /// A continue expression.
    Continue {
        /// The label to jump to.
        label: Label,
    },
    /// A concatenation of a sequence of expressions.
    StringConcat { exprs: &'hir [Slot] },
    /// A value format.
    Format { spec: &'hir FormatSpec, expr: Slot },
}

/// A loop condition.
#[derive(Debug, Clone, Copy)]
enum LoopCondition<'hir> {
    /// A forever loop condition.
    Forever,
    /// A pattern condition.
    Condition { expr: Slot, pat: Pat<'hir> },
    /// An iterator condition.
    Iterator { iter: Slot, binding: Pat<'hir> },
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
    Lit { lit: Slot },
    /// A path pattern.
    Name { name: &'hir str, name_expr: Slot },
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
        patterns: &'hir [Pat<'hir>],
        is_open: bool,
    },
    /// An object pattern.
    Object {
        kind: PatObjectKind<'hir>,
        patterns: &'hir [Pat<'hir>],
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
        name_expr: Slot,
        expr: Slot,
    },
    Lit {
        lit: Slot,
        expr: Slot,
    },
    Vec {
        address: AssemblyAddress,
        expr: Slot,
        is_open: bool,
        items: &'hir [BoundPat<'hir>],
    },
    AnonymousTuple {
        address: AssemblyAddress,
        expr: Slot,
        is_open: bool,
        items: &'hir [BoundPat<'hir>],
    },
    AnonymousObject {
        address: AssemblyAddress,
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
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
struct Slot(NonZeroUsize);

impl Slot {
    /// Construct a new slot.
    fn new(value: usize) -> Option<Self> {
        Some(Self(NonZeroUsize::new(value.wrapping_add(1))?))
    }

    /// Get the index that the slot corresponds to.
    fn index(&self) -> usize {
        self.0.get().wrapping_sub(1)
    }

    /// Map the current expression into some other expression.
    fn map_slot<'hir, M>(self, cx: &mut Ctxt<'_, 'hir>, map: M) -> Result<Slot>
    where
        M: FnOnce(&mut Ctxt<'_, 'hir>, ExprKind<'hir>) -> Result<ExprKind<'hir>>,
    {
        let kind = map(cx, cx.slot(self)?.kind)?;
        let replaced = mem::replace(&mut cx.slot_mut(self)?.kind, kind);
        cx.release_expr_kind(replaced, self)?;
        cx.retain_expr_kind(kind, self)?;
        Ok(self)
    }
}

impl fmt::Debug for Slot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Slot").field(&self.index()).finish()
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

    /// Allocate a new output address slotted by the given expression address.
    fn alloc_for(
        &mut self,
        span: Span,
        slot: Slot,
        allocator: &mut Allocator,
    ) -> Result<AssemblyAddress> {
        let slot = self.slot_mut(span, slot)?;
        Ok(*slot.address.get_or_insert_with(|| allocator.alloc()))
    }

    /// Free the implicit address associated with the given slot.
    fn free_for(&mut self, span: Span, slot: Slot, allocator: &mut Allocator) -> Result<()> {
        let slot = self.slot_mut(span, slot)?;

        // NB: We make freeing optional instead of a hard error, since we might
        // call this function even if an address has not been allocated for the
        // slot.
        if let Some(address) = slot.address.take() {
            allocator.free(span, address)?;
        }

        Ok(())
    }

    /// Load slot storage for the given slot.
    fn slot(&self, span: Span, slot: Slot) -> Result<&SlotStorage<'hir>> {
        let slot = match self.slots.get(slot.index()) {
            Some(slot) => slot,
            None => {
                return Err(CompileError::msg(
                    span,
                    format_args!("failed to look up slot {slot:?}"),
                ))
            }
        };

        Ok(slot)
    }

    /// Load slot storage mutably for the given slot.
    fn slot_mut(&mut self, span: Span, slot: Slot) -> Result<&mut SlotStorage<'hir>> {
        let slot = match self.slots.get_mut(slot.index()) {
            Some(slot) => slot,
            None => {
                return Err(CompileError::msg(
                    span,
                    format_args!("failed to look up slot {slot:?}"),
                ))
            }
        };

        Ok(slot)
    }

    /// Get the identifier for the next slot that will be inserted.
    fn next_slot(&self, span: Span) -> Result<Slot> {
        match Slot::new(self.slots.len()) {
            Some(slot) => Ok(slot),
            None => return Err(CompileError::msg(span, "ran out of slots")),
        }
    }

    /// Insert an expression and return its slot address.
    fn insert_expr(&mut self, span: Span, kind: ExprKind<'hir>) -> Result<Slot> {
        self.insert_with(span, |_| kind)
    }

    /// Allocate an alot address associated with an unknown type.
    #[tracing::instrument(skip_all)]
    fn insert_with<T>(&mut self, span: Span, builder: T) -> Result<Slot>
    where
        T: FnOnce(Slot) -> ExprKind<'hir>,
    {
        let slot = self.next_slot(span)?;

        self.slots.push(SlotStorage {
            span,
            slot,
            kind: builder(slot),
            uses: BTreeMap::new(),
            address: None,
            same: 0,
            branches: 0,
            pending: true,
            sealed: None,
        });

        Ok(slot)
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
        loop_slot: Slot,
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
    ) -> Result<Option<(Binding, Slot, UseKind)>> {
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
                return Ok(Some((binding, slot, use_kind)));
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
    ) -> Result<(Binding, Slot, UseKind)> {
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
    loop_slot: Slot,
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

/// Use summary for a slot.
#[derive(Debug, Clone, Copy)]
struct UseSummary {
    /// How many users this value has.
    same: usize,
    /// How many branches that are using this value.
    branches: usize,
    /// The total number of users of an expression.
    total: usize,
    /// If the value is pending construction.
    pending: bool,
}

impl UseSummary {
    /// Test if this is the last user.
    fn is_last(&self) -> bool {
        self.same == 0 && self.branches == 0
    }

    /// If the expression is waiting to be built.
    fn is_pending(&self) -> bool {
        self.pending
    }
}

/// Use summary for a slot.
#[derive(Debug, Clone, Copy)]
struct SlotUse {
    /// The kind of use which is in place.
    use_kind: UseKind,
    /// How many users this value has.
    summary: UseSummary,
}

impl SlotUse {
    /// Test if this is the only non-branch user.
    fn is_only(&self) -> bool {
        matches!((self.use_kind, self.summary.total), (UseKind::Same, 1))
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

#[derive(Debug, Clone, Copy)]
struct SlotSealed {
    total: usize,
}

#[derive(Debug, Clone)]
struct SlotStorage<'hir> {
    /// The span associated with the slot.
    span: Span,
    /// The output address of the expression.
    slot: Slot,
    /// The kind of the expression.
    kind: ExprKind<'hir>,
    /// The downstream users of this slot.
    uses: BTreeMap<Slot, UseKind>,
    /// The implicit address of the slot.
    address: Option<AssemblyAddress>,
    /// The number of pending users.
    same: usize,
    /// Branches used.
    branches: usize,
    /// If the slot is pending to be built.
    pending: bool,
    /// Is calculated when the first `take_user` is called.
    sealed: Option<SlotSealed>,
}

impl<'hir> SlotStorage<'hir> {
    /// Add a single user.
    fn add_user(&mut self, span: Span, user: Slot, use_kind: UseKind) -> Result<()> {
        let this = self.slot;

        if self.sealed.is_some() {
            return Err(CompileError::msg(
                span,
                format_args!("cannot add user {user:?} to slot {this:?} since it's sealed"),
            ));
        }

        if self.uses.insert(user, use_kind).is_some() {
            return Err(CompileError::msg(
                span,
                format_args!("cannot add user {user:?} to slot {this:?} since it already exists"),
            ));
        }

        match use_kind {
            UseKind::Same => {
                self.same = self.same.saturating_add(1);
            }
            UseKind::Branch => {
                self.branches = self.same.saturating_add(1);
            }
        }

        Ok(())
    }

    /// Remove a single user.
    fn remove_user(&mut self, span: Span, slot: Slot) -> Result<UseKind> {
        let use_kind = match self.uses.remove(&slot) {
            Some(use_kind) => use_kind,
            None => {
                let this = self.slot;
                return Err(CompileError::msg(
                    span,
                    format_args!("{slot:?} is not a user of slot {this:?}"),
                ));
            }
        };

        match use_kind {
            UseKind::Same => {
                self.same = self.same.saturating_sub(1);
            }
            UseKind::Branch => {
                self.branches = self.same.saturating_sub(1);
            }
        }

        Ok(use_kind)
    }

    /// Seal the current slot. This is called automatically when a summary
    /// called for or use is taken.
    fn seal(&mut self) -> SlotSealed {
        *self.sealed.get_or_insert_with(|| SlotSealed {
            total: self.uses.len(),
        })
    }

    /// Get a use summary for the current slot.
    fn use_summary(&mut self) -> UseSummary {
        let sealed = self.seal();

        UseSummary {
            same: self.same,
            branches: self.branches,
            total: sealed.total,
            pending: self.pending,
        }
    }

    /// Mark one use as completed, return the old use count.
    fn take_use(&mut self, span: Span, slot: Slot) -> Result<SlotUse> {
        let sealed = self.seal();
        let use_kind = self.remove_user(span, slot)?;

        match use_kind {
            UseKind::Same => {
                self.same = self.same.saturating_sub(1);
            }
            UseKind::Branch => {
                self.branches = self.branches.saturating_sub(1);
            }
        }

        Ok(SlotUse {
            use_kind,
            summary: UseSummary {
                same: self.same,
                branches: self.branches,
                total: sealed.total,
                pending: self.pending,
            },
        })
    }
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

    /// Reserve the given number of slots at the bottom of the address space.
    fn reserve(&mut self, span: Span, count: usize) -> Result<()> {
        self.bottom = match self.bottom.checked_add(count) {
            Some(bottom) => bottom,
            None => return Err(CompileError::msg(span, "allocator bottom overflow")),
        };

        Ok(())
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
        self.count.saturating_add(self.array_count)
    }
}

/// Test if the expression at the given slot has side effects.
fn has_side_effects(cx: &mut Ctxt<'_, '_>, slot: Slot) -> Result<bool> {
    match cx.slot(slot)?.kind {
        ExprKind::Empty => Ok(false),
        ExprKind::Address { .. } => Ok(true),
        ExprKind::Binding { slot, .. } => has_side_effects(cx, slot),
        ExprKind::TupleFieldAccess { .. } => Ok(true),
        ExprKind::StructFieldAccess { .. } => Ok(true),
        ExprKind::StructFieldAccessGeneric { .. } => Ok(true),
        ExprKind::Assign { .. } => Ok(true),
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
        ExprKind::CallAddress { .. } => Ok(true),
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

/// Assemble the given address.
#[instrument]
fn asm<'hir>(
    cx: &mut Ctxt<'_, 'hir>,
    address: Slot,
    output: AssemblyAddress,
) -> Result<ExprOutcome> {
    let slot_mut = cx.scopes.slot_mut(cx.span, address)?;

    if !slot_mut.pending {
        return Err(CompileError::msg(
            slot_mut.span,
            "slot {slot:?} is being built more than once",
        ));
    }

    slot_mut.pending = false;

    let span = slot_mut.span;
    let kind = slot_mut.kind;
    let this = slot_mut.slot;

    return cx.with_span(span, |cx| {
        match kind {
            ExprKind::Empty => {}
            ExprKind::Address => {
                asm_address(cx, this, output)?;
            }
            ExprKind::Binding { binding, slot, .. } => {
                asm_binding(cx, this, binding, slot, output)?;
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
                asm_unary(cx, this, op, expr, output)?;
            }
            ExprKind::BinaryAssign { lhs, op, rhs } => {
                asm_binary_assign(cx, this, lhs, op, rhs, output)?;
            }
            ExprKind::BinaryConditional { lhs, op, rhs } => {
                asm_binary_conditional(cx, lhs, op, rhs, output)?;
            }
            ExprKind::Binary { lhs, op, rhs } => {
                asm_binary(cx, this, lhs, op, rhs, output)?;
            }
            ExprKind::Index { target, index } => {
                asm_index(cx, this, target, index, output)?;
            }
            ExprKind::Meta { meta, needs, named } => {
                asm_meta(cx, meta, needs, named, output)?;
            }
            ExprKind::Struct { kind, exprs } => {
                asm_struct(cx, this, kind, exprs, output)?;
            }
            ExprKind::Vec { items } => {
                asm_vec(cx, this, items, output)?;
            }
            ExprKind::Range { from, limits, to } => {
                asm_range(cx, this, from, limits, to, output)?;
            }
            ExprKind::Tuple { items } => {
                asm_tuple(cx, this, items, output)?;
            }
            ExprKind::Option { value } => {
                asm_option(cx, this, value, output)?;
            }
            ExprKind::TupleFieldAccess { lhs, index } => {
                asm_tuple_field_access(cx, this, lhs, index, output)?;
            }
            ExprKind::StructFieldAccess { lhs, field, .. } => {
                asm_struct_field_access(cx, this, lhs, field, output)?;
            }
            ExprKind::StructFieldAccessGeneric { .. } => {
                return Err(cx.error(CompileErrorKind::ExpectedExpr));
            }
            ExprKind::Assign {
                binding, lhs, rhs, ..
            } => {
                asm_assign(cx, this, binding, lhs, rhs)?;
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
                asm_call_address(cx, this, function, args, output)?;
            }
            ExprKind::CallHash { hash, args } => {
                asm_call_hash(cx, this, args, hash, output)?;
            }
            ExprKind::CallInstance { lhs, hash, args } => {
                asm_call_instance(cx, lhs, args, hash, output)?;
            }
            ExprKind::CallExpr { expr, args } => {
                asm_call_expr(cx, this, expr, args, output)?;
            }
            ExprKind::Yield { expr } => {
                asm_yield(cx, this, expr, output)?;
            }
            ExprKind::Await { expr } => {
                asm_await(cx, this, expr, output)?;
            }
            ExprKind::Return { expr } => {
                asm_return(cx, this, expr, output)?;
                return Ok(ExprOutcome::Unreachable);
            }
            ExprKind::Try { expr } => {
                asm_try(cx, this, expr, output)?;
            }
            ExprKind::Function { hash } => {
                asm_function(cx, hash, output);
            }
            ExprKind::Closure { hash, captures } => {
                asm_closure(cx, this, captures, hash, output)?;
            }
            ExprKind::Loop {
                condition,
                body,
                start,
                end,
            } => {
                asm_loop(cx, this, condition, body, start, end, output)?;
            }
            ExprKind::Break {
                value,
                label,
                loop_slot,
            } => {
                asm_break(cx, this, value, label, loop_slot)?;
            }
            ExprKind::Continue { label } => {
                asm_continue(cx, label)?;
            }
            ExprKind::StringConcat { exprs } => {
                asm_string_concat(cx, this, exprs, output)?;
            }
            ExprKind::Format { spec, expr } => {
                asm_format(cx, this, spec, expr, output)?;
            }
        }

        Ok(ExprOutcome::Output)
    });

    #[instrument]
    fn asm_address<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        this: Slot,
        output: AssemblyAddress,
    ) -> Result<()> {
        let address = cx.address_for(this)?;

        if address != output {
            cx.push(Inst::Copy { address, output });
        }

        Ok(())
    }

    #[instrument]
    fn asm_binding<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        this: Slot,
        binding: Binding,
        slot: Slot,
        output: AssemblyAddress,
    ) -> Result<()> {
        let user = cx.take_slot_use(slot, this)?;

        // We are the one and only user, so assemble the referenced binding
        // directly on our output.
        if user.is_only() {
            asm(cx, slot, output)?;
            return Ok(());
        }

        let address = cx.alloc_for(slot)?;

        if user.is_pending() {
            asm(cx, slot, address)?;
        }

        let name = cx.scopes.name_to_string(cx.span, binding.name)?;
        let comment = format!("copy `{name}`");
        cx.push_with_comment(Inst::Copy { address, output }, comment);

        // We are the last user, so free the address being held.
        if user.is_last() {
            cx.allocator.free(cx.span, address)?;
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
    fn asm_let<'hir>(cx: &mut Ctxt<'_, 'hir>, pat: Pat<'hir>, expr: Slot) -> Result<()> {
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
        pat: Pat<'hir>,
        expr: Slot,
        label: Label,
    ) -> Result<PatOutcome> {
        let bound_pat = bind_pat(cx, pat, expr)?;
        asm_bound_pat(cx, bound_pat, label)
    }

    #[instrument]
    fn asm_bound_pat<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        bound_pat: BoundPat<'hir>,
        label: Label,
    ) -> Result<PatOutcome> {
        cx.with_span(bound_pat.span, |cx| match bound_pat.kind {
            BoundPatKind::Irrefutable => Ok(PatOutcome::Irrefutable),
            BoundPatKind::IrrefutableSequence { items } => {
                let mut outcome = PatOutcome::Irrefutable;

                for bound_pat in items {
                    outcome = outcome.combine(asm_bound_pat(cx, *bound_pat, label)?);
                }

                Ok(outcome)
            }
            BoundPatKind::Expr {
                name_expr, expr, ..
            } => {
                let summary = cx.slot_mut(name_expr)?.use_summary();

                match (summary.same, summary.branches) {
                    // Assemble eagerly *in case* there are no users.
                    (0, 0) => {
                        // If an expression can have side effects it has to be
                        // executed.
                        if has_side_effects(cx, expr)? {
                            let output = cx.allocator.alloc();
                            asm(cx, expr, output)?;
                            cx.allocator.free(cx.span, output)?;
                        }
                    }
                    // In case there are exactly *1* user, forward this
                    // expression to that user.
                    (1, 0) => {
                        let kind = cx.slot(expr)?.kind;
                        // TODO: also merge users.
                        cx.slot_mut(name_expr)?.kind = kind;
                    }
                    // For more users, assemble this expression on its output
                    // address and replace forward expressions with references
                    // to it.
                    _ => {
                        let output = cx.alloc_for(name_expr)?;
                        asm(cx, expr, output)?;
                        cx.slot_mut(name_expr)?.kind = ExprKind::Address;
                    }
                }

                Ok(PatOutcome::Irrefutable)
            }
            BoundPatKind::Lit { lit, expr } => {
                let user = cx.slot_mut(expr)?.use_summary();
                let address = cx.alloc_for(expr)?;

                if user.is_pending() {
                    asm(cx, expr, address)?;
                }

                match cx.slot(lit)?.kind {
                    ExprKind::Store { value } => {
                        cx.push(Inst::MatchValue {
                            address,
                            value,
                            label,
                        });
                    }
                    ExprKind::String { string } => {
                        let slot = cx.q.unit.new_static_string(cx.span, string)?;
                        cx.push(Inst::MatchString {
                            address,
                            slot,
                            label,
                        });
                    }
                    ExprKind::Bytes { bytes } => {
                        let slot = cx.q.unit.new_static_bytes(cx.span, bytes)?;
                        cx.push(Inst::MatchBytes {
                            address,
                            slot,
                            label,
                        });
                    }
                    _ => {
                        return Err(cx.error(CompileErrorKind::UnsupportedPattern));
                    }
                }

                if user.is_last() {
                    cx.free_for(expr)?;
                }

                Ok(PatOutcome::Refutable)
            }
            BoundPatKind::Vec {
                address,
                expr,
                is_open,
                items,
            } => {
                let expr_use = cx.slot_mut(expr)?.use_summary();
                let expr_output = cx.alloc_for(expr)?;

                if expr_use.is_pending() {
                    asm(cx, expr, expr_output)?;
                }

                cx.push(Inst::MatchSequence {
                    address: expr_output,
                    type_check: TypeCheck::Vec,
                    len: items.len(),
                    exact: !is_open,
                    label,
                    output: address,
                });

                for &pat in items {
                    asm_bound_pat(cx, pat, label)?;
                }

                if expr_use.is_last() {
                    cx.free_for(expr)?;
                }

                Ok(PatOutcome::Refutable)
            }
            BoundPatKind::AnonymousTuple {
                address,
                expr,
                is_open,
                items,
            } => {
                let expr_use = cx.slot_mut(expr)?.use_summary();
                let expr_output = cx.alloc_for(expr)?;

                if expr_use.is_pending() {
                    asm(cx, expr, expr_output)?;
                }

                cx.push(Inst::MatchSequence {
                    address: expr_output,
                    type_check: TypeCheck::Tuple,
                    len: items.len(),
                    exact: !is_open,
                    label,
                    output: address,
                });

                for &pat in items {
                    asm_bound_pat(cx, pat, label)?;
                }

                if expr_use.is_last() {
                    cx.free_for(expr)?;
                }

                Ok(PatOutcome::Refutable)
            }
            BoundPatKind::AnonymousObject {
                address,
                expr,
                slot,
                is_open,
                items,
            } => {
                let expr_use = cx.slot_mut(expr)?.use_summary();
                let expr_output = cx.alloc_for(expr)?;

                if expr_use.is_pending() {
                    asm(cx, expr, expr_output)?;
                }

                cx.push(Inst::MatchObject {
                    address,
                    slot,
                    exact: !is_open,
                    label,
                    output: address,
                });

                for &pat in items {
                    asm_bound_pat(cx, pat, label)?;
                }

                Ok(PatOutcome::Refutable)
            }
            BoundPatKind::TypedSequence {
                type_match,
                expr,
                items,
            } => {
                let expr_use = cx.slot_mut(expr)?.use_summary();
                let expr_output = cx.alloc_for(expr)?;

                if expr_use.is_pending() {
                    asm(cx, expr, expr_output)?;
                }

                match type_match {
                    TypeMatch::BuiltIn { type_check } => cx.push(Inst::MatchBuiltIn {
                        address: expr_output,
                        type_check,
                        label,
                    }),
                    TypeMatch::Type { type_hash } => cx.push(Inst::MatchType {
                        address: expr_output,
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
                            address: expr_output,
                            variant_hash,
                            enum_hash,
                            index,
                            label,
                            output,
                        });
                    }
                }

                for &pat in items {
                    asm_bound_pat(cx, pat, label)?;
                }

                if expr_use.is_last() {
                    cx.free_for(expr)?;
                }

                Ok(PatOutcome::Refutable)
            }
        })
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
        this: Slot,
        op: ExprUnOp,
        expr: Slot,
        output: AssemblyAddress,
    ) -> Result<()> {
        let [address] = cx.addresses(this, [expr], [output])?;

        match op {
            ExprUnOp::Neg => {
                cx.push(Inst::Neg {
                    address: *address,
                    output,
                });
            }
            ExprUnOp::Not => {
                cx.push(Inst::Not {
                    address: *address,
                    output,
                });
            }
        }

        cx.free_iter([address])?;
        Ok(())
    }

    #[instrument]
    fn asm_binary_assign<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        this: Slot,
        lhs: Slot,
        op: ast::BinOp,
        rhs: Slot,
        output: AssemblyAddress,
    ) -> Result<()> {
        let [lhs, rhs] = cx.addresses(this, [lhs, rhs], [output])?;

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
            lhs: *lhs,
            rhs: *rhs,
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

        cx.free_iter([lhs, rhs])?;
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
        this: Slot,
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

        let [lhs, rhs] = cx.addresses(this, [lhs, rhs], [output])?;

        cx.push(Inst::Op {
            op,
            a: *lhs,
            b: *rhs,
            output,
        });

        cx.free_iter([lhs, rhs])?;
        Ok(())
    }

    #[instrument]
    fn asm_index<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        this: Slot,
        target: Slot,
        index: Slot,
        output: AssemblyAddress,
    ) -> Result<()> {
        let [address, index] = cx.addresses(this, [target, index], [output])?;

        cx.push(Inst::IndexGet {
            address: *address,
            index: *index,
            output,
        });

        cx.free_iter([address, index])?;
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
        this: Slot,
        kind: ExprStructKind,
        exprs: &[Slot],
        output: AssemblyAddress,
    ) -> Result<()> {
        let address = cx.array(this, exprs.iter().copied())?;

        match kind {
            ExprStructKind::Anonymous { slot } => {
                cx.push(Inst::Object {
                    address: *address,
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
                    address: *address,
                    slot,
                    output,
                });
            }
            ExprStructKind::StructVariant { hash, slot } => {
                cx.push(Inst::StructVariant {
                    hash,
                    address: *address,
                    slot,
                    output,
                });
            }
        }

        cx.free_iter([address])?;
        Ok(())
    }

    #[instrument]
    fn asm_vec<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        this: Slot,
        items: &[Slot],
        output: AssemblyAddress,
    ) -> Result<()> {
        let address = cx.array(this, items.iter().copied())?;

        cx.push(Inst::Vec {
            address: *address,
            count: address.count(),
            output,
        });

        cx.free_iter([address])?;
        Ok(())
    }

    #[instrument]
    fn asm_range<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        this: Slot,
        from: Slot,
        limits: InstRangeLimits,
        to: Slot,
        output: AssemblyAddress,
    ) -> Result<()> {
        let [from, to] = cx.addresses(this, [from, to], [output])?;

        cx.push(Inst::Range {
            from: *from,
            to: *to,
            limits,
            output,
        });

        cx.free_iter([from, to])?;
        Ok(())
    }

    #[instrument]
    fn asm_tuple<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        this: Slot,
        items: &[Slot],
        output: AssemblyAddress,
    ) -> Result<()> {
        match items {
            &[a] => {
                let [a] = cx.addresses(this, [a], [output])?;

                cx.push(Inst::Tuple1 { args: [*a], output });

                cx.free_iter([a])?;
            }
            &[a, b] => {
                let [a, b] = cx.addresses(this, [a, b], [output])?;

                cx.push(Inst::Tuple2 {
                    args: [*a, *b],
                    output,
                });

                cx.free_iter([a, b])?;
            }
            &[a, b, c] => {
                let [a, b, c] = cx.addresses(this, [a, b, c], [output])?;

                cx.push(Inst::Tuple3 {
                    args: [*a, *b, *c],
                    output,
                });

                cx.free_iter([a, b, c])?;
            }
            &[a, b, c, d] => {
                let [a, b, c, d] = cx.addresses(this, [a, b, c, d], [output])?;

                cx.push(Inst::Tuple4 {
                    args: [*a, *b, *c, *d],
                    output,
                });

                cx.free_iter([a, b, c, d])?;
            }
            args => {
                let address = cx.array(this, args.iter().copied())?;

                cx.push(Inst::Tuple {
                    address: *address,
                    count: address.count(),
                    output,
                });

                cx.free_iter([address])?;
            }
        }

        Ok(())
    }

    #[instrument]
    fn asm_option<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        this: Slot,
        value: Option<Slot>,
        output: AssemblyAddress,
    ) -> Result<()> {
        match value {
            Some(value) => {
                let [address] = cx.addresses(this, [value], [output])?;

                cx.push(Inst::Variant {
                    address: *address,
                    variant: InstVariant::Some,
                    output,
                });

                cx.free_iter([address])?;
            }
            None => {
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
        this: Slot,
        lhs: Slot,
        index: usize,
        output: AssemblyAddress,
    ) -> Result<()> {
        let [address] = cx.addresses(this, [lhs], [output])?;

        cx.push(Inst::TupleIndexGet {
            address: *address,
            index,
            output,
        });

        cx.free_iter([address])?;
        Ok(())
    }

    #[instrument]
    fn asm_struct_field_access<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        this: Slot,
        lhs: Slot,
        field: &str,
        output: AssemblyAddress,
    ) -> Result<()> {
        let [address] = cx.addresses(this, [lhs], [output])?;

        let slot = cx.q.unit.new_static_string(cx.span, field)?;

        cx.push(Inst::ObjectIndexGet {
            address: *address,
            slot,
            output,
        });

        cx.free_iter([address])?;
        Ok(())
    }

    #[instrument]
    fn asm_assign<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        this: Slot,
        binding: Binding,
        lhs: Slot,
        rhs: Slot,
    ) -> Result<()> {
        let lhs_use = cx.take_slot_use(lhs, this)?;
        let rhs_use = cx.take_slot_use(rhs, this)?;

        let lhs_output = cx.alloc_for(lhs)?;

        if rhs_use.is_only() {
            asm(cx, rhs, lhs_output)?;
        } else {
            let rhs_output = cx.alloc_for(rhs)?;

            if rhs_use.is_pending() {
                asm(cx, rhs, rhs_output)?;
            }

            let name = cx.scopes.name_to_string(cx.span, binding.name)?;
            let comment = format!("copy `{name}`");
            cx.push_with_comment(
                Inst::Copy {
                    address: rhs_output,
                    output: lhs_output,
                },
                comment,
            );

            if rhs_use.is_last() {
                cx.free_for(lhs)?;
            }
        }

        if lhs_use.is_last() {
            cx.free_for(lhs)?;
        }

        if rhs_use.is_last() {
            cx.free_for(rhs)?;
        }

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
        this: Slot,
        function: Slot,
        args: &[Slot],
        output: AssemblyAddress,
    ) -> Result<()> {
        let [function] = cx.addresses(this, [function], [output])?;
        let address = cx.array(this, args.iter().copied())?;

        cx.push(Inst::CallFn {
            function: *function,
            address: *address,
            count: address.count(),
            output,
        });

        cx.free_iter([function, address])?;
        Ok(())
    }

    #[instrument]
    fn asm_call_hash<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        this: Slot,
        args: &[Slot],
        hash: Hash,
        output: AssemblyAddress,
    ) -> Result<()> {
        let array = cx.array(this, args.iter().copied())?;

        cx.push(Inst::Call {
            hash,
            address: *array,
            count: array.count(),
            output,
        });

        cx.free_iter([array])?;
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
        this: Slot,
        expr: Slot,
        args: &[Slot],
        output: AssemblyAddress,
    ) -> Result<()> {
        asm(cx, expr, output)?;
        let array = cx.array(this, args.iter().copied())?;

        cx.push(Inst::CallFn {
            function: output,
            address: *array,
            count: array.count(),
            output,
        });

        cx.free_iter([array])?;
        Ok(())
    }

    #[instrument]
    fn asm_yield<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        this: Slot,
        expr: Option<Slot>,
        output: AssemblyAddress,
    ) -> Result<()> {
        Ok(match expr {
            Some(expr) => {
                let [address] = cx.addresses(this, [expr], [output])?;

                cx.push(Inst::Yield {
                    address: output,
                    output,
                });

                cx.free_iter([address])?;
            }
            None => {
                cx.push(Inst::YieldUnit { output });
            }
        })
    }

    #[instrument]
    fn asm_await<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        this: Slot,
        expr: Slot,
        output: AssemblyAddress,
    ) -> Result<()> {
        let [address] = cx.addresses(this, [expr], [output])?;

        cx.push(Inst::Await {
            address: output,
            output,
        });

        cx.free_iter([address])?;
        Ok(())
    }

    #[instrument]
    fn asm_return<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        this: Slot,
        expr: Slot,
        output: AssemblyAddress,
    ) -> Result<()> {
        let [address] = cx.addresses(this, [expr], [output])?;
        cx.push(Inst::Return { address: *address });
        cx.free_iter([address])?;
        Ok(())
    }

    #[instrument]
    fn asm_try<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        this: Slot,
        expr: Slot,
        output: AssemblyAddress,
    ) -> Result<()> {
        let [address] = cx.addresses(this, [expr], [output])?;

        cx.push(Inst::Try {
            address: *address,
            output,
        });

        cx.free_iter([address])?;
        Ok(())
    }

    #[instrument]
    fn asm_function<'hir>(cx: &mut Ctxt<'_, 'hir>, hash: Hash, output: AssemblyAddress) {
        cx.push(Inst::LoadFn { hash, output });
    }

    #[instrument]
    fn asm_closure<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        this: Slot,
        captures: &[Slot],
        hash: Hash,
        output: AssemblyAddress,
    ) -> Result<()> {
        let array = cx.array(this, captures.iter().copied())?;

        cx.push(Inst::Closure {
            hash,
            address: *array,
            count: array.count(),
            output,
        });

        cx.free_iter([array])?;
        Ok(())
    }

    /// Assemble a loop.
    #[instrument]
    fn asm_loop<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        this: Slot,
        condition: LoopCondition<'hir>,
        body: Slot,
        start: Label,
        end: Label,
        output: AssemblyAddress,
    ) -> Result<ExprOutcome> {
        let cleanup = match condition {
            LoopCondition::Forever => {
                cx.label(start)?;
                None
            }
            LoopCondition::Condition { expr, pat } => {
                cx.label(start)?;
                let bound_pat = bind_pat(cx, pat, expr)?;
                asm_bound_pat(cx, bound_pat, end)?;
                None
            }
            LoopCondition::Iterator { iter, binding } => {
                let [iter_var] = cx.addresses(this, [iter], [])?;

                cx.push(Inst::CallInstance {
                    hash: *Protocol::INTO_ITER,
                    address: *iter_var,
                    count: 0,
                    output: *iter_var,
                });

                let value_expr = cx.insert_expr(ExprKind::Empty)?;
                let value_output = cx.alloc_for(value_expr)?;

                cx.push(Inst::IterNext {
                    address: *iter_var,
                    label: end,
                    output: value_output,
                });

                let bound_pat = bind_pat(cx, binding, value_expr)?;
                asm_bound_pat(cx, bound_pat, end)?;
                cx.label(start)?;
                Some(iter_var)
            }
        };

        let outcome = asm(cx, body, output)?;

        cx.push(Inst::Jump { label: start });
        cx.label(end)?;

        if let Some(cleanup) = cleanup {
            cx.free_iter([cleanup])?;
        }

        Ok(outcome)
    }

    /// Assemble a loop break.
    #[instrument]
    fn asm_break<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        this: Slot,
        value: Option<Slot>,
        label: Label,
        loop_slot: Slot,
    ) -> Result<ExprOutcome> {
        if let Some(expr) = value {
            let expr_used = cx.take_slot_use(expr, this)?;
            let loop_slot_used = cx.take_slot_use(loop_slot, this)?;

            let output = cx.alloc_for(loop_slot)?;
            asm(cx, expr, output)?;

            if loop_slot_used.is_last() {
                dbg!("FREE LOOP");
                cx.free_for(loop_slot)?;
            }

            if expr_used.is_last() {
                cx.free_for(expr)?;
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
        this: Slot,
        exprs: &[Slot],
        output: AssemblyAddress,
    ) -> Result<ExprOutcome> {
        let address = cx.array(this, exprs.iter().copied())?;

        cx.push(Inst::StringConcat {
            address: *address,
            count: exprs.len(),
            size_hint: 0,
            output,
        });

        cx.free_iter([address])?;
        Ok(ExprOutcome::Output)
    }

    /// Compile a format expression.
    #[instrument]
    fn asm_format<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        this: Slot,
        spec: &FormatSpec,
        expr: Slot,
        output: AssemblyAddress,
    ) -> Result<ExprOutcome> {
        let [address] = cx.addresses(this, [expr], [output])?;

        cx.push(Inst::Format {
            address: *address,
            spec: *spec,
            output,
        });

        cx.free_iter([address])?;
        Ok(ExprOutcome::Output)
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
                    cx.insert_expr(ExprKind::Let { pat, expr })?
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

        Ok(cx.insert_expr(ExprKind::Block { statements, tail })?)
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
    fn path<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: &'hir hir::Path, needs: Needs) -> Result<Slot> {
        let span = hir.span();
        let loc = Location::new(cx.source_id, span);

        if let Some(ast::PathKind::SelfValue) = hir.as_kind() {
            let (binding, slot, use_kind) = cx.scopes.lookup(loc, cx.scope, cx.q.visitor, SELF)?;

            return Ok(cx.insert_expr(ExprKind::Binding {
                binding,
                slot,
                use_kind,
            })?);
        }

        let named = cx.convert_path(hir)?;

        if let Needs::Value = needs {
            if let Some(local) = named.as_local() {
                let local = local.resolve(resolve_context!(cx.q))?;

                if let Some((binding, slot, use_kind)) =
                    cx.scopes.try_lookup(loc, cx.scope, cx.q.visitor, local)?
                {
                    return Ok(cx.insert_expr(ExprKind::Binding {
                        binding,
                        slot,
                        use_kind,
                    })?);
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
    fn assign<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: &'hir hir::ExprAssign<'hir>) -> Result<Slot> {
        let rhs = expr_value(cx, &hir.rhs)?;
        let this = expr_value(cx, &hir.lhs)?;

        this.map_slot(cx, |cx, kind| match kind {
            ExprKind::Binding {
                binding,
                slot,
                use_kind,
            } => {
                // Handle syntactical reassignments.
                if let UseKind::Same = use_kind {
                    match cx.slot(rhs)?.kind {
                        ExprKind::Binding { slot, .. } => {
                            cx.free_expr(rhs)?;
                            cx.scopes
                                .declare(cx.span, binding.scope, binding.name, slot)?;
                        }
                        _ => {
                            cx.scopes
                                .declare(cx.span, binding.scope, binding.name, rhs)?;
                        }
                    }

                    return Ok(ExprKind::Empty);
                }

                Ok(ExprKind::Assign {
                    binding,
                    lhs: slot,
                    use_kind,
                    rhs,
                })
            }
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
        let this = expr_value(cx, hir.expr)?;

        this.map_slot(cx, |cx, kind| match kind {
            ExprKind::Address { .. } | ExprKind::Binding { .. } => Ok(ExprKind::CallAddress {
                address: this,
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
                        return Ok(cx.slot(value)?.kind);
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
                    return Ok(cx.insert_expr(ExprKind::StructFieldAccess { lhs, field, hash })?);
                }

                if let Some((ident, generics)) = path.try_as_ident_generics() {
                    let n = ident.resolve(resolve_context!(cx.q))?;
                    let hash = Hash::instance_fn_name(n.as_ref());
                    let lhs = expr_value(cx, hir.expr)?;

                    return Ok(cx.insert_expr(ExprKind::StructFieldAccessGeneric {
                        lhs,
                        hash,
                        generics,
                    })?);
                }
            }
            hir::ExprField::LitNumber(field) => {
                let span = field.span();

                let number = field.resolve(resolve_context!(cx.q))?;
                let index = number.as_tuple_index().ok_or_else(|| {
                    CompileError::new(span, CompileErrorKind::UnsupportedTupleIndex { number })
                })?;

                let lhs = expr_value(cx, hir.expr)?;

                return Ok(cx.insert_expr(ExprKind::TupleFieldAccess { lhs, index })?);
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
                    })?);
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
                    })?);
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
        Ok(cx.insert_expr(ExprKind::Unary { op, expr })?)
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
            })?);
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
            })?);
        }

        let lhs = expr_value(cx, hir.lhs)?;
        let rhs = expr(cx, hir.rhs, rhs_needs_of(&hir.op))?;

        return Ok(cx.insert_expr(ExprKind::Binary {
            lhs,
            op: hir.op,
            rhs,
        })?);

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
        Ok(cx.insert_expr(ExprKind::Index { target, index })?)
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
                        let (binding, slot, use_kind) = cx.scopes.lookup(
                            Location::new(cx.source_id, cx.span),
                            cx.scope,
                            cx.q.visitor,
                            &ident.ident,
                        )?;

                        cx.insert_expr(ExprKind::Binding {
                            binding,
                            slot,
                            use_kind,
                        })?
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
    fn return_<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: Option<&'hir hir::Expr<'hir>>) -> Result<Slot> {
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

        Ok(cx.insert_expr(kind)?)
    }

    /// Assemble a yield expression.
    #[instrument]
    fn yield_<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: Option<&'hir hir::Expr>) -> Result<Slot> {
        let expr = match hir {
            Some(hir) => Some(expr_value(cx, hir)?),
            None => None,
        };

        Ok(cx.insert_expr(ExprKind::Yield { expr })?)
    }

    /// Assemble an await expression.
    #[instrument]
    fn await_<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: &'hir hir::Expr) -> Result<Slot> {
        let expr = expr_value(cx, hir)?;
        Ok(cx.insert_expr(ExprKind::Await { expr })?)
    }

    /// Assemble a try expression.
    #[instrument]
    fn try_<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: &'hir hir::Expr) -> Result<Slot> {
        let expr = expr_value(cx, hir)?;
        Ok(cx.insert_expr(ExprKind::Try { expr })?)
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

                let (binding, slot, use_kind) = cx.scopes.lookup(
                    Location::new(cx.source_id, cx.span),
                    cx.scope,
                    cx.q.visitor,
                    &capture.ident,
                )?;

                cx.insert_expr(ExprKind::Binding {
                    binding,
                    slot,
                    use_kind,
                })?
            });

            ExprKind::Closure { hash, captures }
        };

        Ok(cx.insert_expr(kind)?)
    }

    /// Construct a literal value.
    #[instrument]
    fn lit<'hir>(cx: &mut Ctxt<'_, 'hir>, ast: &'hir ast::Lit) -> Result<Slot> {
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
            Some(hir) => path(cx, hir, Needs::Type)?.map_slot(cx, |cx, kind| {
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
    fn tuple<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: &hir::ExprSeq<'hir>) -> Result<Slot> {
        let items = iter!(cx; hir.items, |hir| expr_value(cx, hir)?);
        Ok(cx.insert_expr(ExprKind::Tuple { items })?)
    }

    /// Assemble a vector expression.
    #[instrument]
    fn vec<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: &hir::ExprSeq<'hir>) -> Result<Slot> {
        let items = iter!(cx; hir.items, |hir| expr_value(cx, hir)?);
        Ok(cx.insert_expr(ExprKind::Vec { items })?)
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
    fn lit_number<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: &ast::LitNumber) -> Result<Slot> {
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
    fn loop_<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: &hir::ExprLoop<'hir>) -> Result<Slot> {
        let label = match hir.label {
            Some(label) => Some(cx.scopes.name(label.resolve(resolve_context!(cx.q))?)),
            None => None,
        };

        let start = cx.new_label("loop_start");
        let end = cx.new_label("loop_end");
        let loop_slot = cx.insert_expr(ExprKind::Empty)?;

        let scope = cx
            .scopes
            .push_loop(cx.span, Some(cx.scope), label, start, end, loop_slot)?;

        let condition = match hir.condition {
            hir::LoopCondition::Forever => LoopCondition::Forever,
            hir::LoopCondition::Condition { condition } => {
                let (expr, pat) = match *condition {
                    hir::Condition::Expr(hir) => {
                        let lit = cx.insert_expr(ExprKind::Store {
                            value: InstValue::Bool(true),
                        })?;
                        (expr_value(cx, hir)?, cx.insert_pat(PatKind::Lit { lit }))
                    }
                    hir::Condition::ExprLet(hir) => {
                        let expr = expr_value(cx, hir.expr)?;
                        let pat = cx.with_scope(scope, |cx| pat(cx, hir.pat))?;
                        (expr, pat)
                    }
                };

                LoopCondition::Condition { expr, pat }
            }
            hir::LoopCondition::Iterator { binding, iter } => {
                let iter = expr_value(cx, iter)?;
                let binding = cx.with_scope(scope, |cx| pat(cx, binding))?;
                LoopCondition::Iterator { binding, iter }
            }
        };

        let body = cx.with_scope(scope, |cx| block(cx, hir.body))?;

        cx.scopes.pop(cx.span, scope)?;

        loop_slot.map_slot(cx, |_, _| {
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
    fn break_<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: &hir::ExprBreakValue<'hir>) -> Result<Slot> {
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
    fn continue_<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: Option<&ast::Label>) -> Result<Slot> {
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
    fn macro_call<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: &hir::MacroCall<'hir>) -> Result<Slot> {
        let slot = match hir {
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

        Ok(slot)
    }

    /// Assemble #[builtin] template!(...) macro.
    #[instrument]
    fn builtin_template<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        hir: &hir::BuiltInTemplate<'hir>,
    ) -> Result<Slot> {
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
    ) -> Result<Slot> {
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

    Ok(cx.insert_expr(kind)?)
}

/// Assemble a pattern.
fn pat<'hir>(cx: &mut Ctxt<'_, 'hir>, hir: &hir::Pat<'hir>) -> Result<Pat<'hir>> {
    let mut removed = HashMap::new();
    return pat(cx, hir, &mut removed);

    #[instrument]
    fn pat<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        hir: &hir::Pat<'hir>,
        removed: &mut HashMap<Name, Span>,
    ) -> Result<Pat<'hir>> {
        return cx.with_span(hir.span(), |cx| {
            let slot = match hir.kind {
                hir::PatKind::PatPath(hir) => pat_binding(cx, hir, removed)?,
                hir::PatKind::PatIgnore => cx.insert_pat(PatKind::Ignore),
                hir::PatKind::PatLit(hir) => {
                    let lit = expr_value(cx, hir)?;
                    cx.insert_pat(PatKind::Lit { lit })
                }
                hir::PatKind::PatVec(hir) => {
                    let items = iter!(cx; hir.items, |hir| pat(cx, hir, removed)?);

                    cx.insert_pat(PatKind::Vec {
                        items,
                        is_open: hir.is_open,
                    })
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

        let patterns = iter!(cx; hir.items, |hir| pat(cx, hir, removed)?);

        Ok(cx.insert_pat(PatKind::Tuple {
            kind,
            patterns,
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

        let patterns = iter!(cx; hir.bindings, |binding| {
            if let Some(hir) = binding.pat {
                pat(cx, hir, removed)?
            } else {
                pat_object_key(cx, binding.key, removed)?
            }
        });

        Ok(cx.insert_pat(PatKind::Object { kind, patterns }))
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
        let pat = match pat_path(cx, hir)? {
            PatPath::Name { name } => {
                let name_expr = cx.insert_expr(ExprKind::Empty)?;

                {
                    let name = cx.scopes.name(name);
                    let replaced = cx.scopes.declare(cx.span, cx.scope, name, name_expr)?;

                    if replaced.is_some() {
                        if let Some(span) = removed.insert(name, cx.span) {
                            return Err(cx.error(CompileErrorKind::DuplicateBinding {
                                previous_span: span,
                            }));
                        }
                    }
                }

                cx.insert_pat(PatKind::Name { name, name_expr })
            }
            PatPath::Meta { meta } => cx.insert_pat(PatKind::Meta { meta }),
        };

        Ok(pat)
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
fn bind_pat<'hir>(cx: &mut Ctxt<'_, 'hir>, pat: Pat<'hir>, expr: Slot) -> Result<BoundPat<'hir>> {
    return cx.with_span(pat.span, |cx| match pat.kind {
        PatKind::Ignore => Ok(cx.bound_pat(BoundPatKind::Irrefutable)),
        PatKind::Lit { lit } => bind_pat_lit(cx, lit, expr),
        PatKind::Name { name, name_expr } => Ok(cx.bound_pat(BoundPatKind::Expr {
            name,
            name_expr,
            expr,
        })),
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
        // Match irrefutable patterns.
        match (cx.slot(lit)?.kind, cx.slot(expr)?.kind) {
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

        Ok(cx.bound_pat(BoundPatKind::Lit { lit, expr }))
    }

    /// Bind a vector pattern.
    #[instrument]
    fn bind_pat_vec<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        patterns: &'hir [Pat<'hir>],
        is_open: bool,
        expr: Slot,
    ) -> Result<BoundPat<'hir>> {
        // Try a simpler form of pattern matching through syntactical reassignment.
        match cx.slot(expr)?.kind {
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
            let expr = cx.insert_expr_with_address(ExprKind::Address, cx.allocator.array_address())?;
            bind_pat(
                cx,
                *pat,
                expr,
            )?
        });

        Ok(cx.bound_pat(BoundPatKind::Vec {
            address,
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
        patterns: &'hir [Pat<'hir>],
        is_open: bool,
        expr: Slot,
    ) -> Result<BoundPat<'hir>> {
        match kind {
            PatTupleKind::Typed { type_match } => {
                let items = iter!(cx; patterns.iter().enumerate().rev(), |(index, pat)| {
                    let expr = cx.insert_expr(ExprKind::TupleFieldAccess {
                        lhs: expr,
                        index,
                    })?;

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
                match cx.slot(expr)?.kind {
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

                let address = cx.allocator.array_address();
                cx.allocator.alloc_array_items(patterns.len());

                // NB we bind the arguments in reverse to allow for higher elements
                // in the array to be freed up for subsequent bindings.
                let items = iter!(cx; patterns.iter().rev(), |pat| {
                    cx.allocator.free_array_item(cx.span)?;
                    let expr = cx.insert_expr_with_address(ExprKind::Address, cx.allocator.array_address())?;
                    bind_pat(cx, *pat, expr)?
                });

                Ok(cx.bound_pat(BoundPatKind::AnonymousTuple {
                    expr,
                    is_open,
                    items,
                    address,
                }))
            }
        }
    }

    /// Set up binding for pattern objects.
    #[instrument]
    fn bind_pat_object<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        kind: PatObjectKind<'hir>,
        patterns: &'hir [Pat<'hir>],
        expr: Slot,
    ) -> Result<BoundPat<'hir>> {
        match kind {
            PatObjectKind::Typed { type_match, keys } => {
                let items = iter!(cx; keys.iter().zip(patterns), |(&(field, hash), pat)| {
                    let expr = cx.insert_expr(ExprKind::StructFieldAccess {
                        lhs: expr,
                        field,
                        hash,
                    })?;

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
                match cx.slot(expr)?.kind {
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

                let address = cx.allocator.array_address();
                cx.allocator.alloc_array_items(patterns.len());

                // NB we bind the arguments in reverse to allow for higher elements
                // in the array to be freed up for subsequent bindings.
                let items = iter!(cx; patterns.iter().rev(), |pat| {
                    cx.allocator.free_array_item(cx.span)?;
                    let expr = cx.insert_expr_with_address(ExprKind::Address, cx.allocator.array_address())?;
                    bind_pat(cx, *pat, expr)?
                });

                Ok(cx.bound_pat(BoundPatKind::AnonymousObject {
                    address,
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

/// Walk over an expression to modify each expression that it touches.
fn walk_expr_slots<T>(kind: ExprKind<'_>, mut op: T) -> Result<()>
where
    T: FnMut(Slot, UseKind) -> Result<()>,
{
    match kind {
        ExprKind::Empty => {}
        ExprKind::Address => {}
        ExprKind::Binding { slot, use_kind, .. } => {
            op(slot, use_kind)?;
        }
        ExprKind::Assign {
            lhs, use_kind, rhs, ..
        } => {
            op(lhs, use_kind)?;
            op(rhs, UseKind::Same)?;
        }
        ExprKind::TupleFieldAccess { lhs, .. } => {
            op(lhs, UseKind::Same)?;
        }
        ExprKind::StructFieldAccess { lhs, .. } => {
            op(lhs, UseKind::Same)?;
        }
        ExprKind::StructFieldAccessGeneric { lhs, .. } => {
            op(lhs, UseKind::Same)?;
        }
        ExprKind::AssignStructField { lhs, rhs, .. } => {
            op(lhs, UseKind::Same)?;
            op(rhs, UseKind::Same)?;
        }
        ExprKind::AssignTupleField { lhs, rhs, .. } => {
            op(lhs, UseKind::Same)?;
            op(rhs, UseKind::Same)?;
        }
        ExprKind::Block { .. } => {
            // NB: statements in a block intentionally has *no* users, to
            // ensure that they are swept away during optimization unless
            // they have side effects.
        }
        ExprKind::Let { expr, .. } => {
            op(expr, UseKind::Same)?;
        }
        ExprKind::Store { .. } => {}
        ExprKind::Bytes { .. } => {}
        ExprKind::String { .. } => {}
        ExprKind::Unary { expr, .. } => {
            op(expr, UseKind::Same)?;
        }
        ExprKind::BinaryAssign { lhs, rhs, .. } => {
            op(lhs, UseKind::Same)?;
            op(rhs, UseKind::Same)?;
        }
        ExprKind::BinaryConditional { lhs, rhs, .. } => {
            op(lhs, UseKind::Same)?;
            op(rhs, UseKind::Same)?;
        }
        ExprKind::Binary { lhs, rhs, .. } => {
            op(lhs, UseKind::Same)?;
            op(rhs, UseKind::Same)?;
        }
        ExprKind::Index { target, .. } => {
            op(target, UseKind::Same)?;
        }
        ExprKind::Meta { .. } => {}
        ExprKind::Struct { exprs, .. } => {
            for &slot in exprs {
                op(slot, UseKind::Same)?;
            }
        }
        ExprKind::Tuple { items } => {
            for &slot in items {
                op(slot, UseKind::Same)?;
            }
        }
        ExprKind::Vec { items } => {
            for &slot in items {
                op(slot, UseKind::Same)?;
            }
        }
        ExprKind::Range { from, to, .. } => {
            op(from, UseKind::Same)?;
            op(to, UseKind::Same)?;
        }
        ExprKind::Option { value } => {
            if let Some(slot) = value {
                op(slot, UseKind::Same)?;
            }
        }
        ExprKind::CallAddress { address, args } => {
            op(address, UseKind::Same)?;

            for &slot in args {
                op(slot, UseKind::Same)?;
            }
        }
        ExprKind::CallHash { args, .. } => {
            for &slot in args {
                op(slot, UseKind::Same)?;
            }
        }
        ExprKind::CallInstance { lhs, args, .. } => {
            op(lhs, UseKind::Same)?;

            for &slot in args {
                op(slot, UseKind::Same)?;
            }
        }
        ExprKind::CallExpr { expr, args } => {
            op(expr, UseKind::Same)?;

            for &slot in args {
                op(slot, UseKind::Same)?;
            }
        }
        ExprKind::Yield { expr } => {
            if let Some(slot) = expr {
                op(slot, UseKind::Same)?;
            }
        }
        ExprKind::Await { expr } => {
            op(expr, UseKind::Same)?;
        }
        ExprKind::Return { expr } => {
            op(expr, UseKind::Same)?;
        }
        ExprKind::Try { expr } => {
            op(expr, UseKind::Same)?;
        }
        ExprKind::Function { .. } => {}
        ExprKind::Closure { captures, .. } => {
            for &slot in captures {
                op(slot, UseKind::Same)?;
            }
        }
        ExprKind::Loop { body, .. } => {
            op(body, UseKind::Same)?;
        }
        ExprKind::Break {
            value, loop_slot, ..
        } => {
            if let Some(slot) = value {
                op(slot, UseKind::Same)?;
            }

            op(loop_slot, UseKind::Branch)?;
        }
        ExprKind::Continue { .. } => {}
        ExprKind::StringConcat { exprs } => {
            for &slot in exprs {
                op(slot, UseKind::Same)?;
            }
        }
        ExprKind::Format { expr, .. } => {
            op(expr, UseKind::Same)?;
        }
    }

    Ok(())
}
