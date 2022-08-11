use std::fmt;
use std::mem;
use std::num::NonZeroUsize;

use crate::ast::Span;
use crate::collections::HashMap;
use crate::compile::{
    AssemblyAddress, CompileError, CompileErrorKind, CompileVisitor, Label, Location,
};
use crate::runtime::Address;

type Result<T> = ::std::result::Result<T, CompileError>;

/// Scopes to use when compiling.
#[derive(Default)]
pub(crate) struct Scopes {
    /// Stack of scopes.
    scopes: slab::Slab<Scope>,
    /// Keeping track of every slot.
    slots: slab::Slab<Slot>,
    /// Keep track of the total number of slots used.
    count: usize,
    /// Maximum number of array elements.
    array_index: usize,
    /// The current array count.
    array_count: usize,
    /// The collection of known names in the scope, so that we can generate the
    /// [BindingName] identifier.
    names_by_id: Vec<Box<str>>,
    names_by_name: HashMap<Box<str>, BindingName>,
}

impl Scopes {
    /// Construct an empty scope.
    pub(crate) fn new() -> Self {
        Self {
            scopes: slab::Slab::new(),
            slots: slab::Slab::new(),
            count: 0,
            array_index: 0,
            array_count: 0,
            names_by_id: Vec::new(),
            names_by_name: HashMap::new(),
        }
    }

    /// Perform a shallow clone of a scope (without updating users) and return the id of the cloned scope.
    pub(crate) fn clone_scope(&mut self, span: Span, scope: ScopeId) -> Result<ScopeId> {
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
    pub(crate) fn replace_scope(
        &mut self,
        span: Span,
        old_scope: ScopeId,
        new_scope: ScopeId,
    ) -> Result<()> {
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

    /// The size of the frame.
    pub(crate) fn frame(&self) -> usize {
        self.count.saturating_add(self.array_count)
    }

    /// Reassign one binding to another.
    #[tracing::instrument(skip_all)]
    pub(crate) fn assign(
        &mut self,
        span: Span,
        scope: ScopeId,
        binding: Binding,
        address: AssemblyAddress,
    ) -> Result<bool> {
        tracing::trace!(binding = ?binding, address = ?address);

        let mut current = Some(scope);

        while let Some((id, scope)) = current
            .take()
            .and_then(|s| Some((s, self.scopes.get(s.0)?)))
        {
            // We don't support scoped reassignments across control flows, since
            // that would cause errors.
            if !matches!(scope.control_flow, ControlFlow::None) {
                return Ok(false);
            }

            if id == binding.scope {
                break;
            }

            current = scope.parent;
        }

        let address = match address {
            AssemblyAddress::Slot(slot) => slot,
            _ => return Ok(false),
        };

        let scope = match self.scopes.get_mut(binding.scope.0) {
            Some(scope) => scope,
            None => {
                return Err(CompileError::msg(span, "missing scope"));
            }
        };

        let replaced = match scope.names.get_mut(&binding.name) {
            Some(value) => mem::replace(value, AssemblyAddress::Slot(address)),
            None => {
                return Err(CompileError::msg(span, "missing binding in scope"));
            }
        };

        // Note that we need to retain first in case we end up retaining the
        // same address as the one replaced.
        self.retain_user(span, address)?;

        if let AssemblyAddress::Slot(slot) = replaced {
            self.free_user(span, slot)?;
        }

        Ok(true)
    }

    /// Allocate an assembly address. It starts out with 1 user. Once this
    /// reaches 0 through the use of [free][Scopes::free] or the like the
    /// associated slot will be deallocated.
    #[tracing::instrument(skip_all)]
    pub(crate) fn alloc(&mut self) -> AssemblyAddress {
        let slot = self.slots.insert(Slot { users: 1 });
        self.count = self.slots.len().max(self.count);
        let address = AssemblyAddress::Slot(slot);
        tracing::trace!(count = self.count, address = ?address);
        address
    }

    /// Get a temporary address that will immediately be deallocated. It's only
    /// suitable for use withing a single instruction where no other slot
    /// allocations occur.
    #[tracing::instrument(skip_all)]
    pub(crate) fn temporary(&mut self) -> AssemblyAddress {
        let address = AssemblyAddress::Slot(self.slots.vacant_key());
        self.count = self.slots.len().saturating_add(1).max(self.count);
        tracing::trace!(count = self.count, address = ?address);
        address
    }

    /// Get the current array index as an assembly address.
    pub(crate) fn array_index(&self) -> AssemblyAddress {
        AssemblyAddress::Array(self.array_index)
    }

    /// Mark multiple array items as occupied.
    #[tracing::instrument(skip_all)]
    pub(crate) fn alloc_array_items(&mut self, len: usize) {
        self.array_index = self.array_index.saturating_add(len);
        self.array_count = self.array_count.max(self.array_index);
        tracing::trace!(?self.array_count, ?self.array_index);
    }

    /// Mark one array item as occupied, forcing additional allocations to
    /// utilize higher array indexes.
    #[tracing::instrument(skip_all)]
    pub(crate) fn alloc_array_item(&mut self) {
        self.alloc_array_items(1)
    }

    /// Allocate an array item.
    #[tracing::instrument(skip_all)]
    pub(crate) fn free_array(&mut self, span: Span, items: usize) -> Result<()> {
        self.array_index = match self.array_index.checked_sub(items) {
            Some(array) => array,
            None => return Err(CompileError::msg(span, "scope array index out-of-bounds")),
        };

        tracing::trace!(?self.array_count, ?self.array_index);
        Ok(())
    }

    /// Free a single array item.
    #[tracing::instrument(skip_all)]
    pub(crate) fn free_array_item(&mut self, span: Span) -> Result<()> {
        self.free_array(span, 1)
    }

    /// Free up one user for the given slot.
    #[tracing::instrument(skip_all)]
    pub(crate) fn free(&mut self, span: Span, address: AssemblyAddress) -> Result<()> {
        tracing::trace!(address = ?address);

        if let AssemblyAddress::Slot(slot) = address {
            self.free_user(span, slot)?;
        }

        Ok(())
    }

    /// Retain the given address.
    #[tracing::instrument(skip_all)]
    pub(crate) fn retain(&mut self, span: Span, address: AssemblyAddress) -> Result<()> {
        tracing::trace!(address = ?address);

        if let AssemblyAddress::Slot(slot) = address {
            self.retain_user(span, slot)?;
        }

        Ok(())
    }

    /// Declare a variable.
    #[tracing::instrument(skip_all)]
    pub(crate) fn declare(
        &mut self,
        span: Span,
        scope: ScopeId,
        binding_name: BindingName,
    ) -> Result<(Binding, AssemblyAddress)> {
        let slot = self.slots.insert(Slot { users: 0 });
        self.count = self.slots.len().max(self.count);
        let address = AssemblyAddress::Slot(slot);
        let (binding, _) = self.declare_as(span, scope, binding_name, address)?;
        Ok((binding, address))
    }

    /// Declare a variable with an already known address.
    #[tracing::instrument(skip_all)]
    pub(crate) fn declare_as(
        &mut self,
        span: Span,
        scope: ScopeId,
        binding_name: BindingName,
        address: AssemblyAddress,
    ) -> Result<(Binding, Option<AssemblyAddress>)> {
        let top = match self.scopes.get_mut(scope.0) {
            Some(scope) => scope,
            None => {
                return Err(CompileError::msg(
                    span,
                    "missing scope to declare variable in",
                ))
            }
        };

        tracing::trace!(?binding_name, ?address);

        let replaced = top.names.insert(binding_name, address);

        // Note that we need to retain *first*, in case we are retaining the
        // same address as the one being freed.
        if let AssemblyAddress::Slot(slot) = address {
            self.retain_user(span, slot)?;
        }

        let binding = Binding {
            scope,
            name: binding_name,
        };
        Ok((binding, replaced))
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
    pub(crate) fn push(&mut self, span: Span, parent: Option<ScopeId>) -> Result<ScopeId> {
        self.push_inner(span, parent, ControlFlow::None)
    }

    /// Push a new scope with the given control flow flag.
    #[tracing::instrument(skip(self))]
    pub(crate) fn push_loop(
        &mut self,
        span: Span,
        parent: Option<ScopeId>,
        label: Option<BindingName>,
        start: Label,
        end: Label,
        output: Option<AssemblyAddress>,
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
    pub(crate) fn push_branch(&mut self, span: Span, parent: Option<ScopeId>) -> Result<ScopeId> {
        self.push_inner(span, parent, ControlFlow::Branch)
    }

    /// Pop the last scope.
    #[tracing::instrument(skip_all)]
    pub(crate) fn pop(&mut self, span: Span, id: ScopeId) -> Result<()> {
        let scope = match self.scopes.try_remove(id.0) {
            Some(scope) => scope,
            None => return Err(CompileError::msg(span, "missing scope")),
        };

        tracing::trace!(?id, ?scope);

        if scope.children != 0 {
            return Err(CompileError::msg(
                span,
                "tried to remove scope which still has children",
            ));
        }

        if let Some(parent) = scope.parent.and_then(|p| self.scopes.get_mut(p.0)) {
            parent.children = parent.children.saturating_sub(1);
        }

        for (name, address) in scope.names {
            let s = tracing::trace_span!("pop_free_user", name = ?name);
            let _enter = s.enter();

            if let AssemblyAddress::Slot(slot) = address {
                self.free_user(span, slot)?;
            }
        }

        Ok(())
    }

    /// Try to get the local with the given name. Returns `None` if it's
    /// missing.
    #[tracing::instrument(skip_all)]
    pub(crate) fn try_lookup(
        &mut self,
        _: Location,
        scope: ScopeId,
        _: &mut dyn CompileVisitor,
        name: &str,
    ) -> Result<Option<(Binding, AssemblyAddress)>> {
        let binding_name = self.binding_name(name);
        tracing::trace!(name = name, binding_name = ?binding_name);

        let mut current = Some(scope);

        while let Some((id, scope)) = current
            .take()
            .and_then(|s| Some((s, self.scopes.get(s.0)?)))
        {
            if let Some(address) = scope.lookup(binding_name) {
                tracing::trace!("found: {name:?} => {address:?}");
                // TODO: Support visit variable use somehow.
                // visitor.visit_variable_use(location.source_id, location.span);

                let binding = Binding {
                    scope: id,
                    name: binding_name,
                };

                return Ok(Some((binding, address)));
            }

            current = scope.parent;
        }

        Ok(None)
    }

    /// Lookup the given variable.
    #[tracing::instrument(skip_all)]
    pub(crate) fn lookup(
        &mut self,
        location: Location,
        scope: ScopeId,
        visitor: &mut dyn CompileVisitor,
        name: &str,
    ) -> Result<(Binding, AssemblyAddress)> {
        match self.try_lookup(location, scope, visitor, name)? {
            Some(address) => Ok(address),
            None => Err(CompileError::new(
                location.span,
                CompileErrorKind::MissingLocal {
                    name: name.to_owned(),
                },
            )),
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

    /// Find the first ancestor that matches the given predicate.
    #[tracing::instrument(skip(filter))]
    pub(crate) fn find_ancestor<T>(&self, scope: ScopeId, mut filter: T) -> Option<ControlFlow>
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

    /// Free the user of a slot.
    #[tracing::instrument(skip_all)]
    pub(crate) fn free_user(&mut self, span: Span, slot: usize) -> Result<()> {
        self.modify_user(span, slot, |users| users.checked_sub(1))
    }

    /// Alloc a user for the specified slot.
    #[tracing::instrument(skip_all)]
    fn retain_user(&mut self, span: Span, slot: usize) -> Result<()> {
        self.modify_user(span, slot, |users| users.checked_add(1))
    }

    fn modify_user(
        &mut self,
        span: Span,
        slot: usize,
        apply: impl FnOnce(usize) -> Option<usize>,
    ) -> Result<()> {
        let data = match self.slots.get_mut(slot) {
            Some(data) => data,
            None => {
                tracing::trace!(slot = slot, "missing");
                return Err(CompileError::msg(span, "slot missing for modification"));
            }
        };

        tracing::trace!(slot = slot, users = data.users);

        data.users = match apply(data.users) {
            Some(users) => users,
            None => {
                return Err(CompileError::msg(span, "slot users out of bounds"));
            }
        };

        if data.users == 0 {
            self.slots.remove(slot);
        }

        Ok(())
    }

    /// Get the name of a binding.
    pub(crate) fn name(&self, span: Span, binding_name: BindingName) -> Result<&str> {
        match self.names_by_id.get(binding_name.index()) {
            Some(name) => Ok(name),
            None => Err(CompileError::new(
                span,
                CompileErrorKind::MissingBindingName { binding_name },
            )),
        }
    }

    /// Translate the name of a binding.
    pub(crate) fn binding_name(&mut self, name: &str) -> BindingName {
        if let Some(binding_name) = self.names_by_name.get(name) {
            return *binding_name;
        }

        let binding_name = BindingName::new(self.names_by_id.len()).expect("ran out of name slots");
        self.names_by_id.push(name.into());
        self.names_by_name.insert(name.into(), binding_name);
        binding_name
    }
}

impl fmt::Debug for Scopes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        return f
            .debug_struct("Scopes")
            .field("scopes", &ScopesIter(&self.scopes))
            .field("slots", &ScopesIter(&self.slots))
            .field("count", &self.count)
            .field("array_index", &self.array_index)
            .field("array_count", &self.array_count)
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
pub(crate) struct ScopeId(usize);

impl ScopeId {
    /// The constant scope.
    pub(crate) const CONST: Self = Self(usize::MAX);
}

/// A control flow.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) enum ControlFlow {
    #[default]
    None,
    Loop(LoopControlFlow),
    Branch,
}

/// A loop control flow.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub(crate) struct LoopControlFlow {
    pub(crate) label: Option<BindingName>,
    pub(crate) start: Label,
    pub(crate) end: Label,
    pub(crate) output: Option<AssemblyAddress>,
}

#[derive(Debug, Clone, Default)]
struct Scope {
    parent: Option<ScopeId>,
    names: HashMap<BindingName, AssemblyAddress>,
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
    fn lookup<'data>(&self, binding_name: BindingName) -> Option<AssemblyAddress> {
        Some(*self.names.get(&binding_name)?)
    }
}

/// The name of a binding.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct BindingName(NonZeroUsize);

impl fmt::Debug for BindingName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("BindingName").field(&self.index()).finish()
    }
}

impl BindingName {
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
pub(crate) struct Binding {
    pub(crate) scope: ScopeId,
    pub(crate) name: BindingName,
}

#[derive(Debug)]
struct Slot {
    /// The number of users this slot has. Either anonymous, or named in a scope.
    users: usize,
}
