use crate::collections::HashMap;
use crate::macros::{MacroContext, TokenStream};
use crate::runtime::vm::{CallOffset, CallResult, CallResultOrOffset};
use crate::runtime::{
    Address, ConstValue, GuardedArgs, Protocol, Stack, Unit, UnitFn, UnsafeToValue, VmError,
};
use crate::{Hash, IntoTypeHash};
use std::fmt;
use std::sync::Arc;

/// A type-reduced function handler.
pub(crate) type FunctionHandler =
    dyn Fn(&mut Stack, Address, usize, Address) -> Result<(), VmError> + Send + Sync;

/// A (type erased) macro handler.
pub(crate) type MacroHandler =
    dyn Fn(&mut MacroContext, &TokenStream) -> crate::Result<TokenStream> + Send + Sync;

/// Static run context visible to the virtual machine.
///
/// This contains:
/// * Declared functions.
/// * Declared instance functions.
/// * Built-in type checks.
#[derive(Default, Clone)]
pub struct RuntimeContext {
    /// Registered native function handlers.
    functions: HashMap<Hash, Arc<FunctionHandler>>,
    /// Named constant values
    constants: HashMap<Hash, ConstValue>,
}

impl RuntimeContext {
    pub(crate) fn new(
        functions: HashMap<Hash, Arc<FunctionHandler>>,
        constants: HashMap<Hash, ConstValue>,
    ) -> Self {
        Self {
            functions,
            constants,
        }
    }

    /// Lookup the given native function handler in the context.
    pub fn function(&self, hash: Hash) -> Option<&Arc<FunctionHandler>> {
        self.functions.get(&hash)
    }

    /// Read a constant value from the unit.
    pub fn constant(&self, hash: Hash) -> Option<&ConstValue> {
        self.constants.get(&hash)
    }

    /// Try and perform a context function call with a prepared stack.
    #[tracing::instrument(skip_all)]
    pub(crate) fn fn_call(
        &self,
        hash: Hash,
        stack: &mut Stack,
        address: Address,
        args: usize,
        output: Address,
    ) -> Result<bool, VmError> {
        let handler = match self.function(hash) {
            Some(handler) => handler,
            None => {
                tracing::trace!("missing handle");
                return Ok(false);
            }
        };

        tracing::trace!("calling handler");
        handler(stack, address, args, output)?;
        Ok(true)
    }

    /// Helper to call an index function.
    pub(crate) fn call_index_fn<V, A>(
        &self,
        stack: &mut Stack,
        protocol: Protocol,
        value: V,
        index: usize,
        args: A,
        output: Address,
    ) -> Result<CallResultOrOffset<()>, VmError>
    where
        V: UnsafeToValue,
        A: GuardedArgs,
    {
        // SAFETY: We hold onto the guard for the duration of this call.
        let (value, _value_guard) = unsafe { value.unsafe_to_value()? };

        let address = stack.top()?;
        let full_count = args.count() + 1;
        let hash = Hash::index_fn(protocol, value.type_hash()?, Hash::index(index));

        stack.push(value);

        // SAFETY: We hold onto the guard for the duration of this call.
        let _guard = unsafe { args.unsafe_into_stack(stack)? };

        if !self.fn_call(hash, stack, address, full_count, output)? {
            // Restore the stack since no one has touched it.
            stack.pop_n(full_count)?;
            return Ok(CallResultOrOffset::Unsupported);
        }

        Ok(CallResultOrOffset::Ok)
    }

    /// Helper to call a field function.
    pub(crate) fn call_field_fn<V, H, A>(
        &self,
        stack: &mut Stack,
        protocol: Protocol,
        value: V,
        hash: H,
        args: A,
        output: Address,
    ) -> Result<CallResult<()>, VmError>
    where
        V: UnsafeToValue,
        H: IntoTypeHash,
        A: GuardedArgs,
    {
        // SAFETY: We hold onto the guard for the duration of this call.
        let (value, _value_guard) = unsafe { value.unsafe_to_value()? };

        let address = stack.top()?;
        let full_count = args.count() + 1;
        let hash = Hash::field_fn(protocol, value.type_hash()?, hash.into_type_hash());

        stack.push(value.clone());

        // SAFETY: We hold onto the guard for the duration of this call.
        let _guard = unsafe { args.unsafe_into_stack(stack)? };

        if !self.fn_call(hash, stack, address, full_count, output)? {
            // Restore the stack since no one has touched it.
            stack.pop_n(full_count)?;
            return Ok(CallResult::Unsupported);
        }

        Ok(CallResult::Ok(()))
    }

    /// Helper function to call an instance function.
    pub(crate) fn call_instance_fn<V, H, A>(
        &self,
        stack: &mut Stack,
        unit: &Unit,
        value: V,
        hash: H,
        args: A,
        output: Address,
    ) -> Result<CallResultOrOffset<(V::Guard, A::Guard)>, VmError>
    where
        V: UnsafeToValue,
        H: IntoTypeHash,
        A: GuardedArgs,
    {
        // SAFETY: We hold onto the guard for the duration of this call.
        let (value, value_guard) = unsafe { value.unsafe_to_value()? };

        let full_count = args.count() + 1;
        let type_hash = value.type_hash()?;

        let address = stack.top()?;
        stack.push(value);

        // SAFETY: We hold onto the guard for the duration of this call.
        let args_guard = unsafe { args.unsafe_into_stack(stack)? };

        let hash = Hash::instance_function(type_hash, hash.into_type_hash());

        if let Some(UnitFn::Offset {
            offset,
            call,
            args,
            frame,
        }) = unit.function(hash)
        {
            crate::runtime::vm::check_args(full_count, args)?;

            return Ok(CallResultOrOffset::Offset(CallOffset {
                offset,
                call,
                address,
                args,
                frame,
                output,
                guard: (value_guard, args_guard),
            }));
        }

        if !self.fn_call(hash, stack, address, full_count, output)? {
            // Restore the stack since no one has touched it.
            stack.pop_n(full_count)?;
            return Ok(CallResultOrOffset::Unsupported);
        }

        return Ok(CallResultOrOffset::Ok);
    }
}

impl fmt::Debug for RuntimeContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RuntimeContext")
    }
}

#[cfg(test)]
static_assertions::assert_impl_all!(RuntimeContext: Send, Sync);
