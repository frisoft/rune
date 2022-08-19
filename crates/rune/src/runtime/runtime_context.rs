use std::array;
use std::fmt;
use std::mem;
use std::sync::Arc;

use crate::collections::HashMap;
use crate::macros::{MacroContext, TokenStream};
use crate::runtime::vm::{CallOffset, CallResult, CallResultOrOffset};
use crate::runtime::{
    Address, ConstValue, GuardedArgs, Protocol, Stack, StackError, Unit, UnitFn, UnsafeToValue,
    Value, VmError,
};
use crate::{Hash, IntoTypeHash};

/// A collection of arguments passed into a function.
pub trait Arguments {
    /// Get the next address that was passed in.
    fn next(&mut self) -> Result<Address, StackError>;

    /// Take the next value out of a stack.
    fn take_next(&mut self, stack: &mut Stack) -> Result<Value, StackError> {
        Ok(mem::take(stack.at_mut(self.next()?)?))
    }

    /// Try to take the next value if one is present.
    fn try_take_next(&mut self, stack: &mut Stack) -> Result<Option<Value>, StackError> {
        let address = match self.next() {
            Ok(address) => address,
            Err(StackError) => return Ok(None),
        };

        Ok(Some(mem::take(stack.at_mut(address)?)))
    }

    /// The number of arguments passed in.
    fn count(&self) -> usize;
}

impl fmt::Debug for dyn Arguments {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Arguments")
            .field("count", &self.count())
            .finish()
    }
}

/// A type-reduced function handler.
pub(crate) type FunctionHandler =
    dyn Fn(&mut Stack, &mut dyn Arguments, Address) -> Result<(), VmError> + Send + Sync;

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
        arguments: &mut dyn Arguments,
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
        handler(stack, arguments, output)?;
        Ok(true)
    }

    /// Helper to call an index function.
    pub(crate) fn call_index_fn<const N: usize>(
        &self,
        stack: &mut Stack,
        protocol: Protocol,
        value: Address,
        index: usize,
        args: [Address; N],
        output: Address,
    ) -> Result<CallResultOrOffset<array::IntoIter<Address, 0>>, VmError> {
        let full_count = N + 1;
        let hash = stack.at(value)?.type_hash()?;
        let hash = Hash::index_fn(protocol, hash, Hash::index(index));

        let stack_bottom = stack.swap_frame(full_count)?;

        if !self.fn_call(
            hash,
            stack,
            &mut ArgumentsChain::new(value, args.into_iter()),
            output,
        )? {
            // Restore the stack since no one has touched it.
            stack.restore_frame(stack_bottom);
            return Ok(CallResultOrOffset::Unsupported);
        }

        stack.return_frame(stack_bottom, Address::BASE, output)?;
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

        let full_count = args.count() + 1;
        let old_bottom = stack.swap_frame(full_count)?;
        let hash = Hash::field_fn(protocol, value.type_hash()?, hash.into_type_hash());

        stack.store(Address::BASE, value.clone())?;

        // SAFETY: We hold onto the guard for the duration of this call.
        let _guard = unsafe { args.unsafe_into_stack(Address::FIRST, stack)? };

        if !self.fn_call(hash, stack, &mut Address::BASE.sequence(full_count), output)? {
            stack.restore_frame(old_bottom);
            return Ok(CallResult::Unsupported);
        }

        stack.return_frame(old_bottom, Address::BASE, output)?;
        Ok(CallResult::Ok(()))
    }

    /// Helper function to call an instance function.
    pub(crate) fn call_instance_fn<H, I>(
        &self,
        stack: &mut Stack,
        unit: &Unit,
        address: Address,
        hash: H,
        arguments: I,
        output: Address,
    ) -> Result<CallResultOrOffset<impl ExactSizeIterator<Item = Address>>, VmError>
    where
        H: IntoTypeHash,
        I: IntoIterator<Item = Address>,
        I::IntoIter: ExactSizeIterator,
    {
        let arguments = arguments.into_iter();
        let full_count = arguments.len() + 1;
        let type_hash = stack.at(address)?.type_hash()?;

        let hash = Hash::instance_function(type_hash, hash.into_type_hash());

        let mut arguments = ArgumentsChain::new(address, arguments);

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
                arguments,
                frame,
                output,
            }));
        }

        if !self.fn_call(hash, stack, &mut arguments, output)? {
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

struct ArgumentsChain<I> {
    first: Option<Address>,
    rest: I,
}

impl<I> ArgumentsChain<I> {
    fn new(first: Address, rest: I) -> Self {
        Self {
            first: Some(first),
            rest,
        }
    }
}

impl<I> Arguments for ArgumentsChain<I>
where
    I: ExactSizeIterator<Item = Address>,
{
    fn next(&mut self) -> Result<Address, StackError> {
        <Self as Iterator>::next(self).ok_or(StackError)
    }

    fn count(&self) -> usize {
        self.rest.len() + if self.first.is_some() { 1 } else { 0 }
    }
}

impl<I> Iterator for ArgumentsChain<I>
where
    I: Iterator<Item = Address>,
{
    type Item = Address;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(address) = self.first.take() {
            return Some(address);
        }

        self.rest.next()
    }
}

impl<I> ExactSizeIterator for ArgumentsChain<I>
where
    I: ExactSizeIterator<Item = Address>,
{
    fn len(&self) -> usize {
        self.count()
    }
}
