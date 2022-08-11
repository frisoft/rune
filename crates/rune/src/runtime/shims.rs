//! Various shims used to pass arguments into external functions.

use crate::runtime::{Address, Shared, Stack, Vec, VmError};
use crate::{ToValue, Value};

/// When returned from an external function will cause its sequence of values to
/// be encoded on the stack, starting at the given output address.
pub struct StackSequence<T>(T);

impl<T> StackSequence<T> {
    /// Construct a new stack sequence.
    pub const fn new(value: T) -> Self {
        Self(value)
    }
}

impl<T, const N: usize> ToValue for StackSequence<[T; N]>
where
    T: ToValue,
{
    fn to_value(self) -> Result<Value, VmError> {
        let mut vec = Vec::with_capacity(N);

        for value in self.0 {
            vec.push(value.to_value()?);
        }

        Ok(Value::Vec(Shared::new(vec)))
    }

    fn to_stack(self, stack: &mut Stack, output: Address) -> Result<(), VmError> {
        stack.store(output, self.to_value()?)?;
        Ok(())
    }
}

/// An optional value to be encoded onto the stack.
pub enum StackMaybe<T> {
    /// A present value.
    Present(T),
    /// An absent value.
    Absent,
}

impl<T> StackMaybe<T> {
    /// Construct a present value to be encoded onto the stack.
    pub const fn present(value: T) -> StackMaybe<T> {
        StackMaybe::Present(value)
    }

    /// Produce an absent value which is not encoded onto the stack.
    pub const fn absent() -> StackMaybe<T> {
        StackMaybe::Absent
    }
}

impl<T> ToValue for StackMaybe<T>
where
    T: ToValue,
{
    fn to_value(self) -> Result<Value, VmError> {
        Ok(Value::Option(Shared::new(match self {
            StackMaybe::Present(present) => Some(present.to_value()?),
            StackMaybe::Absent => None,
        })))
    }

    fn to_stack(self, stack: &mut Stack, output: Address) -> Result<(), VmError> {
        if let StackMaybe::Present(value) = self {
            value.to_stack(stack, output)?;
        }

        Ok(())
    }
}
