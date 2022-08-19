use std::iter;
use std::mem;
use std::slice;

use thiserror::Error;

use crate::runtime::{Address, Value};

/// An error raised when interacting with the stack.
#[derive(Debug, Error)]
#[error("tried to access out-of-bounds stack entry")]
#[non_exhaustive]
pub struct StackError;

const _: () = assert!(
    std::mem::size_of::<u32>() <= std::mem::size_of::<usize>(),
    "usize must be larger or equal to 32-bits"
);

/// Access of a mutable pair.
pub(crate) struct InterleavedPairMut<'a> {
    /// The first value in the pair.
    first: &'a mut Value,
    /// If the second value is present. If absent, the first value is used instead.
    second: Option<&'a mut Value>,
}

impl InterleavedPairMut<'_> {
    /// Get the first element in the pair.
    pub(crate) fn first_mut(&mut self) -> &mut Value {
        self.first
    }

    /// Get the second element in the pair.
    pub(crate) fn second_mut(&mut self) -> &mut Value {
        if let Some(ref mut second) = self.second {
            return *second;
        }

        self.first
    }
}

/// The stack of the virtual machine, where all values are stored.
#[derive(Default, Debug, Clone)]
pub struct Stack {
    /// The current stack of values.
    stack: Vec<Value>,
    /// The top of the current stack frame.
    ///
    /// It is not possible to interact with values below this stack frame.
    stack_bottom: usize,
}

impl Stack {
    /// Construct a new stack.
    ///
    /// ```
    /// use rune::runtime::Stack;
    /// use rune::Value;
    ///
    /// # fn main() -> Result<(), rune::runtime::StackError> {
    /// let mut stack = Stack::new();
    /// assert!(stack.pop().is_err());
    /// stack.push(String::from("Hello World"));
    /// assert!(matches!(stack.pop()?, Value::String(..)));
    /// # Ok(()) }
    /// ```
    pub const fn new() -> Self {
        Self {
            stack: Vec::new(),
            stack_bottom: 0,
        }
    }

    /// Construct a new stack with the given capacity pre-allocated.
    ///
    /// ```
    /// use rune::runtime::Stack;
    /// use rune::Value;
    ///
    /// # fn main() -> Result<(), rune::runtime::StackError> {
    /// let mut stack = Stack::with_capacity(16);
    /// assert!(stack.pop().is_err());
    /// stack.push(String::from("Hello World"));
    /// assert!(matches!(stack.pop()?, Value::String(..)));
    /// # Ok(()) }
    /// ```
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            stack: Vec::with_capacity(capacity),
            stack_bottom: 0,
        }
    }

    /// Check if the stack is empty.
    ///
    /// This ignores [stack_bottom] and will just check if the full stack is
    /// empty.
    ///
    /// ```
    /// use rune::runtime::Stack;
    ///
    /// let mut stack = Stack::new();
    /// assert!(stack.is_empty());
    /// stack.push(String::from("Hello World"));
    /// assert!(!stack.is_empty());
    /// ```
    ///
    /// [stack_bottom]: Self::stack_bottom()
    pub fn is_empty(&self) -> bool {
        self.stack.is_empty()
    }

    /// Get the length of the stack.
    ///
    /// This ignores [stack_bottom] and will just return the total length of
    /// the stack.
    ///
    /// ```
    /// use rune::runtime::Stack;
    ///
    /// let mut stack = Stack::new();
    /// assert_eq!(stack.len(), 0);
    /// stack.push(String::from("Hello World"));
    /// assert_eq!(stack.len(), 1);
    /// ```
    ///
    /// [stack_bottom]: Self::stack_bottom()
    pub fn len(&self) -> usize {
        self.stack.len()
    }

    /// Perform a raw access over the stack.
    ///
    /// This ignores [stack_bottom] and will just check that the given slice
    /// index is within range.
    ///
    /// [stack_bottom]: Self::stack_bottom()
    pub fn get<I>(&self, index: I) -> Option<&<I as slice::SliceIndex<[Value]>>::Output>
    where
        I: slice::SliceIndex<[Value]>,
    {
        self.stack.get(index)
    }

    /// Get the stack mutably from the top.
    pub(crate) fn get_mut_top(&mut self, len: usize) -> Result<&mut [Value], StackError> {
        let start = self.stack.len().checked_sub(len).ok_or(StackError)?;
        self.stack.get_mut(start..).ok_or(StackError)
    }

    /// Push a value onto the stack.
    ///
    /// ```
    /// use rune::runtime::Stack;
    /// use rune::Value;
    ///
    /// # fn main() -> Result<(), rune::runtime::StackError> {
    /// let mut stack = Stack::new();
    /// assert!(stack.pop().is_err());
    /// stack.push(String::from("Hello World"));
    /// assert!(matches!(stack.pop()?, Value::String(..)));
    /// # Ok(()) }
    /// ```
    pub fn push<T>(&mut self, value: T)
    where
        Value: From<T>,
    {
        self.stack.push(Value::from(value));
    }

    /// Push a new value onto the stack (expanding it as necessary) and return
    /// its address.
    pub fn push_with_address<T>(&mut self, value: T) -> Result<Address, StackError>
    where
        Value: From<T>,
    {
        let address = self
            .stack
            .len()
            .checked_sub(self.stack_bottom)
            .ok_or(StackError)?;
        let address = Address(address.try_into().map_err(|_| StackError)?);
        self.stack.push(Value::from(value));
        Ok(address)
    }

    /// Pop a value from the stack.
    ///
    /// ```
    /// use rune::runtime::Stack;
    /// use rune::Value;
    ///
    /// # fn main() -> Result<(), rune::runtime::StackError> {
    /// let mut stack = Stack::new();
    /// assert!(stack.pop().is_err());
    /// stack.push(String::from("Hello World"));
    /// assert!(matches!(stack.pop()?, Value::String(..)));
    /// # Ok(()) }
    /// ```
    pub fn pop(&mut self) -> Result<Value, StackError> {
        if self.stack.len() == self.stack_bottom {
            return Err(StackError);
        }

        self.stack.pop().ok_or(StackError)
    }

    /// Extend the current stack with an iterator.
    ///
    /// ```
    /// use rune::runtime::Stack;
    /// use rune::Value;
    ///
    /// # fn main() -> Result<(), rune::runtime::StackError> {
    /// let mut stack = Stack::new();
    ///
    /// stack.extend([Value::from(42i64), Value::from(String::from("foo")), Value::Unit]);
    ///
    /// let mut it = stack.drain(2)?;
    ///
    /// assert!(matches!(it.next(), Some(Value::String(..))));
    /// assert!(matches!(it.next(), Some(Value::Unit)));
    /// assert!(matches!(it.next(), None));
    /// # Ok(()) }
    /// ```
    pub fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = Value>,
    {
        self.stack.extend(iter);
    }

    /// Clear the current stack.
    pub fn clear(&mut self) {
        self.stack.clear();
        self.stack_bottom = 0;
    }

    /// Get the last position on the stack.
    #[inline]
    pub fn last(&self) -> Result<&Value, StackError> {
        self.stack.last().ok_or(StackError)
    }

    /// Iterate over the stack.
    pub fn iter(&self) -> impl Iterator<Item = &Value> + '_ {
        self.stack.iter()
    }

    /// Get the current top of the stack.
    pub(crate) fn top(&self) -> Result<Address, StackError> {
        u32::try_from(self.stack.len())
            .map(Address)
            .map_err(|_| StackError)
    }

    /// Get the offset that corresponds to the bottom of the stack right now.
    ///
    /// The stack is partitioned into call frames, and once we enter a call
    /// frame the bottom of the stack corresponds to the bottom of the current
    /// call frame.
    pub fn stack_bottom(&self) -> usize {
        self.stack_bottom
    }

    /// Store a value at the given stack address.
    ///
    /// ```
    /// use rune::runtime::Stack;
    /// use rune::Value;
    ///
    /// # fn main() -> Result<(), rune::runtime::StackError> {
    /// let mut stack = Stack::new();
    /// assert!(stack.pop().is_err());
    /// stack.push(String::from("Hello World"));
    /// assert!(matches!(stack.pop()?, Value::String(..)));
    /// # Ok(()) }
    /// ```
    #[inline]
    pub fn store<T>(&mut self, at: Address, value: T) -> Result<(), StackError>
    where
        Value: From<T>,
    {
        *self.at_mut(at)? = Value::from(value);
        Ok(())
    }

    /// Drain `count` elements from the top of the stack.
    ///
    /// # Examples
    ///
    /// ```
    /// use rune::runtime::Stack;
    /// use rune::Value;
    ///
    /// # fn main() -> Result<(), rune::runtime::StackError> {
    /// let mut stack = Stack::new();
    ///
    /// stack.push(42i64);
    /// stack.push(String::from("foo"));
    /// stack.push(());
    ///
    /// let mut it = stack.drain(2)?;
    ///
    /// assert!(matches!(it.next(), Some(Value::String(..))));
    /// assert!(matches!(it.next(), Some(Value::Unit)));
    /// assert!(matches!(it.next(), None));
    /// # Ok(()) }
    /// ```
    pub fn drain(
        &mut self,
        count: usize,
    ) -> Result<impl DoubleEndedIterator<Item = Value> + '_, StackError> {
        match self.stack.len().checked_sub(count) {
            Some(start) if start >= self.stack_bottom => Ok(self.stack.drain(start..)),
            _ => Err(StackError),
        }
    }

    /// Drain `count` elements from the top of the stack starting at the specified address.
    ///
    /// # Examples
    ///
    /// ```
    /// use rune::runtime::{Stack, StackAddress};
    /// use rune::Value;
    ///
    /// # fn main() -> Result<(), rune::runtime::StackError> {
    /// let mut stack = Stack::new();
    ///
    /// stack.push(42i64);
    /// stack.push(String::from("foo"));
    /// stack.push(());
    ///
    /// let mut it = stack.drain_at(StackAddress::BASE, 2)?;
    ///
    /// assert!(matches!(it.next(), Some(Value::Integer(42))));
    /// assert!(matches!(it.next(), Some(Value::String(..))));
    /// assert!(matches!(it.next(), None));
    /// # Ok(()) }
    /// ```
    pub fn drain_at(
        &mut self,
        Address(address): Address,
        count: usize,
    ) -> Result<impl DoubleEndedIterator<Item = Value> + '_, StackError> {
        return imp(self, address, count).ok_or(StackError);

        #[inline]
        fn imp(
            this: &mut Stack,
            address: u32,
            count: usize,
        ) -> Option<impl DoubleEndedIterator<Item = Value> + '_> {
            let address = usize::try_from(address).ok()?;
            let start = this.stack_bottom.checked_add(address)?;
            let end = start.checked_add(count)?;
            Some(this.stack.get_mut(start..end)?.into_iter().map(mem::take))
        }
    }

    /// Replace a slice of values on the stack.
    pub(crate) fn write_at<T>(
        &mut self,
        Address(address): Address,
        values: T,
    ) -> Result<(), StackError>
    where
        T: IntoIterator<Item = Value>,
    {
        return imp(self, address, values).ok_or(StackError);

        #[inline]
        fn imp<T>(this: &mut Stack, address: u32, values: T) -> Option<()>
        where
            T: IntoIterator<Item = Value>,
        {
            let address = usize::try_from(address).ok()?;
            let start = this.stack_bottom.checked_add(address)?;

            for (out, value) in this.stack.get_mut(start..)?.iter_mut().zip(values) {
                *out = value;
            }

            Some(())
        }
    }

    /// Pop `n` values from the stack returning an error in case the stack is
    /// too small to pop.
    pub(crate) fn pop_n(&mut self, n: usize) -> Result<(), StackError> {
        drop(self.drain(n)?);
        Ok(())
    }

    /// Access the value at the given frame offset.
    pub(crate) fn at(&self, Address(address): Address) -> Result<&Value, StackError> {
        self.stack_bottom
            .checked_add(address as usize)
            .and_then(|n| self.stack.get(n))
            .ok_or(StackError)
    }

    /// Take the value at the given address.
    pub fn take_at(&mut self, address: Address) -> Result<Value, StackError> {
        Ok(mem::take(self.at_mut(address)?))
    }

    /// Access the value at the given frame offset mutably.
    pub(crate) fn at_mut(&mut self, Address(address): Address) -> Result<&mut Value, StackError> {
        self.stack_bottom
            .checked_add(address as usize)
            .and_then(|n| self.stack.get_mut(n))
            .ok_or(StackError)
    }

    /// Access a pair of addresses mutably.
    ///
    /// The addresses must be distinct.
    pub(crate) fn pair(
        &mut self,
        Address(first): Address,
        Address(second): Address,
    ) -> Result<(&mut Value, &mut Value), StackError> {
        if first == second {
            return Err(StackError);
        }

        let first = self
            .stack_bottom
            .checked_add(first as usize)
            .ok_or(StackError)?;

        let second = self
            .stack_bottom
            .checked_add(second as usize)
            .ok_or(StackError)?;

        if first.max(second) >= self.stack.len() {
            return Err(StackError);
        }

        // SAFETY: Addresses are checked to be in bound above.
        unsafe {
            let ptr = self.stack.as_mut_ptr();
            Ok((&mut *ptr.add(first), &mut *ptr.add(second)))
        }
    }

    /// Address a value on the stack.
    pub(crate) fn interleaved_pair(
        &mut self,
        Address(first): Address,
        Address(second): Address,
    ) -> Result<InterleavedPairMut<'_>, StackError> {
        let first = first as usize;
        let second = second as usize;

        if first == second {
            let first = self.stack_bottom.checked_add(first).ok_or(StackError)?;

            return match self.stack.get_mut(first) {
                Some(first) => Ok(InterleavedPairMut {
                    first,
                    second: None,
                }),
                None => Err(StackError),
            };
        }

        let first = self.stack_bottom.checked_add(first).ok_or(StackError)?;
        let second = self.stack_bottom.checked_add(second).ok_or(StackError)?;

        if first.max(second) >= self.stack.len() {
            return Err(StackError);
        }

        // SAFETY: the addresses are bounds checked just above.
        unsafe {
            let ptr = self.stack.as_mut_ptr();

            Ok(InterleavedPairMut {
                first: &mut *ptr.add(first),
                second: Some(&mut *ptr.add(second)),
            })
        }
    }

    /// Modify the stack bottom to switch it out to the given bottom.
    pub(crate) fn replace_stack_frame(
        &mut self,
        bottom: Address,
        frame: usize,
    ) -> Result<(usize, usize), StackError> {
        let (stack_bottom, new_stack) =
            calculate(self.stack_bottom, bottom, frame).ok_or(StackError)?;
        let stack = self.stack.len();
        self.stack.resize(new_stack, Value::Unit);
        return Ok((mem::replace(&mut self.stack_bottom, stack_bottom), stack));

        fn calculate(
            stack_bottom: usize,
            Address(bottom): Address,
            frame: usize,
        ) -> Option<(usize, usize)> {
            let stack_bottom = stack_bottom.checked_add(usize::try_from(bottom).ok()?)?;
            let stack = stack_bottom.checked_add(frame)?;
            Some((stack_bottom, stack))
        }
    }

    /// Grow the stack so that it matches the given length.
    pub(crate) fn resize_frame(&mut self, frame: usize) -> Result<(), StackError> {
        let frame = self.stack_bottom.checked_add(frame).ok_or(StackError)?;
        self.stack.resize(frame, Value::Unit);
        Ok(())
    }

    /// Replace the stack bottom.
    pub(crate) fn pop_stack_frame(&mut self, stack_bottom: usize, stack: usize) -> usize {
        self.stack.resize(stack, Value::Unit);
        mem::replace(&mut self.stack_bottom, stack_bottom)
    }
}

impl iter::FromIterator<Value> for Stack {
    fn from_iter<T: IntoIterator<Item = Value>>(iter: T) -> Self {
        Self {
            stack: iter.into_iter().collect(),
            stack_bottom: 0,
        }
    }
}

impl From<Vec<Value>> for Stack {
    fn from(stack: Vec<Value>) -> Self {
        Self {
            stack,
            stack_bottom: 0,
        }
    }
}
