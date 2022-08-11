use crate::runtime::future::SelectFuture;
use crate::runtime::key::StringKey;
use crate::runtime::stack::InterleavedPairMut;
use crate::runtime::unit::UnitFn;
use crate::runtime::{budget, Key};
use crate::runtime::{
    Address, AnyObj, Args, Awaited, BorrowMut, BorrowRef, Bytes, Call, Format, FormatSpec,
    FromValue, Function, Future, Generator, GuardedArgs, Inst, InstAssignOp, InstOp,
    InstRangeLimits, InstTarget, InstValue, InstVariant, Object, Panic, Protocol, Range,
    RangeLimits, RuntimeContext, Select, Shared, Stack, StaticString, Stream, Struct, Tuple,
    TypeCheck, Unit, UnitStruct, UnsafeToValue, Value, Variant, VariantData, Vec, VmError,
    VmErrorKind, VmExecution, VmHalt, VmIntegerRepr, VmSendExecution,
};
use crate::{Hash, IntoTypeHash};
use std::fmt;
use std::mem;
use std::sync::Arc;
use std::vec;

/// The result of performing a call.
#[derive(Debug)]
#[must_use]
pub(crate) enum CallResult<T> {
    /// Call successful.
    Ok(T),
    /// Call was not supported.
    Unsupported,
}

/// Helper impls to get rid of the lifetime associated with BorrowRef if needed.
impl<T> CallResult<BorrowRef<'_, T>> {
    /// Return a cloned call result.
    fn cloned(self) -> CallResult<T>
    where
        T: Clone,
    {
        match self {
            CallResult::Ok(value) => CallResult::Ok(value.clone()),
            CallResult::Unsupported => CallResult::Unsupported,
        }
    }
}

#[derive(Debug)]
pub(crate) struct CallOffset<G> {
    /// Offset of the registered function.
    pub(crate) offset: usize,
    /// The way the function is called.
    pub(crate) call: Call,
    /// Stack address to allocate arguments from.
    pub(crate) address: Address,
    /// The number of arguments the function takes.
    pub(crate) args: usize,
    /// Slots to allocate for the function call.
    pub(crate) frame: usize,
    /// Output address to write to.
    pub(crate) output: Address,
    /// Guards used in the offset call.
    pub(crate) guard: G,
}

/// The result from a dynamic call. Indicates if the attempted operation is
/// supported.
#[derive(Debug)]
#[must_use]
pub(crate) enum CallResultOrOffset<G> {
    /// Call successful. Return value is on the stack.
    Ok,
    /// Call failed because function was missing so the method is unsupported.
    /// Contains target value.
    Unsupported,
    /// Offset to call.
    Offset(CallOffset<G>),
}

impl<G> CallResultOrOffset<G> {
    /// Apply the call result to the virtual machine if it resulted in an offset
    /// call.
    #[inline]
    fn or_call_offset_with(self, vm: &mut Vm) -> Result<CallResult<()>, VmError> {
        match self {
            CallResultOrOffset::Ok => Ok(CallResult::Ok(())),
            CallResultOrOffset::Unsupported => Ok(CallResult::Unsupported),
            CallResultOrOffset::Offset(CallOffset {
                offset,
                call,
                address,
                args,
                frame,
                output,
                guard: _guard,
            }) => {
                vm.call_offset_fn(offset, call, address, args, frame, output)?;
                Ok(CallResult::Ok(()))
            }
        }
    }
}

/// A stack which references variables indirectly from a slab.
#[derive(Debug, Clone)]
pub struct Vm {
    /// Context associated with virtual machine.
    context: Arc<RuntimeContext>,
    /// Unit associated with virtual machine.
    unit: Arc<Unit>,
    /// The current instruction pointer.
    ip: usize,
    /// The current stack.
    stack: Stack,
    /// Frames relative to the stack.
    call_frames: vec::Vec<CallFrame>,
}

impl Vm {
    /// Construct a new virtual machine.
    pub const fn new(context: Arc<RuntimeContext>, unit: Arc<Unit>) -> Self {
        Self::with_stack(context, unit, Stack::new())
    }

    /// Construct a new virtual machine with a custom stack.
    pub const fn with_stack(context: Arc<RuntimeContext>, unit: Arc<Unit>, stack: Stack) -> Self {
        Self {
            context,
            unit,
            ip: 0,
            stack,
            call_frames: vec::Vec::new(),
        }
    }

    /// Construct a vm with a default empty [RuntimeContext]. This is useful
    /// when the [Unit] was constructed with an empty
    /// [Context][crate::compile::Context].
    pub fn without_runtime(unit: Arc<Unit>) -> Self {
        Self::new(Default::default(), unit)
    }

    /// Test if the virtual machine is the same context and unit as specified.
    pub fn is_same(&self, context: &Arc<RuntimeContext>, unit: &Arc<Unit>) -> bool {
        Arc::ptr_eq(&self.context, context) && Arc::ptr_eq(&self.unit, unit)
    }

    /// Set  the current instruction pointer.
    #[inline]
    pub fn set_ip(&mut self, ip: usize) {
        self.ip = ip;
    }

    /// Get the stack.
    #[inline]
    pub fn call_frames(&self) -> &[CallFrame] {
        &self.call_frames
    }

    /// Get the stack.
    #[inline]
    pub fn stack(&self) -> &Stack {
        &self.stack
    }

    /// Get the stack mutably.
    #[inline]
    pub fn stack_mut(&mut self) -> &mut Stack {
        &mut self.stack
    }

    /// Access the context related to the virtual machine.
    #[inline]
    pub fn context(&self) -> &Arc<RuntimeContext> {
        &self.context
    }

    /// Access the underlying unit of the virtual machine.
    #[inline]
    pub fn unit(&self) -> &Arc<Unit> {
        &self.unit
    }

    /// Access the current instruction pointer.
    #[inline]
    pub fn ip(&self) -> usize {
        self.ip
    }

    /// Advance the instruction pointer.
    #[inline]
    pub(crate) fn advance(&mut self) {
        self.ip = self.ip.wrapping_add(1);
    }

    /// Reset this virtual machine, freeing all memory used.
    pub fn clear(&mut self) {
        self.ip = 0;
        self.stack.clear();
        self.call_frames.clear();
    }

    /// Modify the current instruction pointer.
    pub fn modify_ip(&mut self, offset: isize) {
        self.ip = if offset < 0 {
            self.ip.wrapping_sub(-offset as usize)
        } else {
            self.ip.wrapping_add(offset as usize)
        };
    }

    /// Run the given vm to completion.
    ///
    /// If any async instructions are encountered, this will error.
    pub fn complete(self) -> Result<Value, VmError> {
        let mut execution = VmExecution::new(self);
        execution.complete()
    }

    /// Run the given vm to completion with support for async functions.
    pub async fn async_complete(self) -> Result<Value, VmError> {
        let mut execution = VmExecution::new(self);
        execution.async_complete().await
    }

    /// Call the function identified by the given name.
    ///
    /// Computing the function hash from the name can be a bit costly, so it's
    /// worth noting that it can be precalculated:
    ///
    /// ```
    /// use rune::Hash;
    ///
    /// let name = Hash::type_hash(&["main"]);
    /// ```
    ///
    /// # Examples
    ///
    /// ```,no_run
    /// use rune::{Context, Unit, FromValue, Source};
    /// use std::sync::Arc;
    ///
    /// # fn main() -> rune::Result<()> {
    /// let context = Context::with_default_modules()?;
    /// let context = Arc::new(context.runtime());
    ///
    /// // Normally the unit would be created by compiling some source,
    /// // and since this one is empty it won't do anything.
    /// let unit = Arc::new(Unit::default());
    ///
    /// let mut vm = rune::Vm::new(context, unit);
    ///
    /// let output = vm.execute(&["main"], (33i64,))?.complete()?;
    /// let output = i64::from_value(output)?;
    ///
    /// println!("output: {}", output);
    /// # Ok(()) }
    /// ```
    ///
    /// You can use a `Vec<Value>` to provide a variadic collection of
    /// arguments.
    ///
    /// ```,no_run
    /// use rune::{Context, Unit, FromValue, Source, ToValue};
    /// use std::sync::Arc;
    ///
    /// # fn main() -> rune::Result<()> {
    /// let context = Context::with_default_modules()?;
    /// let context = Arc::new(context.runtime());
    ///
    /// // Normally the unit would be created by compiling some source,
    /// // and since this one is empty it won't do anything.
    /// let unit = Arc::new(Unit::default());
    ///
    /// let mut vm = rune::Vm::new(context, unit);
    ///
    /// let mut args = Vec::new();
    /// args.push(1u32.to_value()?);
    /// args.push(String::from("Hello World").to_value()?);
    ///
    /// let output = vm.execute(&["main"], args)?.complete()?;
    /// let output = i64::from_value(output)?;
    ///
    /// println!("output: {}", output);
    /// # Ok(()) }
    /// ```
    pub fn execute<A, N>(&mut self, name: N, args: A) -> Result<VmExecution<&mut Self>, VmError>
    where
        N: IntoTypeHash,
        A: Args,
    {
        self.set_entrypoint(name, args.count())?;
        args.into_stack(&mut self.stack)?;
        Ok(VmExecution::new(self))
    }

    /// An `execute` variant that returns an execution which implements
    /// [`Send`], allowing it to be sent and executed on a different thread.
    ///
    /// This is accomplished by preventing values escaping from being
    /// non-exclusively sent with the execution or escaping the execution. We
    /// only support encoding arguments which themselves are `Send`.
    pub fn send_execute<A, N>(mut self, name: N, args: A) -> Result<VmSendExecution, VmError>
    where
        N: IntoTypeHash,
        A: Send + Args,
    {
        // Safety: make sure the stack is clear, preventing any values from
        // being sent along with the virtual machine.
        self.stack.clear();

        self.set_entrypoint(name, args.count())?;
        args.into_stack(&mut self.stack)?;
        Ok(VmSendExecution(VmExecution::new(self)))
    }

    /// Call the given function immediately, returning the produced value.
    ///
    /// This function permits for using references since it doesn't defer its
    /// execution.
    ///
    /// # Panics
    ///
    /// If any of the arguments passed in are references, and that references is
    /// captured somewhere in the call as [`Mut<T>`] or [`Ref<T>`]
    /// this call will panic as we are trying to free the metadata relatedc to
    /// the reference.
    ///
    /// [`Mut<T>`]: crate::runtime::Mut
    /// [`Ref<T>`]: crate::runtime::Ref
    pub fn call<A, N>(&mut self, name: N, args: A) -> Result<Value, VmError>
    where
        N: IntoTypeHash,
        A: GuardedArgs,
    {
        self.set_entrypoint(name, args.count())?;

        // Safety: We hold onto the guard until the vm has completed and
        // `VmExecution` will clear the stack before this function returns.
        // Erronously or not.
        let guard = unsafe { args.unsafe_into_stack(&mut self.stack)? };

        let value = {
            // Clearing the stack here on panics has safety implications - see
            // above.
            let vm = ClearStack(self);
            VmExecution::new(&mut *vm.0).complete()?
        };

        // Note: this might panic if something in the vm is holding on to a
        // reference of the value. We should prevent it from being possible to
        // take any owned references to values held by this.
        drop(guard);
        Ok(value)
    }

    /// Call the given function immediately asynchronously, returning the
    /// produced value.
    ///
    /// This function permits for using references since it doesn't defer its
    /// execution.
    ///
    /// # Panics
    ///
    /// If any of the arguments passed in are references, and that references is
    /// captured somewhere in the call as [`Mut<T>`] or [`Ref<T>`]
    /// this call will panic as we are trying to free the metadata relatedc to
    /// the reference.
    ///
    /// [`Mut<T>`]: crate::runtime::Mut
    /// [`Ref<T>`]: crate::runtime::Ref
    pub async fn async_call<A, N>(&mut self, name: N, args: A) -> Result<Value, VmError>
    where
        N: IntoTypeHash,
        A: GuardedArgs,
    {
        self.set_entrypoint(name, args.count())?;

        // Safety: We hold onto the guard until the vm has completed and
        // `VmExecution` will clear the stack before this function returns.
        // Erronously or not.
        let guard = unsafe { args.unsafe_into_stack(&mut self.stack)? };

        let value = {
            // Clearing the stack here on panics has safety implications - see
            // above.
            let vm = ClearStack(self);
            VmExecution::new(&mut *vm.0).async_complete().await?
        };

        // Note: this might panic if something in the vm is holding on to a
        // reference of the value. We should prevent it from being possible to
        // take any owned references to values held by this.
        drop(guard);
        Ok(value)
    }

    /// Simplified external helper to call instance function associated with
    /// this virtual machine.
    pub(crate) fn call_instance_fn<V, H, A>(
        &mut self,
        value: V,
        hash: H,
        args: A,
    ) -> Result<Value, VmError>
    where
        V: UnsafeToValue,
        H: IntoTypeHash,
        A: GuardedArgs,
    {
        let hash = hash.into_type_hash();
        let output = self.stack.push_with_address(Value::Unit)?;

        if let CallResult::Unsupported = self
            .context
            .call_instance_fn(&mut self.stack, &self.unit, value, hash, args, output)?
            .or_call_offset_with(self)?
        {
            return Err(VmError::from(VmErrorKind::MissingFunction { hash }));
        }

        Ok(mem::take(self.stack.at_mut(output)?))
    }

    /// Update the instruction pointer to match the function matching the given
    /// name and check that the number of argument matches.
    fn set_entrypoint<N>(&mut self, name: N, count: usize) -> Result<(), VmError>
    where
        N: IntoTypeHash,
    {
        let hash = name.into_type_hash();

        let info = self.unit.function(hash).ok_or_else(|| {
            if let Some(item) = name.into_item() {
                VmError::from(VmErrorKind::MissingEntry { hash, item })
            } else {
                VmError::from(VmErrorKind::MissingEntryHash { hash })
            }
        })?;

        let (offset, frame) = match info {
            // NB: we ignore the calling convention.
            // everything is just async when called externally.
            UnitFn::Offset {
                offset,
                args: expected,
                frame,
                ..
            } => {
                check_args(count, expected)?;
                (offset, frame)
            }
            _ => {
                return Err(VmError::from(VmErrorKind::MissingFunction { hash }));
            }
        };

        self.ip = offset;
        self.stack.clear();
        self.stack.resize_frame(frame)?;
        self.call_frames.clear();
        Ok(())
    }

    fn internal_boolean_ops(
        &mut self,
        int_op: fn(i64, i64) -> bool,
        float_op: fn(f64, f64) -> bool,
        op: &'static str,
        lhs: Address,
        rhs: Address,
        output: Address,
    ) -> Result<(), VmError> {
        let rhs = self.stack.at(rhs)?;
        let lhs = self.stack.at(lhs)?;

        let out = match (lhs, rhs) {
            (Value::Integer(lhs), Value::Integer(rhs)) => int_op(*lhs, *rhs),
            (Value::Float(lhs), Value::Float(rhs)) => float_op(*lhs, *rhs),
            (lhs, rhs) => {
                return Err(VmError::from(VmErrorKind::UnsupportedBinaryOperation {
                    op,
                    lhs: lhs.type_info()?,
                    rhs: rhs.type_info()?,
                }))
            }
        };

        self.stack.store(output, out)?;
        Ok(())
    }

    /// Push a new call frame.
    ///
    /// This will cause the `args` number of elements on the stack to be
    /// associated and accessible to the new call frame.
    #[tracing::instrument(skip(self))]
    pub(crate) fn push_call_frame(
        &mut self,
        ip: usize,
        address: Address,
        frame: usize,
        output: Address,
    ) -> Result<(), VmError> {
        let (stack_bottom, stack) = self.stack.replace_stack_frame(address, frame)?;

        let frame = CallFrame {
            ip: self.ip,
            stack_bottom,
            stack,
            output,
        };

        tracing::trace!(?frame, "pushing call frame");

        self.call_frames.push(frame);
        self.ip = ip.wrapping_sub(1);
        Ok(())
    }

    /// Pop a call frame and return it.
    #[tracing::instrument(skip(self))]
    fn pop_call_frame(&mut self) -> Result<(bool, Address), VmError> {
        let frame = match self.call_frames.pop() {
            Some(frame) => frame,
            None => {
                self.stack.resize_frame(1)?;
                return Ok((true, Address::BASE));
            }
        };

        tracing::trace!(?frame, "popping call frame");
        self.stack.pop_stack_frame(frame.stack_bottom, frame.stack);
        self.ip = frame.ip;
        Ok((false, frame.output))
    }

    /// Internal implementation of the instance check.
    fn test_is_instance(&mut self, lhs: Address, rhs: Address) -> Result<bool, VmError> {
        let b = self.stack.at(rhs)?;
        let a = self.stack.at(lhs)?;

        let hash = match b {
            Value::Type(hash) => hash,
            _ => {
                return Err(VmError::from(VmErrorKind::UnsupportedIs {
                    value: a.type_info()?,
                    test_type: b.type_info()?,
                }));
            }
        };

        Ok(a.type_hash()? == *hash)
    }

    fn internal_boolean_op(
        &mut self,
        bool_op: impl FnOnce(bool, bool) -> bool,
        op: &'static str,
        lhs: Address,
        rhs: Address,
        output: Address,
    ) -> Result<(), VmError> {
        let rhs = self.stack.at(rhs)?.clone();
        let lhs = self.stack.at(lhs)?.clone();

        let out = match (lhs, rhs) {
            (Value::Bool(lhs), Value::Bool(rhs)) => bool_op(lhs, rhs),
            (lhs, rhs) => {
                return Err(VmError::from(VmErrorKind::UnsupportedBinaryOperation {
                    op,
                    lhs: lhs.type_info()?,
                    rhs: rhs.type_info()?,
                }));
            }
        };

        self.stack.store(output, out)?;
        Ok(())
    }

    /// Construct a future from calling an async function.
    fn call_generator_fn(
        &mut self,
        offset: usize,
        address: Address,
        args: usize,
        frame: usize,
        output: Address,
    ) -> Result<(), VmError> {
        let mut stack = self.stack.drain_at(address, args)?.collect::<Stack>();
        stack.resize_frame(frame)?;
        let mut vm = Self::with_stack(self.context.clone(), self.unit.clone(), stack);
        vm.ip = offset;
        self.stack.store(output, Generator::new(vm))?;
        Ok(())
    }

    /// Construct a stream from calling a function.
    fn call_stream_fn(
        &mut self,
        offset: usize,
        address: Address,
        args: usize,
        frame: usize,
        output: Address,
    ) -> Result<(), VmError> {
        let mut stack = self.stack.drain_at(address, args)?.collect::<Stack>();
        stack.resize_frame(frame)?;
        let mut vm = Self::with_stack(self.context.clone(), self.unit.clone(), stack);
        vm.ip = offset;
        self.stack.store(output, Stream::new(vm))?;
        Ok(())
    }

    /// Construct a future from calling a function.
    fn call_async_fn(
        &mut self,
        offset: usize,
        address: Address,
        args: usize,
        frame: usize,
        output: Address,
    ) -> Result<(), VmError> {
        let mut stack = self.stack.drain_at(address, args)?.collect::<Stack>();
        stack.resize_frame(frame)?;
        let mut vm = Self::with_stack(self.context.clone(), self.unit.clone(), stack);
        vm.ip = offset;
        self.stack.store(output, Future::new(vm.async_complete()))?;
        Ok(())
    }

    /// Helper function to call the function at the given offset.
    pub(crate) fn call_offset_fn(
        &mut self,
        offset: usize,
        call: Call,
        address: Address,
        args: usize,
        frame: usize,
        output: Address,
    ) -> Result<(), VmError> {
        match call {
            Call::Async => {
                self.call_async_fn(offset, address, args, frame, output)?;
            }
            Call::Immediate => {
                self.push_call_frame(offset, address, frame, output)?;
            }
            Call::Stream => {
                self.call_stream_fn(offset, address, args, frame, output)?;
            }
            Call::Generator => {
                self.call_generator_fn(offset, address, args, frame, output)?;
            }
        }

        Ok(())
    }

    fn internal_num_assign(
        &mut self,
        lhs: Address,
        rhs: Address,
        target: InstTarget,
        protocol: Protocol,
        error: fn() -> VmErrorKind,
        integer_op: fn(i64, i64) -> Option<i64>,
        float_op: fn(f64, f64) -> f64,
        output: Address,
    ) -> Result<(), VmError> {
        target_value(
            self,
            lhs,
            rhs,
            target,
            protocol,
            output,
            |lhs, rhs| match (lhs, rhs) {
                (Value::Integer(lhs), Value::Integer(rhs)) => {
                    let out = integer_op(*lhs, *rhs).ok_or_else(error)?;
                    *lhs = out;
                    Ok(CallResult::Ok(()))
                }
                (Value::Float(lhs), Value::Float(rhs)) => {
                    let out = float_op(*lhs, *rhs);
                    *lhs = out;
                    Ok(CallResult::Ok(()))
                }
                _ => Ok(CallResult::Unsupported),
            },
        )
    }

    /// Internal impl of a numeric operation.
    fn internal_num(
        &mut self,
        protocol: Protocol,
        error: fn() -> VmErrorKind,
        integer_op: fn(i64, i64) -> Option<i64>,
        float_op: fn(f64, f64) -> f64,
        lhs_address: Address,
        rhs_address: Address,
        output: Address,
    ) -> Result<(), VmError> {
        let rhs = self.stack.at(rhs_address)?;
        let lhs = self.stack.at(lhs_address)?;

        match (lhs, rhs) {
            (Value::Integer(lhs), Value::Integer(rhs)) => {
                self.stack
                    .store(output, integer_op(*lhs, *rhs).ok_or_else(error)?)?;
                return Ok(());
            }
            (Value::Float(lhs), Value::Float(rhs)) => {
                self.stack.store(output, float_op(*lhs, *rhs))?;
                return Ok(());
            }
            _ => {}
        }

        let lhs = lhs.clone();
        let rhs = rhs.clone();

        if let CallResult::Ok(()) = self
            .context
            .call_instance_fn(&mut self.stack, &self.unit, lhs, protocol, (rhs,), output)?
            .or_call_offset_with(self)?
        {
            return Ok(());
        }

        Err(VmError::from(VmErrorKind::UnsupportedBinaryOperation {
            op: protocol.name,
            lhs: self.stack.at(lhs_address)?.type_info()?,
            rhs: self.stack.at(rhs_address)?.type_info()?,
        }))
    }

    /// Internal impl of a numeric operation.
    fn internal_infallible_bitwise(
        &mut self,
        protocol: Protocol,
        integer_op: fn(i64, i64) -> i64,
        lhs_address: Address,
        rhs_address: Address,
        output: Address,
    ) -> Result<(), VmError> {
        let lhs = self.stack.at(lhs_address)?;
        let rhs = self.stack.at(rhs_address)?;

        let (lhs, rhs) = match (lhs, rhs) {
            (Value::Integer(lhs), Value::Integer(rhs)) => {
                self.stack.store(output, integer_op(*lhs, *rhs))?;
                return Ok(());
            }
            (lhs, rhs) => (lhs, rhs),
        };

        let lhs = lhs.clone();
        let rhs = rhs.clone();

        if let CallResult::Ok(()) = self
            .context
            .call_instance_fn(
                &mut self.stack,
                &self.unit,
                lhs.clone(),
                protocol,
                (rhs.clone(),),
                output,
            )?
            .or_call_offset_with(self)?
        {
            return Ok(());
        }

        Err(VmError::from(VmErrorKind::UnsupportedBinaryOperation {
            op: protocol.name,
            lhs: self.stack.at(lhs_address)?.type_info()?,
            rhs: self.stack.at(rhs_address)?.type_info()?,
        }))
    }

    /// Internal impl of a numeric operation.
    fn internal_infallible_bitwise_bool(
        &mut self,
        protocol: Protocol,
        integer_op: fn(i64, i64) -> i64,
        bool_op: fn(bool, bool) -> bool,
        lhs_address: Address,
        rhs_address: Address,
        output: Address,
    ) -> Result<(), VmError> {
        let lhs = self.stack.at(lhs_address)?;
        let rhs = self.stack.at(rhs_address)?;

        let (lhs, rhs) = match (lhs, rhs) {
            (Value::Integer(lhs), Value::Integer(rhs)) => {
                self.stack.store(output, integer_op(*lhs, *rhs))?;
                return Ok(());
            }
            (Value::Bool(lhs), Value::Bool(rhs)) => {
                self.stack.store(output, bool_op(*lhs, *rhs))?;
                return Ok(());
            }
            (lhs, rhs) => (lhs, rhs),
        };

        let lhs = lhs.clone();
        let rhs = rhs.clone();

        if let CallResult::Ok(()) = self
            .context
            .call_instance_fn(
                &mut self.stack,
                &self.unit,
                lhs.clone(),
                protocol,
                (rhs.clone(),),
                output,
            )?
            .or_call_offset_with(self)?
        {
            return Ok(());
        }

        Err(VmError::from(VmErrorKind::UnsupportedBinaryOperation {
            op: protocol.name,
            lhs: self.stack.at(lhs_address)?.type_info()?,
            rhs: self.stack.at(rhs_address)?.type_info()?,
        }))
    }

    fn internal_infallible_bitwise_assign(
        &mut self,
        lhs: Address,
        rhs: Address,
        target: InstTarget,
        protocol: Protocol,
        integer_op: fn(&mut i64, i64),
        output: Address,
    ) -> Result<(), VmError> {
        target_value(
            self,
            lhs,
            rhs,
            target,
            protocol,
            output,
            |lhs, rhs| match (lhs, rhs) {
                (Value::Integer(lhs), Value::Integer(rhs)) => {
                    integer_op(lhs, *rhs);
                    Ok(CallResult::Ok(()))
                }
                _ => Ok(CallResult::Unsupported),
            },
        )
    }

    fn internal_bitwise(
        &mut self,
        protocol: Protocol,
        error: fn() -> VmErrorKind,
        integer_op: fn(i64, i64) -> Option<i64>,
        lhs_address: Address,
        rhs_address: Address,
        output: Address,
    ) -> Result<(), VmError> {
        let lhs = self.stack.at(lhs_address)?;
        let rhs = self.stack.at(rhs_address)?;

        match (lhs, rhs) {
            (Value::Integer(lhs), Value::Integer(rhs)) => {
                self.stack
                    .store(output, integer_op(*lhs, *rhs).ok_or_else(error)?)?;
                return Ok(());
            }
            _ => {}
        }

        let lhs = lhs.clone();
        let rhs = lhs.clone();

        if let CallResult::Ok(()) = self
            .context
            .call_instance_fn(
                &mut self.stack,
                &self.unit,
                lhs.clone(),
                protocol,
                (rhs.clone(),),
                output,
            )?
            .or_call_offset_with(self)?
        {
            return Ok(());
        }

        Err(VmError::from(VmErrorKind::UnsupportedBinaryOperation {
            op: protocol.name,
            lhs: self.stack.at(lhs_address)?.type_info()?,
            rhs: self.stack.at(rhs_address)?.type_info()?,
        }))
    }

    fn internal_bitwise_assign(
        &mut self,
        lhs: Address,
        rhs: Address,
        target: InstTarget,
        protocol: Protocol,
        error: fn() -> VmErrorKind,
        integer_op: fn(i64, i64) -> Option<i64>,
        output: Address,
    ) -> Result<(), VmError> {
        target_value(
            self,
            lhs,
            rhs,
            target,
            protocol,
            output,
            |lhs, rhs| match (lhs, rhs) {
                (Value::Integer(lhs), Value::Integer(rhs)) => {
                    let out = integer_op(*lhs, *rhs).ok_or_else(error)?;
                    *lhs = out;
                    Ok(CallResult::Ok(()))
                }
                _ => Ok(CallResult::Unsupported),
            },
        )
    }

    #[cfg_attr(feature = "bench", inline(never))]
    #[tracing::instrument(skip(self))]
    fn op_await(&mut self, address: Address) -> Result<Shared<Future>, VmError> {
        let value = mem::take(self.stack.at_mut(address)?);
        value.into_shared_future()
    }

    #[cfg_attr(feature = "bench", inline(never))]
    #[tracing::instrument(skip(self))]
    fn op_select(
        &mut self,
        address: Address,
        len: usize,
        output: Address,
    ) -> Result<Option<Select>, VmError> {
        let futures = futures_util::stream::FuturesUnordered::new();

        for (branch, value) in self.stack.drain_at(address, len)?.enumerate() {
            let future = value.into_shared_future()?.into_mut()?;

            if !future.is_completed() {
                futures.push(SelectFuture::new(branch, future));
            }
        }

        // NB: nothing to poll.
        if futures.is_empty() {
            self.stack.store(output, ())?;
            return Ok(None);
        }

        Ok(Some(Select::new(futures)))
    }

    /// Store a specific literal value at the given `output`.
    #[cfg_attr(feature = "bench", inline(never))]
    #[tracing::instrument(skip(self))]
    fn op_store(&mut self, value: InstValue, output: Address) -> Result<(), VmError> {
        *self.stack.at_mut(output)? = value.into_value();
        Ok(())
    }

    /// Copy a value from a position relative to the top of the stack, to the
    /// top of the stack.
    #[cfg_attr(feature = "bench", inline(never))]
    #[tracing::instrument(skip(self))]
    fn op_copy(&mut self, address: Address, output: Address) -> Result<(), VmError> {
        let mut pair = self.stack.interleaved_pair(address, output)?;
        *pair.second_mut() = pair.first_mut().clone();
        Ok(())
    }

    /// Move a value from a position relative to the top of the stack, to the
    /// top of the stack.
    #[cfg_attr(feature = "bench", inline(never))]
    #[tracing::instrument(skip(self))]
    fn op_move(&mut self, address: Address, output: Address) -> Result<(), VmError> {
        let mut pair = self.stack.interleaved_pair(address, output)?;
        *pair.second_mut() = mem::take(pair.first_mut());
        Ok(())
    }

    #[cfg_attr(feature = "bench", inline(never))]
    #[tracing::instrument(skip(self))]
    fn op_drop(&mut self, address: Address) -> Result<(), VmError> {
        *self.stack.at_mut(address)? = Value::Unit;
        Ok(())
    }

    /// Perform a jump operation.
    #[cfg_attr(feature = "bench", inline(never))]
    #[tracing::instrument(skip(self))]
    fn op_jump(&mut self, offset: isize) -> Result<(), VmError> {
        self.modify_ip(offset);
        Ok(())
    }

    /// Jump the given offset if `address` matches the expected value.
    #[cfg_attr(feature = "bench", inline(never))]
    #[tracing::instrument(skip(self))]
    fn op_jump_if(
        &mut self,
        address: Address,
        offset: isize,
        condition: bool,
    ) -> Result<(), VmError> {
        if self.stack.at(address)?.as_bool()? == condition {
            self.modify_ip(offset);
        }

        Ok(())
    }

    /// Perform a branch-conditional jump operation.
    #[cfg_attr(feature = "bench", inline(never))]
    #[tracing::instrument(skip(self))]
    fn op_jump_if_branch(
        &mut self,
        address: Address,
        branch: i64,
        offset: isize,
    ) -> Result<(), VmError> {
        let value = self.stack.at_mut(address)?;

        if let Value::Integer(current) = value {
            if *current == branch {
                *value = Value::Unit;
                self.modify_ip(offset);
            }
        }

        Ok(())
    }

    /// Construct a new vec.
    #[cfg_attr(feature = "bench", inline(never))]
    #[tracing::instrument(skip(self))]
    fn op_vec(&mut self, address: Address, count: usize, output: Address) -> Result<(), VmError> {
        let data = self
            .stack
            .drain_at(address, count)?
            .collect::<vec::Vec<_>>();
        self.stack.store(output, Vec::from(data))?;
        Ok(())
    }

    /// Construct a new tuple.
    #[cfg_attr(feature = "bench", inline(never))]
    #[tracing::instrument(skip(self))]
    fn op_tuple(&mut self, address: Address, count: usize, output: Address) -> Result<(), VmError> {
        let tuple = self.stack.drain_at(address, count)?.collect::<Box<[_]>>();
        self.stack.store(output, Tuple::from(tuple))?;
        Ok(())
    }

    /// Construct a new tuple with a fixed number of arguments.
    #[cfg_attr(feature = "bench", inline(never))]
    #[tracing::instrument(skip(self))]
    fn op_tuple_n(&mut self, args: &[Address], output: Address) -> Result<(), VmError> {
        let mut tuple = vec::Vec::with_capacity(args.len());

        for arg in args {
            tuple.push(self.stack.at(*arg)?.clone());
        }

        self.stack.store(output, Tuple::from(tuple))?;
        Ok(())
    }

    /// Push the tuple that is on top of the stack.
    #[cfg_attr(feature = "bench", inline(never))]
    #[tracing::instrument(skip(self))]
    fn op_unpack_tuple(&mut self, address: Address, mut base: Address) -> Result<(), VmError> {
        let tuple = mem::take(self.stack.at_mut(address)?).into_tuple()?;

        for value in tuple.take()?.into_iter() {
            self.stack.store(base, value)?;
            base = base.step()?;
        }

        Ok(())
    }

    #[cfg_attr(feature = "bench", inline(never))]
    #[tracing::instrument(skip(self))]
    fn op_not(&mut self, address: Address, output: Address) -> Result<(), VmError> {
        let mut pair = self.stack.interleaved_pair(address, output)?;

        let value = match pair.first_mut() {
            Value::Bool(value) => Value::from(!*value),
            Value::Integer(value) => Value::from(!*value),
            other => {
                return Err(VmError::from(VmErrorKind::UnsupportedUnaryOperation {
                    op: "!",
                    operand: other.type_info()?,
                }));
            }
        };

        *pair.second_mut() = value;
        Ok(())
    }

    #[cfg_attr(feature = "bench", inline(never))]
    #[tracing::instrument(skip(self))]
    fn op_neg(&mut self, address: Address, output: Address) -> Result<(), VmError> {
        let mut pair = self.stack.interleaved_pair(address, output)?;

        let value = match pair.first_mut() {
            Value::Float(value) => Value::from(-*value),
            Value::Integer(value) => Value::from(-*value),
            other => {
                return Err(VmError::from(VmErrorKind::UnsupportedUnaryOperation {
                    op: "-",
                    operand: other.type_info()?,
                }));
            }
        };

        *pair.second_mut() = value;
        Ok(())
    }

    #[cfg_attr(feature = "bench", inline(never))]
    #[tracing::instrument(skip(self))]
    fn op_op(
        &mut self,
        op: InstOp,
        lhs: Address,
        rhs: Address,
        output: Address,
    ) -> Result<(), VmError> {
        use std::convert::TryFrom as _;

        match op {
            InstOp::Add => {
                self.internal_num(
                    Protocol::ADD,
                    || VmErrorKind::Overflow,
                    i64::checked_add,
                    std::ops::Add::add,
                    lhs,
                    rhs,
                    output,
                )?;
            }
            InstOp::Sub => {
                self.internal_num(
                    Protocol::SUB,
                    || VmErrorKind::Underflow,
                    i64::checked_sub,
                    std::ops::Sub::sub,
                    lhs,
                    rhs,
                    output,
                )?;
            }
            InstOp::Mul => {
                self.internal_num(
                    Protocol::MUL,
                    || VmErrorKind::Overflow,
                    i64::checked_mul,
                    std::ops::Mul::mul,
                    lhs,
                    rhs,
                    output,
                )?;
            }
            InstOp::Div => {
                self.internal_num(
                    Protocol::DIV,
                    || VmErrorKind::DivideByZero,
                    i64::checked_div,
                    std::ops::Div::div,
                    lhs,
                    rhs,
                    output,
                )?;
            }
            InstOp::Rem => {
                self.internal_num(
                    Protocol::REM,
                    || VmErrorKind::DivideByZero,
                    i64::checked_rem,
                    std::ops::Rem::rem,
                    lhs,
                    rhs,
                    output,
                )?;
            }
            InstOp::BitAnd => {
                use std::ops::BitAnd as _;
                self.internal_infallible_bitwise_bool(
                    Protocol::BIT_AND,
                    i64::bitand,
                    bool::bitand,
                    lhs,
                    rhs,
                    output,
                )?;
            }
            InstOp::BitXor => {
                use std::ops::BitXor as _;
                self.internal_infallible_bitwise_bool(
                    Protocol::BIT_XOR,
                    i64::bitxor,
                    bool::bitxor,
                    lhs,
                    rhs,
                    output,
                )?;
            }
            InstOp::BitOr => {
                use std::ops::BitOr as _;
                self.internal_infallible_bitwise_bool(
                    Protocol::BIT_OR,
                    i64::bitor,
                    bool::bitor,
                    lhs,
                    rhs,
                    output,
                )?;
            }
            InstOp::Shl => {
                self.internal_bitwise(
                    Protocol::SHL,
                    || VmErrorKind::Overflow,
                    |a, b| a.checked_shl(u32::try_from(b).ok()?),
                    lhs,
                    rhs,
                    output,
                )?;
            }
            InstOp::Shr => {
                self.internal_infallible_bitwise(
                    Protocol::SHR,
                    std::ops::Shr::shr,
                    lhs,
                    rhs,
                    output,
                )?;
            }
            InstOp::Gt => {
                self.internal_boolean_ops(|a, b| a > b, |a, b| a > b, ">", lhs, rhs, output)?;
            }
            InstOp::Gte => {
                self.internal_boolean_ops(|a, b| a >= b, |a, b| a >= b, ">=", lhs, rhs, output)?;
            }
            InstOp::Lt => {
                self.internal_boolean_ops(|a, b| a < b, |a, b| a < b, "<", lhs, rhs, output)?;
            }
            InstOp::Lte => {
                self.internal_boolean_ops(|a, b| a <= b, |a, b| a <= b, "<=", lhs, rhs, output)?;
            }
            InstOp::Eq => {
                let rhs = self.stack.at(rhs)?.clone();
                let lhs = self.stack.at(lhs)?.clone();
                let test = Value::value_ptr_eq(self, &lhs, &rhs)?;
                self.stack.store(output, test)?;
            }
            InstOp::Neq => {
                let rhs = self.stack.at(rhs)?.clone();
                let lhs = self.stack.at(lhs)?.clone();
                let test = Value::value_ptr_eq(self, &lhs, &rhs)?;
                self.stack.store(output, !test)?;
            }
            InstOp::And => {
                self.internal_boolean_op(|a, b| a && b, "&&", lhs, rhs, output)?;
            }
            InstOp::Or => {
                self.internal_boolean_op(|a, b| a || b, "||", lhs, rhs, output)?;
            }
            InstOp::Is => {
                let is_instance = self.test_is_instance(lhs, rhs)?;
                self.stack.store(output, is_instance)?;
            }
            InstOp::IsNot => {
                let is_instance = self.test_is_instance(lhs, rhs)?;
                self.stack.store(output, !is_instance)?;
            }
        }

        Ok(())
    }

    #[cfg_attr(feature = "bench", inline(never))]
    #[tracing::instrument(skip(self))]
    fn op_assign(
        &mut self,
        lhs: Address,
        rhs: Address,
        target: InstTarget,
        op: InstAssignOp,
        output: Address,
    ) -> Result<(), VmError> {
        use std::convert::TryFrom as _;

        match op {
            InstAssignOp::Add => {
                self.internal_num_assign(
                    lhs,
                    rhs,
                    target,
                    Protocol::ADD_ASSIGN,
                    || VmErrorKind::Overflow,
                    i64::checked_add,
                    std::ops::Add::add,
                    output,
                )?;
            }
            InstAssignOp::Sub => {
                self.internal_num_assign(
                    lhs,
                    rhs,
                    target,
                    Protocol::SUB_ASSIGN,
                    || VmErrorKind::Underflow,
                    i64::checked_sub,
                    std::ops::Sub::sub,
                    output,
                )?;
            }
            InstAssignOp::Mul => {
                self.internal_num_assign(
                    lhs,
                    rhs,
                    target,
                    Protocol::MUL_ASSIGN,
                    || VmErrorKind::Overflow,
                    i64::checked_mul,
                    std::ops::Mul::mul,
                    output,
                )?;
            }
            InstAssignOp::Div => {
                self.internal_num_assign(
                    lhs,
                    rhs,
                    target,
                    Protocol::DIV_ASSIGN,
                    || VmErrorKind::DivideByZero,
                    i64::checked_div,
                    std::ops::Div::div,
                    output,
                )?;
            }
            InstAssignOp::Rem => {
                self.internal_num_assign(
                    lhs,
                    rhs,
                    target,
                    Protocol::REM_ASSIGN,
                    || VmErrorKind::DivideByZero,
                    i64::checked_rem,
                    std::ops::Rem::rem,
                    output,
                )?;
            }
            InstAssignOp::BitAnd => {
                self.internal_infallible_bitwise_assign(
                    lhs,
                    rhs,
                    target,
                    Protocol::BIT_AND_ASSIGN,
                    std::ops::BitAndAssign::bitand_assign,
                    output,
                )?;
            }
            InstAssignOp::BitXor => {
                self.internal_infallible_bitwise_assign(
                    lhs,
                    rhs,
                    target,
                    Protocol::BIT_XOR_ASSIGN,
                    std::ops::BitXorAssign::bitxor_assign,
                    output,
                )?;
            }
            InstAssignOp::BitOr => {
                self.internal_infallible_bitwise_assign(
                    lhs,
                    rhs,
                    target,
                    Protocol::BIT_OR_ASSIGN,
                    std::ops::BitOrAssign::bitor_assign,
                    output,
                )?;
            }
            InstAssignOp::Shl => {
                self.internal_bitwise_assign(
                    lhs,
                    rhs,
                    target,
                    Protocol::SHL_ASSIGN,
                    || VmErrorKind::Overflow,
                    |a, b| a.checked_shl(u32::try_from(b).ok()?),
                    output,
                )?;
            }
            InstAssignOp::Shr => {
                self.internal_infallible_bitwise_assign(
                    lhs,
                    rhs,
                    target,
                    Protocol::SHR_ASSIGN,
                    std::ops::ShrAssign::shr_assign,
                    output,
                )?;
            }
        }

        Ok(())
    }

    /// Perform an index set operation.
    #[cfg_attr(feature = "bench", inline(never))]
    #[tracing::instrument(skip(self))]
    fn op_index_set(
        &mut self,
        address: Address,
        index: Address,
        value: Address,
        output: Address,
    ) -> Result<(), VmError> {
        {
            let target = self.stack.at(address)?;
            let index = self.stack.at(index)?;
            let value = self.stack.at(value)?;

            if let CallResult::Ok(()) = builtin(target, index, value)? {
                return Ok(());
            }

            let target = target.clone();
            let index = index.clone();
            let value = value.clone();

            if let CallResult::Ok(()) = self
                .context
                .call_instance_fn(
                    &mut self.stack,
                    &self.unit,
                    target,
                    Protocol::INDEX_SET,
                    (index, value),
                    output,
                )?
                .or_call_offset_with(self)?
            {
                return Ok(());
            }
        }

        return Err(VmError::from(VmErrorKind::UnsupportedIndexSet {
            target: self.stack.at(address)?.type_info()?,
            index: self.stack.at(index)?.type_info()?,
            value: self.stack.at(value)?.type_info()?,
        }));

        /// Built-in implementation of index set.
        fn builtin(
            target: &Value,
            index: &Value,
            value: &Value,
        ) -> Result<CallResult<()>, VmError> {
            // NB: local storage for string.
            let local_field;

            let field = match &index {
                Value::String(string) => {
                    local_field = string.borrow_ref()?;
                    local_field.as_str()
                }
                Value::StaticString(string) => string.as_ref(),
                _ => return Ok(CallResult::Unsupported),
            };

            match &target {
                Value::Object(object) => {
                    let mut object = object.borrow_mut()?;
                    object.insert(field.to_owned(), value.clone());
                    Ok(CallResult::Ok(()))
                }
                Value::Struct(typed_object) => {
                    let mut typed_object = typed_object.borrow_mut()?;

                    if let Some(v) = typed_object.get_mut(field) {
                        *v = value.clone();
                        return Ok(CallResult::Ok(()));
                    }

                    Err(VmError::from(VmErrorKind::MissingField {
                        field: field.to_owned(),
                        target: typed_object.type_info(),
                    }))
                }
                Value::Variant(variant) => {
                    let mut variant = variant.borrow_mut()?;

                    if let VariantData::Struct(st) = variant.data_mut() {
                        if let Some(v) = st.get_mut(field) {
                            *v = value.clone();
                            return Ok(CallResult::Ok(()));
                        }

                        return Err(VmError::from(VmErrorKind::MissingField {
                            field: field.to_owned(),
                            target: variant.type_info(),
                        }));
                    }

                    Ok(CallResult::Unsupported)
                }
                _ => Ok(CallResult::Unsupported),
            }
        }
    }

    #[inline]
    #[tracing::instrument(skip(self))]
    fn op_return_internal(&mut self, return_value: Value) -> Result<bool, VmError> {
        let (exit, output) = self.pop_call_frame()?;
        self.stack.store(output, return_value)?;
        Ok(exit)
    }

    #[cfg_attr(feature = "bench", inline(never))]
    #[tracing::instrument(skip(self))]
    fn op_return(&mut self, address: Address) -> Result<bool, VmError> {
        let return_value = mem::take(self.stack.at_mut(address)?);
        self.op_return_internal(return_value)
    }

    #[cfg_attr(feature = "bench", inline(never))]
    #[tracing::instrument(skip(self))]
    fn op_return_unit(&mut self) -> Result<bool, VmError> {
        let (exit, output) = self.pop_call_frame()?;
        self.stack.store(output, ())?;
        Ok(exit)
    }

    #[cfg_attr(feature = "bench", inline(never))]
    #[tracing::instrument(skip(self))]
    fn op_load_instance_fn(
        &mut self,
        address: Address,
        hash: Hash,
        output: Address,
    ) -> Result<(), VmError> {
        let mut pair = self.stack.interleaved_pair(address, output)?;
        let type_hash = pair.first_mut().type_hash()?;
        let hash = Hash::instance_function(type_hash, hash);
        *pair.second_mut() = Value::Type(hash);
        Ok(())
    }

    /// Perform an index get operation.
    #[cfg_attr(feature = "bench", inline(never))]
    #[tracing::instrument(skip(self))]
    fn op_index_get(
        &mut self,
        address: Address,
        index_address: Address,
        output: Address,
    ) -> Result<(), VmError> {
        let target = self.stack.at(address)?;
        let index = self.stack.at(index_address)?;

        if let CallResult::Ok(value) = builtin(target, index)? {
            self.stack.store(output, value)?;
            return Ok(());
        }

        let target = self.stack.at(address)?.clone();
        let index = self.stack.at(index_address)?.clone();

        if let CallResult::Ok(()) = self
            .context
            .call_instance_fn(
                &mut self.stack,
                &self.unit,
                target,
                Protocol::INDEX_GET,
                (index,),
                output,
            )?
            .or_call_offset_with(self)?
        {
            return Ok(());
        }

        return Err(VmError::from(VmErrorKind::UnsupportedIndexGet {
            target: self.stack.at(address)?.type_info()?,
            index: self.stack.at(index_address)?.type_info()?,
        }));

        fn builtin<'a>(target: &'a Value, index: &Value) -> Result<CallResult<Value>, VmError> {
            match index {
                Value::String(string) => {
                    let string_ref = string.borrow_ref()?;
                    Ok(try_object_like_index_get(target, string_ref.as_str())?.cloned())
                }
                Value::StaticString(string) => {
                    Ok(try_object_like_index_get(target, string.as_ref())?.cloned())
                }
                Value::Integer(index) => {
                    use std::convert::TryInto as _;

                    let index = match (*index).try_into() {
                        Ok(index) => index,
                        Err(..) => {
                            return Err(VmError::from(VmErrorKind::MissingIndex {
                                target: target.type_info()?,
                                index: VmIntegerRepr::from(*index),
                            }));
                        }
                    };

                    Ok(try_tuple_like_index_get(target, index)?.cloned())
                }
                _ => Ok(CallResult::Unsupported),
            }
        }
    }

    /// Perform an index get operation specialized for tuples.
    #[cfg_attr(feature = "bench", inline(never))]
    #[tracing::instrument(skip(self))]
    fn op_tuple_index_get(
        &mut self,
        address: Address,
        index: usize,
        output: Address,
    ) -> Result<(), VmError> {
        {
            let value = {
                let value = self.stack.at(address)?;
                try_tuple_like_index_get(value, index)?.cloned()
            };

            if let CallResult::Ok(value) = value {
                self.stack.store(output, value)?;
                return Ok(());
            }
        }

        let value = self.stack.at(address)?.clone();

        if let CallResult::Ok(()) = self
            .context
            .call_index_fn(
                &mut self.stack,
                Protocol::GET,
                value.clone(),
                index,
                (),
                output,
            )?
            .or_call_offset_with(self)?
        {
            return Ok(());
        }

        Err(VmError::from(VmErrorKind::UnsupportedTupleIndexGet {
            target: self.stack.at(address)?.type_info()?,
        }))
    }

    /// Perform an index get operation specialized for tuples.
    #[cfg_attr(feature = "bench", inline(never))]
    #[tracing::instrument(skip(self))]
    fn op_tuple_index_set(
        &mut self,
        address: Address,
        value: Address,
        index: usize,
        output: Address,
    ) -> Result<(), VmError> {
        let mut pair = self.stack.interleaved_pair(address, value)?;

        if builtin(&mut pair, index)? {
            return Ok(());
        }

        let target = pair.first_mut().clone();
        let value = pair.second_mut().clone();

        if let CallResult::Ok(()) = self
            .context
            .call_index_fn(
                &mut self.stack,
                Protocol::SET,
                target,
                index,
                (value,),
                output,
            )?
            .or_call_offset_with(self)?
        {
            return Ok(());
        };

        return Err(VmError::from(VmErrorKind::UnsupportedTupleIndexSet {
            target: self.stack.at(address)?.type_info()?,
        }));

        /// Built-in implementation of getting a string index on an
        /// object-like type.
        fn builtin(pair: &mut InterleavedPairMut<'_>, index: usize) -> Result<bool, VmError> {
            match mem::take(pair.first_mut()) {
                Value::Unit => Ok(false),
                Value::Tuple(tuple) => {
                    let mut tuple = tuple.borrow_mut()?;

                    if let Some(target) = tuple.get_mut(index) {
                        *target = mem::take(pair.second_mut());
                        return Ok(true);
                    }

                    Ok(false)
                }
                Value::Vec(vec) => {
                    let mut vec = vec.borrow_mut()?;

                    if let Some(target) = vec.get_mut(index) {
                        *target = mem::take(pair.second_mut());
                        return Ok(true);
                    }

                    Ok(false)
                }
                Value::Result(result) => {
                    let mut result = result.borrow_mut()?;

                    let target = match &mut *result {
                        Ok(ok) if index == 0 => ok,
                        Err(err) if index == 0 => err,
                        _ => return Ok(false),
                    };

                    *target = mem::take(pair.second_mut());
                    Ok(true)
                }
                Value::Option(option) => {
                    let mut option = option.borrow_mut()?;

                    let target = match &mut *option {
                        Some(some) if index == 0 => some,
                        _ => return Ok(false),
                    };

                    *target = mem::take(pair.second_mut());
                    Ok(true)
                }
                Value::TupleStruct(tuple_struct) => {
                    let mut tuple_struct = tuple_struct.borrow_mut()?;

                    if let Some(target) = tuple_struct.get_mut(index) {
                        *target = mem::take(pair.second_mut());
                        return Ok(true);
                    }

                    Ok(false)
                }
                Value::Variant(variant) => {
                    let mut variant = variant.borrow_mut()?;

                    if let VariantData::Tuple(data) = variant.data_mut() {
                        if let Some(target) = data.get_mut(index) {
                            *target = mem::take(pair.second_mut());
                            return Ok(true);
                        }
                    }

                    Ok(false)
                }
                _ => Ok(false),
            }
        }
    }

    /// Perform a specialized index get operation on an object.
    #[cfg_attr(feature = "bench", inline(never))]
    #[tracing::instrument(skip(self))]
    fn op_object_index_get(
        &mut self,
        address: Address,
        field_slot: usize,
        output: Address,
    ) -> Result<(), VmError> {
        let target = self.stack.at(address)?;
        let index = self.unit.lookup_string(field_slot)?;

        if let CallResult::Ok(value) = builtin(target, index)? {
            match value {
                Some(value) => {
                    let value = value.clone();
                    self.stack.store(output, value)?;
                    return Ok(());
                }
                None => {
                    return Err(VmError::from(VmErrorKind::MissingIndexKey {
                        target: target.type_info()?,
                        index: Key::String(StringKey::StaticString(index.clone())),
                    }));
                }
            }
        }

        let hash = index.hash();
        let target = target.clone();

        if let CallResult::Ok(()) =
            self.context
                .call_field_fn(&mut self.stack, Protocol::GET, target, hash, (), output)?
        {
            return Ok(());
        }

        return Err(VmError::from(VmErrorKind::UnsupportedObjectSlotIndexGet {
            target: self.stack.at(address)?.type_info()?,
        }));

        /// Built-in implementation of the operation.
        fn builtin(target: &Value, index: &str) -> Result<CallResult<Option<Value>>, VmError> {
            match target {
                Value::Object(object) => {
                    Ok(CallResult::Ok(object.borrow_ref()?.get(index).cloned()))
                }
                Value::Struct(typed_object) => Ok(CallResult::Ok(
                    typed_object.borrow_ref()?.get(index).cloned(),
                )),
                Value::Variant(variant) => {
                    let output =
                        BorrowRef::try_map(variant.borrow_ref()?, |value| match value.data() {
                            VariantData::Struct(data) => Some(data),
                            _ => None,
                        });

                    if let Some(output) = output {
                        return Ok(CallResult::Ok(output.get(index).cloned()));
                    }

                    Ok(CallResult::Unsupported)
                }
                _ => Ok(CallResult::Unsupported),
            }
        }
    }

    /// Perform a specialized index set operation on an object.
    #[cfg_attr(feature = "bench", inline(never))]
    #[tracing::instrument(skip(self))]
    fn op_object_index_set(
        &mut self,
        address: Address,
        field_slot: usize,
        value_address: Address,
        output: Address,
    ) -> Result<(), VmError> {
        let (target, value) = self.stack.pair(address, value_address)?;
        let field = self.unit.lookup_string(field_slot)?;

        if let CallResult::Ok(()) = builtin(target, field, value)? {
            return Ok(());
        };

        let hash = field.hash();
        let target = target.clone();
        let value = value.clone();

        if let CallResult::Ok(()) = self.context.call_field_fn(
            &mut self.stack,
            Protocol::SET,
            target,
            hash,
            (value,),
            output,
        )? {
            return Ok(());
        }

        return Err(VmError::from(VmErrorKind::UnsupportedObjectSlotIndexSet {
            target: self.stack.at(address)?.type_info()?,
        }));

        /// Built-in implementation of the operation.
        fn builtin(
            target: &mut Value,
            field: &StaticString,
            value: &Value,
        ) -> Result<CallResult<()>, VmError> {
            match target {
                Value::Object(object) => {
                    let mut object = object.borrow_mut()?;
                    object.insert(field.as_str().to_owned(), value.clone());
                    Ok(CallResult::Ok(()))
                }
                Value::Struct(typed_object) => {
                    let mut typed_object = typed_object.borrow_mut()?;

                    if let Some(v) = typed_object.get_mut(field.as_str()) {
                        *v = value.clone();
                        return Ok(CallResult::Ok(()));
                    }

                    Err(VmError::from(VmErrorKind::MissingField {
                        field: field.as_str().to_owned(),
                        target: typed_object.type_info(),
                    }))
                }
                Value::Variant(variant) => {
                    let mut variant = variant.borrow_mut()?;

                    if let VariantData::Struct(data) = variant.data_mut() {
                        if let Some(v) = data.get_mut(field.as_str()) {
                            *v = value.clone();
                            return Ok(CallResult::Ok(()));
                        }

                        return Err(VmError::from(VmErrorKind::MissingField {
                            field: field.as_str().to_owned(),
                            target: variant.type_info(),
                        }));
                    }

                    Ok(CallResult::Unsupported)
                }
                _ => Ok(CallResult::Unsupported),
            }
        }
    }

    /// Operation to allocate an object.
    #[cfg_attr(feature = "bench", inline(never))]
    #[tracing::instrument(skip(self))]
    fn op_object(&mut self, address: Address, slot: usize, output: Address) -> Result<(), VmError> {
        let keys = self
            .unit
            .lookup_object_keys(slot)
            .ok_or(VmErrorKind::MissingStaticObjectKeys { slot })?;

        let mut object = Object::with_capacity(keys.len());
        let values = self.stack.drain_at(address, keys.len())?;

        for (key, value) in keys.iter().zip(values) {
            object.insert(key.clone(), value);
        }

        self.stack.store(output, Shared::new(object))?;
        Ok(())
    }

    /// Operation to allocate an object.
    #[cfg_attr(feature = "bench", inline(never))]
    #[tracing::instrument(skip(self))]
    fn op_range(
        &mut self,
        from: Address,
        to: Address,
        limits: InstRangeLimits,
        output: Address,
    ) -> Result<(), VmError> {
        let mut pair = self.stack.interleaved_pair(from, to)?;
        let start = Option::<Value>::from_value(mem::take(pair.first_mut()))?;
        let end = Option::<Value>::from_value(mem::take(pair.second_mut()))?;

        let limits = match limits {
            InstRangeLimits::HalfOpen => RangeLimits::HalfOpen,
            InstRangeLimits::Closed => RangeLimits::Closed,
        };

        let range = Range::new(start, end, limits);
        self.stack.store(output, Shared::new(range))?;
        Ok(())
    }

    /// Operation to allocate an empty struct.
    #[cfg_attr(feature = "bench", inline(never))]
    #[tracing::instrument(skip(self))]
    fn op_empty_struct(&mut self, hash: Hash, output: Address) -> Result<(), VmError> {
        let rtti = self
            .unit
            .lookup_rtti(hash)
            .ok_or(VmErrorKind::MissingRtti { hash })?;

        self.stack
            .store(output, UnitStruct { rtti: rtti.clone() })?;
        Ok(())
    }

    /// Operation to allocate an object struct.
    #[cfg_attr(feature = "bench", inline(never))]
    #[tracing::instrument(skip(self))]
    fn op_struct(
        &mut self,
        hash: Hash,
        address: Address,
        slot: usize,
        output: Address,
    ) -> Result<(), VmError> {
        let keys = self
            .unit
            .lookup_object_keys(slot)
            .ok_or(VmErrorKind::MissingStaticObjectKeys { slot })?;

        let rtti = self
            .unit
            .lookup_rtti(hash)
            .ok_or(VmErrorKind::MissingRtti { hash })?;

        let values = self.stack.drain_at(address, keys.len())?;
        let mut data = Object::with_capacity(keys.len());

        for (key, value) in keys.iter().zip(values) {
            data.insert(key.clone(), value);
        }

        self.stack.store(
            output,
            Struct {
                rtti: rtti.clone(),
                data,
            },
        )?;

        Ok(())
    }

    /// Operation to allocate an object.
    #[cfg_attr(feature = "bench", inline(never))]
    #[tracing::instrument(skip(self))]
    fn op_unit_variant(&mut self, hash: Hash, output: Address) -> Result<(), VmError> {
        let rtti = self
            .unit
            .lookup_variant_rtti(hash)
            .ok_or(VmErrorKind::MissingVariantRtti { hash })?;

        self.stack.store(output, Variant::unit(rtti.clone()))?;
        Ok(())
    }

    /// Operation to allocate an object variant.
    #[cfg_attr(feature = "bench", inline(never))]
    #[tracing::instrument(skip(self))]
    fn op_struct_variant(
        &mut self,
        hash: Hash,
        address: Address,
        slot: usize,
        output: Address,
    ) -> Result<(), VmError> {
        let keys = self
            .unit
            .lookup_object_keys(slot)
            .ok_or(VmErrorKind::MissingStaticObjectKeys { slot })?;

        let rtti = self
            .unit
            .lookup_variant_rtti(hash)
            .ok_or(VmErrorKind::MissingVariantRtti { hash })?;

        let mut data = Object::with_capacity(keys.len());
        let values = self.stack.drain_at(address, keys.len())?;

        for (key, value) in keys.iter().zip(values) {
            data.insert(key.clone(), value);
        }

        self.stack
            .store(output, Variant::struct_(rtti.clone(), data))?;
        Ok(())
    }

    #[cfg_attr(feature = "bench", inline(never))]
    #[tracing::instrument(skip(self))]
    fn op_string(&mut self, slot: usize, output: Address) -> Result<(), VmError> {
        let string = self.unit.lookup_string(slot)?;
        self.stack.store(output, string.clone())?;
        Ok(())
    }

    #[cfg_attr(feature = "bench", inline(never))]
    #[tracing::instrument(skip(self))]
    fn op_bytes(&mut self, slot: usize, output: Address) -> Result<(), VmError> {
        let bytes = self.unit.lookup_bytes(slot)?.to_owned();
        self.stack.store(output, Bytes::from_vec(bytes))?;
        Ok(())
    }

    /// Optimize operation to perform string concatenation.
    #[cfg_attr(feature = "bench", inline(never))]
    #[tracing::instrument(skip(self))]
    fn op_string_concat(
        &mut self,
        mut address: Address,
        count: usize,
        size_hint: usize,
        output: Address,
    ) -> Result<(), VmError> {
        let mut out = String::with_capacity(size_hint);
        let mut buf = String::with_capacity(16);

        for _ in 0..count {
            buf.clear();
            let value = self.stack.at(address)?;

            if let Err(fmt::Error) = value.string_display_with(&mut out, &mut buf, &mut *self)? {
                return Err(VmError::from(VmErrorKind::FormatError));
            }

            address = address.step()?;
        }

        self.stack.store(output, out)?;
        Ok(())
    }

    /// Push a format specification onto the stack.
    #[cfg_attr(feature = "bench", inline(never))]
    #[tracing::instrument(skip(self))]
    fn op_format(
        &mut self,
        value: Address,
        spec: FormatSpec,
        output: Address,
    ) -> Result<(), VmError> {
        let mut pair = self.stack.interleaved_pair(value, output)?;

        *pair.second_mut() = Value::from(Format {
            value: pair.first_mut().clone(),
            spec,
        });

        Ok(())
    }

    #[cfg_attr(feature = "bench", inline(never))]
    #[tracing::instrument(skip(self))]
    fn op_match_value(
        &mut self,
        address: Address,
        value: InstValue,
        offset: isize,
    ) -> Result<(), VmError> {
        let matches = match (self.stack.at(address)?, value) {
            (Value::Unit, InstValue::Unit) => true,
            (Value::Bool(a), InstValue::Bool(b)) => *a == b,
            (Value::Byte(a), InstValue::Byte(b)) => *a == b,
            (Value::Char(a), InstValue::Char(b)) => *a == b,
            (Value::Integer(a), InstValue::Integer(b)) => *a == b,
            (Value::Float(a), InstValue::Float(b)) => *a == b,
            (Value::Type(a), InstValue::Type(b)) => *a == b,
            _ => false,
        };

        if !matches {
            self.modify_ip(offset);
        }

        Ok(())
    }

    /// Perform the try operation on the given stack location.
    #[cfg_attr(feature = "bench", inline(never))]
    #[tracing::instrument(skip(self))]
    fn op_try(&mut self, address: Address, output: Address) -> Result<bool, VmError> {
        let mut pair = self.stack.interleaved_pair(address, output)?;
        let return_value = pair.first_mut();

        let unwrapped = match return_value {
            Value::Result(result) => match &*result.borrow_ref()? {
                Result::Ok(value) => Some(value.clone()),
                Result::Err(..) => None,
            },
            Value::Option(option) => option.borrow_ref()?.as_ref().cloned(),
            other => {
                return Err(VmError::from(VmErrorKind::UnsupportedTryOperand {
                    actual: other.type_info()?,
                }))
            }
        };

        if let Some(value) = unwrapped {
            self.stack.store(output, value)?;
            return Ok(false);
        }

        let return_value = mem::take(return_value);
        self.op_return_internal(return_value)
    }

    /// Test if the top of stack is equal to the string at the given static
    /// string slot.
    #[cfg_attr(feature = "bench", inline(never))]
    #[tracing::instrument(skip(self))]
    fn op_match_string(
        &mut self,
        address: Address,
        slot: usize,
        offset: isize,
    ) -> Result<(), VmError> {
        let equal = match self.stack.at(address)? {
            Value::String(actual) => {
                let string = self.unit.lookup_string(slot)?;
                let actual = actual.borrow_ref()?;
                *actual == ***string
            }
            Value::StaticString(actual) => {
                let string = self.unit.lookup_string(slot)?;
                ***actual == ***string
            }
            _ => false,
        };

        if !equal {
            self.modify_ip(offset);
        }

        Ok(())
    }

    /// Test if the top of stack is equal to the string at the given static
    /// bytes slot.
    #[cfg_attr(feature = "bench", inline(never))]
    #[tracing::instrument(skip(self))]
    fn op_match_bytes(
        &mut self,
        address: Address,
        slot: usize,
        offset: isize,
    ) -> Result<(), VmError> {
        let equal = match self.stack.at(address)? {
            Value::Bytes(actual) => *actual.borrow_ref()? == *self.unit.lookup_bytes(slot)?,
            _ => false,
        };

        if !equal {
            self.modify_ip(offset);
        }

        Ok(())
    }

    #[cfg_attr(feature = "bench", inline(never))]
    #[tracing::instrument(skip(self))]
    fn op_match_type(
        &mut self,
        address: Address,
        type_hash: Hash,
        offset: isize,
    ) -> Result<(), VmError> {
        if self.stack.at(address)?.type_hash()? != type_hash {
            self.modify_ip(offset);
        }

        Ok(())
    }

    #[cfg_attr(feature = "bench", inline(never))]
    #[tracing::instrument(skip(self))]
    fn op_match_variant(
        &mut self,
        address: Address,
        variant_hash: Hash,
        enum_hash: Hash,
        index: usize,
        offset: isize,
        output: Address,
    ) -> Result<(), VmError> {
        match self.stack.at(address)?.clone() {
            Value::Variant(variant) => {
                let variant = variant.borrow_ref()?;

                if variant.rtti().hash != variant_hash {
                    self.modify_ip(offset);
                }
            }
            Value::Any(any) => {
                if any.borrow_ref()?.type_hash() != enum_hash {
                    self.modify_ip(offset);
                }

                if matches!(
                    self.call_is_variant(&any, index, output)?,
                    CallResult::Unsupported | CallResult::Ok(false)
                ) {
                    self.modify_ip(offset);
                }
            }
            _ => {
                self.modify_ip(offset);
            }
        }

        Ok(())
    }

    #[cfg_attr(feature = "bench", inline(never))]
    #[tracing::instrument(skip(self))]
    fn op_match_sequence(
        &mut self,
        address: Address,
        ty: TypeCheck,
        len: usize,
        exact: bool,
        offset: isize,
        output: Address,
    ) -> Result<(), VmError> {
        let value = self.stack.at(address)?.clone();

        match self.on_tuple(ty, &value)? {
            Some(tuple) => match tuple.get(..len) {
                Some(values) if !exact || exact && tuple.len() == len => {
                    self.stack.write_at(output, values.iter().cloned())?;
                }
                _ => {
                    self.modify_ip(offset);
                }
            },
            None => {
                self.modify_ip(offset);
                return Ok(());
            }
        };

        Ok(())
    }

    #[cfg_attr(feature = "bench", inline(never))]
    #[tracing::instrument(skip(self))]
    fn op_match_builtin(
        &mut self,
        address: Address,
        type_check: TypeCheck,
        offset: isize,
    ) -> Result<(), VmError> {
        let value = self.stack.at(address)?;

        let is_match = match (type_check, value) {
            (TypeCheck::Tuple, Value::Tuple(..)) => true,
            (TypeCheck::Vec, Value::Vec(..)) => true,
            (TypeCheck::Result(v), Value::Result(result)) => {
                let result = result.borrow_ref()?;

                match (v, &*result) {
                    (0, Ok(..)) => true,
                    (1, Err(..)) => true,
                    _ => false,
                }
            }
            (TypeCheck::Option(v), Value::Option(option)) => {
                let option = option.borrow_ref()?;

                match (v, &*option) {
                    (0, Some(..)) => true,
                    (1, None) => true,
                    _ => false,
                }
            }
            (TypeCheck::GeneratorState(v), Value::GeneratorState(state)) => {
                use crate::runtime::GeneratorState::*;
                let state = state.borrow_ref()?;

                match (v, &*state) {
                    (0, Complete(..)) => true,
                    (1, Yielded(..)) => true,
                    _ => false,
                }
            }
            (TypeCheck::Unit, Value::Unit) => true,
            _ => false,
        };

        if !is_match {
            self.modify_ip(offset);
        }

        Ok(())
    }

    #[cfg_attr(feature = "bench", inline(never))]
    #[tracing::instrument(skip(self))]
    fn op_match_object(
        &mut self,
        address: Address,
        slot: usize,
        exact: bool,
        offset: isize,
        output: Address,
    ) -> Result<(), VmError> {
        match self.stack.at(address)?.clone() {
            Value::Object(object) => {
                let object = object.borrow_ref()?;

                let keys = self
                    .unit
                    .lookup_object_keys(slot)
                    .ok_or(VmErrorKind::MissingStaticObjectKeys { slot })?;

                if !test(&*object, keys, exact) {
                    self.modify_ip(offset);
                    return Ok(());
                }

                let values = keys.iter().flat_map(|key| Some(object.get(key)?.clone()));
                self.stack.write_at(output, values)?;
            }
            _ => {
                self.modify_ip(offset);
            }
        };

        return Ok(());

        fn test(object: &Object, keys: &[String], exact: bool) -> bool {
            if exact && object.len() != keys.len() || object.len() < keys.len() {
                return false;
            }

            for key in keys {
                if !object.contains_key(key) {
                    return false;
                }
            }

            true
        }
    }

    /// Push the given variant onto the stack.
    #[cfg_attr(feature = "bench", inline(never))]
    #[tracing::instrument(skip(self))]
    fn op_variant(
        &mut self,
        address: Address,
        variant: InstVariant,
        output: Address,
    ) -> Result<(), VmError> {
        // TODO: address is unused for InstVariant::None.
        let mut pair = self.stack.interleaved_pair(address, output)?;

        let value = match variant {
            InstVariant::Some => {
                let some = mem::take(pair.first_mut());
                Value::Option(Shared::new(Some(some)))
            }
            InstVariant::None => Value::Option(Shared::new(None)),
            InstVariant::Ok => {
                let some = mem::take(pair.first_mut());
                Value::Result(Shared::new(Ok(some)))
            }
            InstVariant::Err => {
                let some = mem::take(pair.first_mut());
                Value::Result(Shared::new(Err(some)))
            }
        };

        *pair.second_mut() = value;
        Ok(())
    }

    /// Load a function as a value onto the stack.
    #[cfg_attr(feature = "bench", inline(never))]
    #[tracing::instrument(skip(self))]
    fn op_load_fn(&mut self, hash: Hash, output: Address) -> Result<(), VmError> {
        let function = match self.unit.function(hash) {
            Some(info) => match info {
                UnitFn::Offset {
                    offset,
                    call,
                    args,
                    frame,
                } => Function::from_vm_offset(
                    self.context.clone(),
                    self.unit.clone(),
                    offset,
                    call,
                    args,
                    frame,
                    hash,
                ),
                UnitFn::UnitStruct { hash } => {
                    let rtti = self
                        .unit
                        .lookup_rtti(hash)
                        .ok_or(VmErrorKind::MissingRtti { hash })?;

                    Function::from_unit_struct(rtti.clone())
                }
                UnitFn::TupleStruct { hash, args } => {
                    let rtti = self
                        .unit
                        .lookup_rtti(hash)
                        .ok_or(VmErrorKind::MissingRtti { hash })?;

                    Function::from_tuple_struct(rtti.clone(), args)
                }
                UnitFn::UnitVariant { hash } => {
                    let rtti = self
                        .unit
                        .lookup_variant_rtti(hash)
                        .ok_or(VmErrorKind::MissingVariantRtti { hash })?;

                    Function::from_unit_variant(rtti.clone())
                }
                UnitFn::TupleVariant { hash, args } => {
                    let rtti = self
                        .unit
                        .lookup_variant_rtti(hash)
                        .ok_or(VmErrorKind::MissingVariantRtti { hash })?;

                    Function::from_tuple_variant(rtti.clone(), args)
                }
            },
            None => {
                let handler = self
                    .context
                    .function(hash)
                    .ok_or(VmErrorKind::MissingFunction { hash })?;

                Function::from_handler(handler.clone(), hash)
            }
        };

        *self.stack.at_mut(output)? = Value::Function(Shared::new(function));
        Ok(())
    }

    /// Construct a closure on the top of the stack.
    #[cfg_attr(feature = "bench", inline(never))]
    #[tracing::instrument(skip(self))]
    fn op_closure(
        &mut self,
        hash: Hash,
        address: Address,
        count: usize,
        output: Address,
    ) -> Result<(), VmError> {
        let info = self
            .unit
            .function(hash)
            .ok_or(VmErrorKind::MissingFunction { hash })?;

        let (offset, call, args, frame) = match info {
            UnitFn::Offset {
                offset,
                call,
                args,
                frame,
            } => (offset, call, args, frame),
            _ => return Err(VmError::from(VmErrorKind::MissingFunction { hash })),
        };

        let environment = self.stack.drain_at(address, count)?.collect();

        let function = Function::from_vm_closure(
            self.context.clone(),
            self.unit.clone(),
            offset,
            call,
            args,
            frame,
            environment,
            hash,
        );

        *self.stack.at_mut(output)? = Value::Function(Shared::new(function));
        Ok(())
    }

    /// Implementation of a function call.
    #[cfg_attr(feature = "bench", inline(never))]
    #[tracing::instrument(skip(self))]
    fn op_call(
        &mut self,
        hash: Hash,
        address: Address,
        count: usize,
        output: Address,
    ) -> Result<(), VmError> {
        match self.unit.function(hash) {
            Some(info) => match info {
                UnitFn::Offset {
                    offset,
                    call,
                    args: expected,
                    frame,
                } => {
                    check_args(count, expected)?;
                    self.call_offset_fn(offset, call, address, count, frame, output)?;
                }
                UnitFn::UnitStruct { hash } => {
                    check_args(count, 0)?;

                    let rtti = self
                        .unit
                        .lookup_rtti(hash)
                        .ok_or(VmErrorKind::MissingRtti { hash })?;

                    self.stack.store(output, Value::unit_struct(rtti.clone()))?;
                }
                UnitFn::TupleStruct {
                    hash,
                    args: expected,
                } => {
                    check_args(count, expected)?;
                    let tuple = self.stack.drain_at(address, count)?.collect();

                    let rtti = self
                        .unit
                        .lookup_rtti(hash)
                        .ok_or(VmErrorKind::MissingRtti { hash })?;

                    self.stack
                        .store(output, Value::tuple_struct(rtti.clone(), tuple))?;
                }
                UnitFn::TupleVariant {
                    hash,
                    args: expected,
                } => {
                    check_args(count, expected)?;

                    let rtti = self
                        .unit
                        .lookup_variant_rtti(hash)
                        .ok_or(VmErrorKind::MissingVariantRtti { hash })?;

                    let tuple = self.stack.drain_at(address, count)?.collect();
                    self.stack
                        .store(output, Value::tuple_variant(rtti.clone(), tuple))?;
                }
                UnitFn::UnitVariant { hash } => {
                    check_args(count, 0)?;

                    let rtti = self
                        .unit
                        .lookup_variant_rtti(hash)
                        .ok_or(VmErrorKind::MissingVariantRtti { hash })?;

                    self.stack
                        .store(output, Value::unit_variant(rtti.clone()))?;
                }
            },
            None => {
                if !self
                    .context
                    .fn_call(hash, &mut self.stack, address, count, output)?
                {
                    return Err(VmError::from(VmErrorKind::MissingFunction { hash }));
                }
            }
        }

        Ok(())
    }

    #[cfg_attr(feature = "bench", inline(never))]
    #[tracing::instrument(skip(self))]
    fn op_call_instance(
        &mut self,
        hash: Hash,
        address: Address,
        count: usize,
        output: Address,
    ) -> Result<(), VmError> {
        let count = count + 1;
        let type_hash = self.stack.at(address)?.type_hash()?;
        let hash = Hash::instance_function(type_hash, hash);
        let unit_fn = self.unit.function(hash);

        if let Some(UnitFn::Offset {
            offset,
            call,
            args: expected,
            frame,
        }) = unit_fn
        {
            check_args(count, expected)?;
            self.call_offset_fn(offset, call, address, count, frame, output)?;
            return Ok(());
        }

        if !self
            .context
            .fn_call(hash, &mut self.stack, address, count, output)?
        {
            return Err(VmError::from(VmErrorKind::MissingInstanceFunction {
                instance: self.stack.at(Address::BASE)?.type_info()?,
                hash,
            }));
        }

        Ok(())
    }

    #[cfg_attr(feature = "bench", inline(never))]
    #[tracing::instrument(skip(self))]
    fn op_call_fn(
        &mut self,
        function: Address,
        address: Address,
        count: usize,
        output: Address,
    ) -> Result<Option<VmHalt>, VmError> {
        let function = self.stack.at(function)?;

        let hash = match function {
            Value::Type(hash) => *hash,
            Value::Function(ref function) => {
                let function = function.clone().into_ref()?;
                return function.call_with_vm(self, address, count, output);
            }
            actual => {
                let actual_type = actual.type_info()?;
                return Err(VmError::from(VmErrorKind::UnsupportedCallFn {
                    actual_type,
                }));
            }
        };

        self.op_call(hash, address, count, output)?;
        Ok(None)
    }

    #[cfg_attr(feature = "bench", inline(never))]
    #[tracing::instrument(skip(self))]
    fn op_iter_next(
        &mut self,
        address: Address,
        offset: isize,
        output: Address,
    ) -> Result<(), VmError> {
        let iterator = self.stack.at(address)?;

        let value = match iterator {
            Value::Iterator(iterator) => iterator.borrow_mut()?.next()?,
            Value::Any(any) => {
                let any = any.clone();

                if let CallResult::Unsupported = self.call_next(&any, output)? {
                    return Err(VmError::from(VmErrorKind::UnsupportedIterNextOperand {
                        actual: any.borrow_ref()?.type_info(),
                    }));
                }

                match &*mem::take(self.stack.at_mut(output)?)
                    .into_option()?
                    .borrow_ref()?
                {
                    Some(some) => Some(some.clone()),
                    None => None,
                }
            }
            other => {
                return Err(VmError::from(VmErrorKind::UnsupportedIterNextOperand {
                    actual: other.type_info()?,
                }))
            }
        };

        match value {
            Some(value) => {
                *self.stack.at_mut(output)? = value;
            }
            None => {
                self.modify_ip(offset);
            }
        }

        Ok(())
    }

    /// Call the provided closure within the context of this virtual machine.
    ///
    /// This allows for calling protocol function helpers like
    /// [Value::string_display] which requires access to a virtual machine.
    ///
    /// ```,no_run
    /// use rune::{Context, Unit, FromValue, Source};
    /// use std::sync::Arc;
    ///
    /// # fn main() -> rune::Result<()> {
    /// let context = Context::with_default_modules()?;
    /// let context = Arc::new(context.runtime());
    ///
    /// // Normally the unit would be created by compiling some source,
    /// // and since this one is empty it'll just error.
    /// let unit = Arc::new(Unit::default());
    ///
    /// let mut vm = rune::Vm::new(context, unit);
    ///
    /// let output = vm.execute(&["main"], ())?.complete()?;
    ///
    /// // Call the string_display protocol on `output`. This requires
    /// // access to a virtual machine since it might use functions
    /// // registered in the unit associated with it.
    /// let mut s = String::new();
    /// let mut buf = String::new();
    ///
    /// // Note: We do an extra unwrap because the return value is
    /// // `fmt::Result`.
    /// vm.with(|| output.string_display(&mut s, &mut buf))?.expect("formatting should succeed");
    /// # Ok(()) }
    /// ```
    pub fn with<F, T>(&mut self, f: F) -> T
    where
        F: FnOnce() -> T,
    {
        let _guard = crate::runtime::env::Guard::new(&self.context, &self.unit);
        f()
    }

    /// Evaluate a single instruction.
    #[tracing::instrument(skip(self))]
    pub(crate) fn run(&mut self) -> Result<VmHalt, VmError> {
        // NB: set up environment so that native function can access context and
        // unit.
        let _guard = crate::runtime::env::Guard::new(&self.context, &self.unit);

        loop {
            if !budget::take() {
                return Ok(VmHalt::Limited);
            }

            let inst = *self
                .unit
                .instruction_at(self.ip)
                .ok_or(VmErrorKind::IpOutOfBounds)?;

            tracing::trace!("{}: {}", self.ip, inst);

            match inst {
                Inst::Not { address, output } => {
                    self.op_not(address, output)?;
                }
                Inst::Neg { address, output } => {
                    self.op_neg(address, output)?;
                }
                Inst::Closure {
                    hash,
                    address,
                    count,
                    output,
                } => {
                    self.op_closure(hash, address, count, output)?;
                }
                Inst::Call {
                    hash,
                    address,
                    count,
                    output,
                } => {
                    self.op_call(hash, address, count, output)?;
                }
                Inst::CallInstance {
                    hash,
                    address,
                    count,
                    output,
                } => {
                    self.op_call_instance(hash, address, count, output)?;
                }
                Inst::CallFn {
                    function,
                    address,
                    count,
                    output,
                } => {
                    if let Some(reason) = self.op_call_fn(function, address, count, output)? {
                        return Ok(reason);
                    }
                }
                Inst::LoadInstanceFn {
                    address,
                    hash,
                    output,
                } => {
                    self.op_load_instance_fn(address, hash, output)?;
                }
                Inst::IndexGet {
                    address,
                    index,
                    output,
                } => {
                    self.op_index_get(address, index, output)?;
                }
                Inst::TupleIndexGet {
                    address,
                    index,
                    output,
                } => {
                    self.op_tuple_index_get(address, index, output)?;
                }
                Inst::TupleIndexSet {
                    address,
                    value,
                    index,
                    output,
                } => {
                    self.op_tuple_index_set(address, value, index, output)?;
                }
                Inst::ObjectIndexGet {
                    address,
                    slot,
                    output,
                } => {
                    self.op_object_index_get(address, slot, output)?;
                }
                Inst::ObjectIndexSet {
                    address,
                    slot,
                    value,
                    output,
                } => {
                    self.op_object_index_set(address, slot, value, output)?;
                }
                Inst::IndexSet {
                    address,
                    index,
                    value,
                    output,
                } => {
                    self.op_index_set(address, index, value, output)?;
                }
                Inst::Return { address } => {
                    if self.op_return(address)? {
                        self.advance();
                        return Ok(VmHalt::Exited);
                    }
                }
                Inst::ReturnUnit => {
                    if self.op_return_unit()? {
                        self.advance();
                        return Ok(VmHalt::Exited);
                    }
                }
                Inst::Await { address, output } => {
                    let future = self.op_await(address)?;
                    // NB: the future itself will advance the virtual machine.
                    return Ok(VmHalt::Awaited(Awaited::Future(future, output)));
                }
                Inst::Select {
                    address,
                    len,
                    output,
                    branch_output,
                } => {
                    if let Some(select) = self.op_select(address, len, output)? {
                        // NB: the future itself will advance the virtual machine.
                        return Ok(VmHalt::Awaited(Awaited::Select(
                            select,
                            output,
                            branch_output,
                        )));
                    }
                }
                Inst::LoadFn { hash, output } => {
                    self.op_load_fn(hash, output)?;
                }
                Inst::Store { value, output } => {
                    self.op_store(value, output)?;
                }
                Inst::Copy { address, output } => {
                    self.op_copy(address, output)?;
                }
                Inst::Move { address, output } => {
                    self.op_move(address, output)?;
                }
                Inst::Drop { address } => {
                    self.op_drop(address)?;
                }
                Inst::Jump { offset } => {
                    self.op_jump(offset)?;
                }
                Inst::JumpIf { address, offset } => {
                    self.op_jump_if(address, offset, true)?;
                }
                Inst::JumpIfNot { address, offset } => {
                    self.op_jump_if(address, offset, false)?;
                }
                Inst::JumpIfBranch {
                    address,
                    branch,
                    offset,
                } => {
                    self.op_jump_if_branch(address, branch, offset)?;
                }
                Inst::Vec {
                    address,
                    count,
                    output,
                } => {
                    self.op_vec(address, count, output)?;
                }
                Inst::Tuple {
                    address,
                    count,
                    output,
                } => {
                    self.op_tuple(address, count, output)?;
                }
                Inst::Tuple1 { args, output } => {
                    self.op_tuple_n(&args[..], output)?;
                }
                Inst::Tuple2 { args, output } => {
                    self.op_tuple_n(&args[..], output)?;
                }
                Inst::Tuple3 { args, output } => {
                    self.op_tuple_n(&args[..], output)?;
                }
                Inst::Tuple4 { args, output } => {
                    self.op_tuple_n(&args[..], output)?;
                }
                Inst::UnpackTuple { address, output } => {
                    self.op_unpack_tuple(address, output)?;
                }
                Inst::Object {
                    address,
                    slot,
                    output,
                } => {
                    self.op_object(address, slot, output)?;
                }
                Inst::Range {
                    from,
                    to,
                    limits,
                    output,
                } => {
                    self.op_range(from, to, limits, output)?;
                }
                Inst::UnitStruct { hash, output } => {
                    self.op_empty_struct(hash, output)?;
                }
                Inst::Struct {
                    hash,
                    address,
                    slot,
                    output,
                } => {
                    self.op_struct(hash, address, slot, output)?;
                }
                Inst::UnitVariant { hash, output } => {
                    self.op_unit_variant(hash, output)?;
                }
                Inst::StructVariant {
                    hash,
                    address,
                    slot,
                    output,
                } => {
                    self.op_struct_variant(hash, address, slot, output)?;
                }
                Inst::String { slot, output } => {
                    self.op_string(slot, output)?;
                }
                Inst::Bytes { slot, output } => {
                    self.op_bytes(slot, output)?;
                }
                Inst::StringConcat {
                    address,
                    count,
                    size_hint,
                    output,
                } => {
                    self.op_string_concat(address, count, size_hint, output)?;
                }
                Inst::Format {
                    address,
                    spec,
                    output,
                } => {
                    self.op_format(address, spec, output)?;
                }
                Inst::Try { address, output } => {
                    if self.op_try(address, output)? {
                        self.advance();
                        return Ok(VmHalt::Exited);
                    }
                }
                Inst::MatchValue {
                    address,
                    value,
                    offset,
                } => {
                    self.op_match_value(address, value, offset)?;
                }
                Inst::MatchString {
                    address,
                    slot,
                    offset,
                } => {
                    self.op_match_string(address, slot, offset)?;
                }
                Inst::MatchBytes {
                    address,
                    slot,
                    offset,
                } => {
                    self.op_match_bytes(address, slot, offset)?;
                }
                Inst::MatchType {
                    address,
                    type_hash,
                    offset,
                } => {
                    self.op_match_type(address, type_hash, offset)?;
                }
                Inst::MatchVariant {
                    address,
                    variant_hash,
                    enum_hash,
                    index,
                    offset,
                    output,
                } => {
                    self.op_match_variant(address, variant_hash, enum_hash, index, offset, output)?;
                }
                Inst::MatchSequence {
                    address,
                    type_check,
                    len,
                    exact,
                    offset,
                    output,
                } => {
                    self.op_match_sequence(address, type_check, len, exact, offset, output)?;
                }
                Inst::MatchBuiltIn {
                    address,
                    type_check,
                    offset,
                } => {
                    self.op_match_builtin(address, type_check, offset)?;
                }
                Inst::MatchObject {
                    address,
                    slot,
                    exact,
                    offset,
                    output,
                } => {
                    self.op_match_object(address, slot, exact, offset, output)?;
                }
                Inst::Yield { address, output } => {
                    return Ok(VmHalt::Yielded {
                        address: Some(address),
                        output,
                    });
                }
                Inst::YieldUnit { output } => {
                    return Ok(VmHalt::Yielded {
                        address: None,
                        output,
                    });
                }
                Inst::Variant {
                    address,
                    variant,
                    output,
                } => {
                    self.op_variant(address, variant, output)?;
                }
                Inst::Op { op, a, b, output } => {
                    self.op_op(op, a, b, output)?;
                }
                Inst::Assign {
                    lhs,
                    rhs,
                    target,
                    op,
                    output,
                } => {
                    self.op_assign(lhs, rhs, target, op, output)?;
                }
                Inst::IterNext {
                    address,
                    offset,
                    output,
                } => {
                    self.op_iter_next(address, offset, output)?;
                }
                Inst::Panic { reason } => {
                    return Err(VmError::from(VmErrorKind::Panic {
                        reason: Panic::from(reason),
                    }));
                }
            }

            self.advance();
        }
    }
}

impl AsMut<Vm> for Vm {
    fn as_mut(&mut self) -> &mut Vm {
        self
    }
}

impl AsRef<Vm> for Vm {
    fn as_ref(&self) -> &Vm {
        self
    }
}

/// A call frame.
///
/// This is used to store the return point after an instruction has been run.
#[derive(Debug, Clone, Copy)]
pub struct CallFrame {
    /// The stored instruction pointer.
    ip: usize,
    /// The top of the stack at the time of the call to ensure stack isolation
    /// across function calls.
    ///
    /// I.e. a function should not be able to manipulate the size of any other
    /// stack than its own.
    stack_bottom: usize,
    /// The size of the stack when the call was entered.
    stack: usize,
    /// Where to write the return value of the stack frame.
    output: Address,
}

impl CallFrame {
    /// Get the instruction pointer of the call frame.
    pub fn ip(&self) -> usize {
        self.ip
    }

    /// Get the bottom of the stack of the current call frame.
    pub fn stack_bottom(&self) -> usize {
        self.stack_bottom
    }
}

/// Clear stack on drop.
struct ClearStack<'a>(&'a mut Vm);

impl Drop for ClearStack<'_> {
    fn drop(&mut self) {
        self.0.stack.clear();
    }
}

/// Check that arguments matches expected or raise the appropriate error.
pub(crate) fn check_args(args: usize, expected: usize) -> Result<(), VmError> {
    if args != expected {
        return Err(VmError::from(VmErrorKind::BadArgumentCount {
            actual: args,
            expected,
        }));
    }

    Ok(())
}

/// Implementation of getting a string index on an object-like type.
fn try_tuple_like_index_get(
    target: &Value,
    index: usize,
) -> Result<CallResult<BorrowRef<'_, Value>>, VmError> {
    let value = match target {
        Value::Unit => None,
        Value::Tuple(tuple) => BorrowRef::try_map(tuple.borrow_ref()?, |v| v.get(index)),
        Value::Vec(vec) => BorrowRef::try_map(vec.borrow_ref()?, |v| v.get(index)),
        Value::Result(result) => BorrowRef::try_map(result.borrow_ref()?, |v| match (index, v) {
            (0, Ok(value)) => Some(value),
            (1, Err(value)) => Some(value),
            _ => None,
        }),
        Value::Option(option) => BorrowRef::try_map(option.borrow_ref()?, |v| match (index, &*v) {
            (0, Some(value)) => Some(value),
            _ => None,
        }),
        Value::GeneratorState(state) => {
            use crate::runtime::GeneratorState::*;

            BorrowRef::try_map(state.borrow_ref()?, |v| match (index, &*v) {
                (0, Yielded(value)) => Some(value),
                (1, Complete(value)) => Some(value),
                _ => None,
            })
        }
        Value::TupleStruct(tuple) => {
            BorrowRef::try_map(tuple.borrow_ref()?, |v| v.data().get(index))
        }
        Value::Variant(variant) => {
            BorrowRef::try_map(variant.borrow_ref()?, |variant| match variant.data() {
                VariantData::Tuple(tuple) => tuple.get(index),
                _ => None,
            })
        }
        _ => return Ok(CallResult::Unsupported),
    };

    let value = match value {
        Some(value) => value,
        None => {
            return Err(VmError::from(VmErrorKind::MissingIndex {
                target: target.type_info()?,
                index: VmIntegerRepr::from(index),
            }));
        }
    };

    Ok(CallResult::Ok(value))
}

/// Implementation of getting a mutable value out of a tuple-like value.
fn try_tuple_like_index_get_mut(
    target: &Value,
    index: usize,
) -> Result<CallResult<BorrowMut<'_, Value>>, VmError> {
    let value = match target {
        Value::Tuple(tuple) => {
            let tuple = tuple.borrow_mut()?;
            BorrowMut::try_map(tuple, |tuple| tuple.get_mut(index))
        }
        Value::Vec(vec) => BorrowMut::try_map(vec.borrow_mut()?, |vec| vec.get_mut(index)),
        Value::Result(result) => {
            BorrowMut::try_map(result.borrow_mut()?, |result| match (index, result) {
                (0, Ok(value)) => Some(value),
                (1, Err(value)) => Some(value),
                _ => None,
            })
        }
        Value::Option(option) => {
            BorrowMut::try_map(option.borrow_mut()?, |option| match (index, option) {
                (0, Some(value)) => Some(value),
                _ => None,
            })
        }
        Value::GeneratorState(state) => {
            use crate::runtime::GeneratorState::*;

            BorrowMut::try_map(state.borrow_mut()?, |state| match (index, state) {
                (0, Yielded(value)) => Some(value),
                (1, Complete(value)) => Some(value),
                _ => None,
            })
        }
        Value::TupleStruct(tuple) => {
            BorrowMut::try_map(tuple.borrow_mut()?, |tuple| tuple.get_mut(index))
        }
        Value::Variant(variant) => {
            let data =
                BorrowMut::try_map(variant.borrow_mut()?, |variant| match variant.data_mut() {
                    VariantData::Tuple(data) => Some(data),
                    _ => None,
                });

            let data = match data {
                Some(data) => data,
                None => {
                    return Ok(CallResult::Unsupported);
                }
            };

            BorrowMut::try_map(data, |data| data.get_mut(index))
        }
        _ => return Ok(CallResult::Unsupported),
    };

    if let Some(value) = value {
        return Ok(CallResult::Ok(value));
    }

    Err(VmError::from(VmErrorKind::MissingIndex {
        target: target.type_info()?,
        index: VmIntegerRepr::from(index),
    }))
}

/// Implementation of getting a string index on an object-like type.
fn try_object_like_index_get<'a>(
    target: &'a Value,
    field: &str,
) -> Result<CallResult<BorrowRef<'a, Value>>, VmError> {
    let value = match &target {
        Value::Object(target) => BorrowRef::try_map(target.borrow_ref()?, |v| v.get(field)),
        Value::Struct(target) => BorrowRef::try_map(target.borrow_ref()?, |v| v.get(field)),
        Value::Variant(variant) => BorrowRef::try_map(variant.borrow_ref()?, |v| match v.data() {
            VariantData::Struct(target) => target.get(field),
            _ => None,
        }),
        _ => return Ok(CallResult::Unsupported),
    };

    let value = match value {
        Some(value) => value,
        None => {
            return Err(VmError::from(VmErrorKind::MissingField {
                target: target.type_info()?,
                field: field.to_owned(),
            }));
        }
    };

    Ok(CallResult::Ok(value))
}

/// Implementation of getting a mutable string index on an object-like type.
fn try_object_like_index_get_mut<'value>(
    target: &'value Value,
    field: &str,
) -> Result<CallResult<BorrowMut<'value, Value>>, VmError> {
    let value = match target {
        Value::Object(object) => {
            BorrowMut::try_map(object.borrow_mut()?, |object| object.get_mut(field))
        }
        Value::Struct(st) => BorrowMut::try_map(st.borrow_mut()?, |st| st.get_mut(field)),
        Value::Variant(variant) => {
            let data =
                BorrowMut::try_map(variant.borrow_mut()?, |variant| match variant.data_mut() {
                    VariantData::Struct(data) => Some(data),
                    _ => None,
                });

            let data = match data {
                Some(data) => data,
                None => return Ok(CallResult::Unsupported),
            };

            BorrowMut::try_map(data, |data| data.get_mut(field))
        }
        _ => return Ok(CallResult::Unsupported),
    };

    if let Some(value) = value {
        return Ok(CallResult::Ok(value));
    }

    Err(VmError::from(VmErrorKind::MissingField {
        target: target.type_info()?,
        field: field.to_owned(),
    }))
}

/// Helper to resolve an [InstTarget].
fn target_value<T>(
    vm: &mut Vm,
    lhs_address: Address,
    rhs_address: Address,
    target: InstTarget,
    protocol: Protocol,
    output: Address,
    builtin: T,
) -> Result<(), VmError>
where
    T: FnOnce(&mut Value, &Value) -> Result<CallResult<()>, VmError>,
{
    let (lhs, rhs) = vm.stack.pair(lhs_address, rhs_address)?;

    if let CallResult::Ok(()) = apply_builtin(&vm.unit, lhs, rhs, target, builtin)? {
        return Ok(());
    }

    let lhs = lhs.clone();
    let rhs = rhs.clone();

    if let CallResult::Ok(()) = vm
        .context
        .call_instance_fn(&mut vm.stack, &vm.unit, lhs, protocol, (rhs,), output)?
        .or_call_offset_with(vm)?
    {
        return Ok(());
    }

    return Err(VmError::from(VmErrorKind::UnsupportedBinaryOperation {
        op: protocol.name,
        lhs: vm.stack.at(lhs_address)?.type_info()?,
        rhs: vm.stack.at(rhs_address)?.type_info()?,
    }));

    fn apply_builtin<T>(
        unit: &Unit,
        lhs: &mut Value,
        rhs: &Value,
        target: InstTarget,
        builtin: T,
    ) -> Result<CallResult<()>, VmError>
    where
        T: FnOnce(&mut Value, &Value) -> Result<CallResult<()>, VmError>,
    {
        match target {
            InstTarget::Offset => builtin(lhs, rhs),
            InstTarget::TupleField(index) => {
                if let CallResult::Ok(mut value) = try_tuple_like_index_get_mut(lhs, index)? {
                    return builtin(&mut *value, rhs);
                }

                Err(VmError::from(VmErrorKind::UnsupportedTupleIndexGet {
                    target: lhs.type_info()?,
                }))
            }
            InstTarget::ObjectField(field) => {
                let field = unit.lookup_string(field)?;

                if let CallResult::Ok(mut value) = try_object_like_index_get_mut(lhs, field)? {
                    return builtin(&mut *value, rhs);
                }

                Err(VmError::from(VmErrorKind::UnsupportedObjectSlotIndexGet {
                    target: lhs.type_info()?,
                }))
            }
        }
    }
}

impl Vm {
    /// Unpack the given tuple.
    fn on_tuple<'a>(
        &mut self,
        type_check: TypeCheck,
        value: &'a Value,
    ) -> Result<Option<BorrowRef<'a, [Value]>>, VmError> {
        use std::slice;

        match (type_check, value) {
            (TypeCheck::Tuple, Value::Tuple(tuple)) => {
                Ok(Some(BorrowRef::map(tuple.borrow_ref()?, |value| &**value)))
            }
            (TypeCheck::Vec, Value::Vec(vec)) => {
                Ok(Some(BorrowRef::map(vec.borrow_ref()?, |value| &**value)))
            }
            (TypeCheck::Result(v), Value::Result(result)) => {
                Ok(BorrowRef::try_map(result.borrow_ref()?, |result| {
                    match (v, result) {
                        (0, Ok(ok)) => Some(slice::from_ref(ok)),
                        (1, Err(err)) => Some(slice::from_ref(err)),
                        _ => None,
                    }
                }))
            }
            (TypeCheck::Option(v), Value::Option(option)) => {
                Ok(BorrowRef::try_map(option.borrow_ref()?, |option| {
                    match (v, option) {
                        (0, Some(some)) => Some(slice::from_ref(some)),
                        (1, None) => Some(&[]),
                        _ => None,
                    }
                }))
            }
            (TypeCheck::GeneratorState(v), Value::GeneratorState(state)) => {
                use crate::runtime::GeneratorState::*;
                Ok(BorrowRef::try_map(state.borrow_ref()?, |state| {
                    match (v, state) {
                        (0, Complete(complete)) => Some(slice::from_ref(complete)),
                        (1, Yielded(yielded)) => Some(slice::from_ref(yielded)),
                        _ => None,
                    }
                }))
            }
            (TypeCheck::Unit, Value::Unit) => Ok(Some(BorrowRef::from_static(&[]))),
            _ => Ok(None),
        }
    }

    /// Perform a function call to determine if `any` matches the variant given
    /// by `index`.
    fn call_is_variant(
        &mut self,
        any: &Shared<AnyObj>,
        index: usize,
        output: Address,
    ) -> Result<CallResult<bool>, VmError> {
        let value = Value::Any(any.clone());

        if let CallResult::Unsupported = self
            .context
            .call_instance_fn(
                &mut self.stack,
                &self.unit,
                value,
                Protocol::IS_VARIANT,
                (index,),
                output,
            )?
            .or_call_offset_with(self)?
        {
            Ok(CallResult::Unsupported)
        } else {
            Ok(CallResult::Ok(self.stack.take_at(output)?.into_bool()?))
        }
    }

    /// Call [Protocol::NEXT] on the given value.
    fn call_next(
        &mut self,
        any: &Shared<AnyObj>,
        output: Address,
    ) -> Result<CallResult<()>, VmError> {
        let value = Value::Any(any.clone());

        if let CallResult::Unsupported = self
            .context
            .call_instance_fn(
                &mut self.stack,
                &self.unit,
                value,
                Protocol::NEXT,
                (),
                output,
            )?
            .or_call_offset_with(self)?
        {
            Ok(CallResult::Unsupported)
        } else {
            Ok(CallResult::Ok(()))
        }
    }
}
