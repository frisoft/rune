// TODO: remove this
#![allow(unused)]

use std::fmt;

use rune_macros::Instruction;
use serde::{Deserialize, Serialize};

use crate::runtime::stack::StackError;
use crate::runtime::{FormatSpec, Value};
use crate::Hash;

/// Pre-canned panic reasons.
///
/// To formulate a custom reason, use [crate::runtime::Panic::custom].
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PanicReason {
    /// Not implemented.
    NotImplemented,
    /// A pattern didn't match where it unconditionally has to.
    UnmatchedPattern,
    /// Tried to poll a future that has already been completed.
    FutureCompleted,
}

impl PanicReason {
    /// The identifier of the panic.
    fn ident(&self) -> &'static str {
        match *self {
            Self::NotImplemented => "not implemented",
            Self::UnmatchedPattern => "unmatched pattern",
            Self::FutureCompleted => "future completed",
        }
    }
}

impl fmt::Display for PanicReason {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::NotImplemented => write!(fmt, "functionality has not been implemented yet")?,
            Self::UnmatchedPattern => write!(fmt, "pattern did not match")?,
            Self::FutureCompleted => {
                write!(fmt, "tried to poll future that has already been completed")?
            }
        }

        Ok(())
    }
}

/// Type checks for built-in types.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[non_exhaustive]
pub enum TypeCheck {
    /// Matches a unit type.
    Unit,
    /// Matches an anonymous tuple.
    Tuple,
    /// Matches an anonymous object.
    Object,
    /// Matches a vector.
    Vec,
    /// An option type, and the specified variant index.
    Option(usize),
    /// A result type, and the specified variant index.
    Result(usize),
    /// A generator state type, and the specified variant index.
    GeneratorState(usize),
}

/// An operation in the stack-based virtual machine.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Instruction)]
#[rune(module = "crate")]
pub enum Inst {
    /// Not operator. Takes a boolean from the given address  and inverts its
    /// logical value.
    Not {
        /// The boolean to invert.
        #[rune(address)]
        address: Address,
        /// Where to store the inverted boolean.
        #[rune(address)]
        output: Address,
    },
    /// Negate the numerical value on the stack.
    Neg {
        /// The value to negate.
        #[rune(address)]
        address: Address,
        /// Where to store the negated value.
        #[rune(address)]
        output: Address,
    },
    /// Construct a closure that takes the given number of arguments and
    /// captures `count` elements starting at `address`.
    Closure {
        /// The hash of the internally stored closure function.
        hash: Hash,
        /// Base address to allocate environment from.
        #[rune(address)]
        address: Address,
        /// The number of arguments to store in the environment on the stack.
        count: usize,
        /// Where to store the loaded closure.
        #[rune(address)]
        output: Address,
    },
    /// Perform a function call.
    ///
    /// It will construct a new stack frame which includes the last `args`
    /// number of entries.
    Call {
        /// The hash of the function to call.
        hash: Hash,
        /// The base where the call arguments are located.
        #[rune(address)]
        address: Address,
        /// The number of arguments expected on the stack for this call.
        count: usize,
        /// Where to store the return value of the call.
        #[rune(address)]
        output: Address,
    },
    /// Perform a instance function call.
    ///
    /// The instance being called on should be on top of the stack, followed by
    /// `args` number of arguments.
    CallInstance {
        /// The hash of the name of the function to call.
        hash: Hash,
        /// The base where the call arguments are located.
        #[rune(address)]
        address: Address,
        /// The number of arguments expected on the stack for this call.
        count: usize,
        /// Where to store the return value when calling the instance function.
        #[rune(address)]
        output: Address,
    },
    /// Perform a function call on a function pointer stored on the stack.
    CallFn {
        /// Address from where to load the function.
        #[rune(address)]
        function: Address,
        /// Address to value which describes the function to call.
        #[rune(address)]
        address: Address,
        /// The number of arguments expected on the stack for this call.
        count: usize,
        /// Where to write the return value of the function.
        #[rune(address)]
        output: Address,
    },
    /// Lookup the specified instance function and put it on the stack. This
    /// might help in cases where a single instance function is called many
    /// times (like in a loop) since it avoids calculating its full hash on
    /// every iteration.
    ///
    /// Note that this does not resolve that the instance function exists, only
    /// that the instance does.
    LoadInstanceFn {
        /// Address where the instance is located.
        #[rune(address)]
        address: Address,
        /// The name hash of the instance function.
        hash: Hash,
        /// Where to write the loaded instance function.
        #[rune(address)]
        output: Address,
    },
    /// Perform an index get operation. Pushing the result on the stack.
    IndexGet {
        /// Address to the target of the operation.
        #[rune(address)]
        address: Address,
        /// How the index is addressed.
        #[rune(address)]
        index: Address,
        /// The output of an index get operation.
        #[rune(address)]
        output: Address,
    },
    /// Get the given index out of a tuple on the given address.
    /// Errors if the item doesn't exist or the item is not a tuple.
    TupleIndexGet {
        /// Address to the target of the operation.
        #[rune(address)]
        address: Address,
        /// The index to fetch.
        index: usize,
        /// Output address.
        #[rune(address)]
        output: Address,
    },
    /// Set the given index of the tuple on the stack, with the given value.
    TupleIndexSet {
        /// Target of the set operation.
        #[rune(address)]
        address: Address,
        /// Address of value to set.
        #[rune(address)]
        value: Address,
        /// The index to set.
        index: usize,
        /// Where to store the output of this operation.
        #[rune(address)]
        output: Address,
    },
    /// Get the given index out of an object on the given address.
    /// Errors if the item doesn't exist or the item is not an object.
    ///
    /// The index is identifier by a static string slot, which is provided as an
    /// argument.
    ObjectIndexGet {
        /// Address of the object to index.
        #[rune(address)]
        address: Address,
        /// The static string slot corresponding to the index to fetch.
        slot: usize,
        /// Address to store the output of the operation.
        #[rune(address)]
        output: Address,
    },
    /// Set the given index out of an object on the given address.
    /// Errors if the item doesn't exist or the item is not an object.
    ///
    /// The index is identifier by a static string slot, which is provided as an
    /// argument.
    ObjectIndexSet {
        /// Address of object where to set a value.
        #[rune(address)]
        address: Address,
        /// The static string slot corresponding to the index to set.
        slot: usize,
        /// The value of the address being set.
        #[rune(address)]
        value: Address,
        /// Where to store the output of this operation.
        #[rune(address)]
        output: Address,
    },
    /// Perform an index set operation.
    IndexSet {
        /// The target that is being set.
        #[rune(address)]
        address: Address,
        /// The index on the target that is being set.
        #[rune(address)]
        index: Address,
        /// The value that is being set.
        #[rune(address)]
        value: Address,
        /// Where to store the output of this operation.
        #[rune(address)]
        output: Address,
    },
    /// Select over `len` futures on the stack. Sets the `branch` register to
    /// the index of the branch that completed. And pushes its value on the
    /// stack.
    ///
    /// This operation will block the VM until at least one of the underlying
    /// futures complete.
    Select {
        /// The base address where to select futures over.
        #[rune(address)]
        address: Address,
        /// The number of values being selected over.
        count: usize,
        /// The location to store the output of the evaluated select.
        #[rune(address)]
        output: Address,
        /// Where to store the branch address.
        #[rune(address)]
        branch_output: Address,
    },
    /// Load the given function by hash and push onto the stack.
    LoadFn {
        /// The hash of the function to push.
        hash: Hash,
        /// The location to load the function to.
        #[rune(address)]
        output: Address,
    },
    /// Store a value onto the current stack at the given location.
    Store {
        /// The value to push.
        value: InstValue,
        /// The address to store the value.
        #[rune(address)]
        output: Address,
    },
    /// Drop the value at the given stack location.
    Drop {
        /// Address of value to drop.
        #[rune(address)]
        address: Address,
    },
    /// Copy a value from one location of the stack to another.
    Copy {
        /// Address to copy the value from.
        #[rune(address)]
        address: Address,
        /// Address to copy the value to.
        #[rune(address)]
        output: Address,
    },
    /// Move a value. The value at the original location will be deinitialized.
    Move {
        /// Address to move the value from.
        #[rune(address)]
        address: Address,
        /// Address to move the value to.
        #[rune(address)]
        output: Address,
    },
    /// Pop the current stack frame and restore the instruction pointer from it.
    ///
    /// The value indicated by `address` will be returned from the current stack
    /// frame.
    Return {
        /// The address of the value to return.
        #[rune(address)]
        address: Address,
    },
    /// Pop the current stack frame and restore the instruction pointer from it.
    ///
    /// A unit will be returned from the current stack frame.
    ReturnUnit,
    /// Unconditionally jump to `offset` relative to the current instruction
    /// pointer.
    Jump {
        /// Offset to jump to.
        #[rune(label)]
        offset: isize,
    },
    /// Jump to `offset` relative to the current instruction pointer if the
    /// value stored at `address` is boolean `true`.
    JumpIf {
        /// Address of the condition for the jump.
        #[rune(address)]
        address: Address,
        /// Offset to jump to.
        #[rune(label)]
        offset: isize,
    },
    /// Jump to `offset` relative to the current instruction pointer if the
    /// value stored at `address` is boolean `false`.
    JumpIfNot {
        /// Address of the condition for the jump.
        #[rune(address)]
        address: Address,
        /// Offset to jump to.
        #[rune(label)]
        offset: isize,
    },
    /// Compares the `branch` register with the given address, and if they
    /// match pops the given address and performs the jump to offset.
    JumpIfBranch {
        /// The address to compare the value with.
        #[rune(address)]
        address: Address,
        /// The branch value to compare against.
        branch: i64,
        /// The offset to jump.
        #[rune(label)]
        offset: isize,
    },
    /// Construct a vector by popping `count` elements off the stack. Writes the
    /// allocated vector to `output`.
    Vec {
        /// The base address where the vector is allocated.
        #[rune(address)]
        address: Address,
        /// The size of the vector.
        count: usize,
        /// The address where we write the newly allocated vector.
        #[rune(address)]
        output: Address,
    },
    /// Construct a push a one-tuple value onto the stack.
    Tuple1 {
        /// First element of the tuple.
        #[rune(address, debug)]
        args: [Address; 1],
        /// where to write the allocated tuple.
        #[rune(address)]
        output: Address,
    },
    /// Construct a push a two-tuple value onto the stack.
    Tuple2 {
        /// Tuple arguments.
        #[rune(address, debug)]
        args: [Address; 2],
        /// where to write the allocated tuple.
        #[rune(address)]
        output: Address,
    },
    /// Construct a push a three-tuple value onto the stack.
    Tuple3 {
        /// Tuple arguments.
        #[rune(address, debug)]
        args: [Address; 3],
        /// where to write the allocated tuple.
        #[rune(address)]
        output: Address,
    },
    /// Construct a push a four-tuple value onto the stack.
    Tuple4 {
        /// Tuple arguments.
        #[rune(address, debug)]
        args: [Address; 4],
        /// where to write the allocated tuple.
        #[rune(address)]
        output: Address,
    },
    /// Construct a tuple by popping `count` elements off the stack. Writes the
    /// tuple to the `output` address.
    Tuple {
        /// The base address where the tuple is allocated.
        #[rune(address)]
        address: Address,
        /// The size of the tuple.
        count: usize,
        /// where to write the allocated tuple.
        #[rune(address)]
        output: Address,
    },
    /// Take the tuple that is on top of the stack and unpack its content onto the
    /// stack starting at `output`.
    UnpackTuple {
        /// Address of the tuple to unpack.
        #[rune(address)]
        address: Address,
        /// The output address to push the tuple onto.
        #[rune(address)]
        output: Address,
    },
    /// Construct a push an object onto the stack. The number of elements
    /// in the object are determined the slot of the object keys `slot` and are
    /// popped from the stack.
    ///
    /// For each element, a value is popped corresponding to the object key.
    Object {
        /// The base address where object values are allocated from.
        #[rune(address)]
        address: Address,
        /// The static slot of the object keys.
        slot: usize,
        /// The address to write the newly allocated object.
        #[rune(address)]
        output: Address,
    },
    /// Construct a range. This will pop the start and end of the range from the
    /// stack.
    Range {
        /// Where to load the from value from.
        #[rune(address)]
        from: Address,
        /// Where to load the to value from.
        #[rune(address)]
        to: Address,
        /// The limits of the range.
        limits: InstRangeLimits,
        /// Where to write the range object.
        #[rune(address)]
        output: Address,
    },
    /// Construct a push an object of the given type onto the stack. The type is
    /// an empty struct.
    UnitStruct {
        /// The type of the object to construct.
        hash: Hash,
        /// Where to write the unit struct.
        #[rune(address)]
        output: Address,
    },
    /// Construct a push an object of the given type onto the stack. The number
    /// of elements in the object are determined the slot of the object keys
    /// `slot` and are popped from the stack.
    ///
    /// For each element, a value is popped corresponding to the object key.
    Struct {
        /// The type of the object to construct.
        hash: Hash,
        /// Base address where to allocate struct fields from.
        #[rune(address)]
        address: Address,
        /// The static slot of the object keys.
        slot: usize,
        /// Where to write the allocated struct.
        #[rune(address)]
        output: Address,
    },
    /// Construct a push an object variant of the given type onto the stack. The
    /// type is an empty struct.
    UnitVariant {
        /// The type hash of the object variant to construct.
        hash: Hash,
        /// Where to write the allocated unit variant.
        #[rune(address)]
        output: Address,
    },
    /// Construct a push an object variant of the given type onto the stack. The
    /// number of elements in the object are determined the slot of the object
    /// keys `slot` and are popped from the stack.
    ///
    /// For each element, a value is popped corresponding to the object key.
    StructVariant {
        /// The type hash of the object variant to construct.
        hash: Hash,
        /// The base address where fields are allocated from.
        #[rune(address)]
        address: Address,
        /// The static slot of the object keys.
        slot: usize,
        /// Where to write the allocated struct variant.
        #[rune(address)]
        output: Address,
    },
    /// Load a literal string from a static string slot.
    String {
        /// The static string slot to load the string from.
        slot: usize,
        /// Where to write the string.
        #[rune(address)]
        output: Address,
    },
    /// Load a literal byte string from a static byte string slot.
    Bytes {
        /// The static byte string slot to load the string from.
        slot: usize,
        /// Where to write the byte buffer.
        #[rune(address)]
        output: Address,
    },
    /// Read the given number of values from the stack starting at `address`,
    /// and concatenate a string from them.
    ///
    /// This is a dedicated template-string optimization.
    StringConcat {
        /// Base address to concatenate string from.
        #[rune(address)]
        address: Address,
        /// The number of string components to fetch from the stack.
        count: usize,
        /// The minimum string size used.
        size_hint: usize,
        /// Where to store the result of the string concatenation.
        #[rune(address)]
        output: Address,
    },
    /// Write a combined format specification and value onto the stack. The value
    /// used is the last value on the stack.
    Format {
        /// Address of the value to format.
        #[rune(address)]
        address: Address,
        /// The format specification to use.
        #[rune(debug)]
        spec: FormatSpec,
        /// Where to write the newly allocated format specification.
        #[rune(address)]
        output: Address,
    },
    /// Await the future that is on the stack and push the value that it
    /// produces.
    Await {
        /// The address of the value to await.
        #[rune(address)]
        address: Address,
        /// The stack address to store the result of the await.
        #[rune(address)]
        output: Address,
    },
    /// Perform the try operation which takes the value at the given `address`
    /// and tries to unwrap it or return from the current call frame.
    Try {
        /// Address to test if value.
        #[rune(address)]
        address: Address,
        /// Address to write the output of the operation if it doesn't return.
        #[rune(address)]
        output: Address,
    },
    /// Test if the given address is a unit.
    MatchValue {
        /// Address to test.
        #[rune(address)]
        address: Address,
        /// Literal value to match.
        value: InstValue,
        /// Where to jump if the specified address does not match a unit.
        #[rune(label)]
        offset: isize,
    },
    /// Compare the given address against a static string slot.
    MatchString {
        /// Address to test.
        #[rune(address)]
        address: Address,
        /// The slot to test against.
        slot: usize,
        /// Where to jump if the string does not match.
        #[rune(label)]
        offset: isize,
    },
    /// Compare the given address against a static bytes slot.
    MatchBytes {
        /// Address to test.
        #[rune(address)]
        address: Address,
        /// The slot to test against.
        slot: usize,
        /// Where to jump if the bytes does not match.
        #[rune(label)]
        offset: isize,
    },
    /// Test that the given address has the given type.
    MatchType {
        /// Address to test.
        #[rune(address)]
        address: Address,
        /// The type hash to match against.
        type_hash: Hash,
        /// Where to jump if the type doesn't match.
        #[rune(label)]
        offset: isize,
    },
    /// Test if the specified variant matches. This is distinct from
    /// [Inst::MatchType] because it will match immediately on the variant type
    /// if appropriate which is possible for internal types, but external types
    /// will require an additional runtime check for matching.
    MatchVariant {
        /// Address to test.
        #[rune(address)]
        address: Address,
        /// The exact type hash of the variant.
        variant_hash: Hash,
        /// The container type.
        enum_hash: Hash,
        /// The index of the variant.
        index: usize,
        /// Where to jump on a mismatch.
        #[rune(label)]
        offset: isize,
        /// Output of any operation needed to test if the variant is a variant.
        #[rune(address)]
        output: Address,
    },
    /// Test if the given address is the given builtin type or variant.
    MatchBuiltIn {
        /// Address to test.
        #[rune(address)]
        address: Address,
        /// The type to check for.
        #[rune(debug)]
        type_check: TypeCheck,
        /// Where to jump on a mismatch.
        #[rune(label)]
        offset: isize,
    },
    /// Test that the given address is a tuple with the given length
    /// requirements.
    MatchSequence {
        /// Address to test.
        #[rune(address)]
        address: Address,
        /// Type constraints that the sequence must match.
        #[rune(debug)]
        type_check: TypeCheck,
        /// The minimum length to test for.
        len: usize,
        /// Whether the operation should check exact `true` or minimum length
        /// `false`.
        exact: bool,
        /// Where to jump in case the sequence does not match.
        #[rune(label)]
        offset: isize,
        /// The address where the sequence is unpacked if it matches.
        #[rune(address)]
        output: Address,
    },
    /// Test that the given address is an object matching the given slot of
    /// object keys.
    MatchObject {
        /// Address to test.
        #[rune(address)]
        address: Address,
        /// The slot of object keys to use.
        slot: usize,
        /// Whether the operation should check exact `true` or minimum length
        /// `false`.
        exact: bool,
        /// Where to jump in case the sequence does not match.
        #[rune(label)]
        offset: isize,
        /// Where to write the output of the operation.
        #[rune(address)]
        output: Address,
    },
    /// Perform a generator yield where the value yielded is expected to be
    /// found at the given address.
    ///
    /// This causes the virtual machine to suspend itself.
    Yield {
        /// Address of the value to yield.
        #[rune(address)]
        address: Address,
        /// Address where to store the output of the yield.
        #[rune(address)]
        output: Address,
    },
    /// Perform a generator yield which produces a unit.
    ///
    /// This causes the virtual machine to suspend itself.
    YieldUnit {
        /// Address where to store the output of the yield.
        #[rune(address)]
        output: Address,
    },
    /// Construct a built-in variant onto the stack.
    ///
    /// The variant will pop as many values of the stack as necessary to
    /// construct it.
    Variant {
        /// Base address where values that the variant is constructed from is loaded.
        #[rune(address)]
        address: Address,
        /// The kind of built-in variant to construct.
        variant: InstVariant,
        /// Where to write the newly allocated variant.
        #[rune(address)]
        output: Address,
    },
    /// A built-in operation like `a + b` that takes its operands and pushes its
    /// result to and from the stack.
    Op {
        /// The actual operation.
        op: InstOp,
        /// The address of the first argument.
        #[rune(address)]
        a: Address,
        /// The address of the second argument.
        #[rune(address)]
        b: Address,
        /// The output of the operation.
        #[rune(address)]
        output: Address,
    },
    /// A built-in operation that assigns to the left-hand side operand. Like `a
    /// += b`.
    ///
    /// The target determines the left hand side operation.
    Assign {
        /// Left hand side operand.
        #[rune(address)]
        lhs: Address,
        /// Right hand side operand.
        #[rune(address)]
        rhs: Address,
        /// The target of the operation.
        target: InstTarget,
        /// The actual operation.
        op: InstAssignOp,
        /// Where to store the output of the operation.
        #[rune(address)]
        output: Address,
    },
    /// Advance an iterator at the given position.
    IterNext {
        /// The address of the value being advanced.
        #[rune(address)]
        address: Address,
        /// A relative jump to perform if the iterator could not be advanced.
        #[rune(label)]
        offset: isize,
        /// Where to store the output of the advanced iterator.
        #[rune(address)]
        output: Address,
    },
    /// Cause the VM to panic and error out without a reason.
    ///
    /// This should only be used during testing or extreme scenarios that are
    /// completely unrecoverable.
    Panic {
        /// The reason for the panic.
        reason: PanicReason,
    },
}

/// The address of a value on the stack. Address 0 is the return value of the
/// current stack frame.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Address(pub(crate) u32);

impl Address {
    /// The base address of the stack.
    pub const BASE: Self = Self(0);

    /// Step to the next stack address.
    pub(crate) fn step(self) -> Result<Self, StackError> {
        Ok(Self(self.0.checked_add(1).ok_or(StackError::new())?))
    }
}

impl fmt::Display for Address {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Range limits of a range expression.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum InstRangeLimits {
    /// A half-open range `a .. b`.
    HalfOpen,
    /// A closed range `a ..= b`.
    Closed,
}

impl fmt::Display for InstRangeLimits {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HalfOpen => write!(f, ".."),
            Self::Closed => write!(f, "..="),
        }
    }
}

/// The target of an operation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum InstTarget {
    /// Target is an offset to the current call frame.
    Offset,
    /// Target the field of a tuple.
    TupleField(usize),
    /// Target the field of an object.
    ObjectField(usize),
}

impl fmt::Display for InstTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Offset => write!(f, "offset"),
            Self::TupleField(slot) => write!(f, "tuple-field({})", slot),
            Self::ObjectField(slot) => write!(f, "object-field({})", slot),
        }
    }
}

/// An operation between two values on the machine.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum InstAssignOp {
    /// The add operation. `a + b`.
    Add,
    /// The sub operation. `a - b`.
    Sub,
    /// The multiply operation. `a * b`.
    Mul,
    /// The division operation. `a / b`.
    Div,
    /// The remainder operation. `a % b`.
    Rem,
    /// The bitwise and operation. `a & b`.
    BitAnd,
    /// The bitwise xor operation. `a ^ b`.
    BitXor,
    /// The bitwise or operation. `a | b`.
    BitOr,
    /// The shift left operation. `a << b`.
    Shl,
    /// The shift right operation. `a << b`.
    Shr,
}

impl fmt::Display for InstAssignOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Add => {
                write!(f, "+")?;
            }
            Self::Sub => {
                write!(f, "-")?;
            }
            Self::Mul => {
                write!(f, "*")?;
            }
            Self::Div => {
                write!(f, "/")?;
            }
            Self::Rem => {
                write!(f, "%")?;
            }
            Self::BitAnd => {
                write!(f, "&")?;
            }
            Self::BitXor => {
                write!(f, "^")?;
            }
            Self::BitOr => {
                write!(f, "|")?;
            }
            Self::Shl => {
                write!(f, "<<")?;
            }
            Self::Shr => {
                write!(f, ">>")?;
            }
        }

        Ok(())
    }
}

/// An operation between two values on the machine.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum InstOp {
    /// The add operation. `a + b`.
    Add,
    /// The sub operation. `a - b`.
    Sub,
    /// The multiply operation. `a * b`.
    Mul,
    /// The division operation. `a / b`.
    Div,
    /// The remainder operation. `a % b`.
    Rem,
    /// The bitwise and operation. `a & b`.
    BitAnd,
    /// The bitwise xor operation. `a ^ b`.
    BitXor,
    /// The bitwise or operation. `a | b`.
    BitOr,
    /// The shift left operation. `a << b`.
    Shl,
    /// The shift right operation. `a << b`.
    Shr,
    /// Compare two values on the stack for lt and push the result as a
    /// boolean on the stack.
    Lt,
    /// Compare two values on the stack for gt and push the result as a
    /// boolean on the stack.
    Gt,
    /// Compare two values on the stack for lte and push the result as a
    /// boolean on the stack.
    Lte,
    /// Compare two values on the stack for gte and push the result as a
    /// boolean on the stack.
    Gte,
    /// Compare two values on the stack for equality and push the result as a
    /// boolean on the stack.
    ///
    /// # Operation
    ///
    /// ```text
    /// <b>
    /// <a>
    /// => <bool>
    /// ```
    Eq,
    /// Compare two values on the stack for inequality and push the result as a
    /// boolean on the stack.
    ///
    /// # Operation
    ///
    /// ```text
    /// <b>
    /// <a>
    /// => <bool>
    /// ```
    Neq,
    /// Test if the given address is an instance of the second item on the
    /// stack.
    ///
    /// # Operation
    ///
    /// ```text
    /// <type>
    /// <value>
    /// => <boolean>
    /// ```
    Is,
    /// Test if the given address is not an instance of the second item on
    /// the stack.
    ///
    /// # Operation
    ///
    /// ```text
    /// <type>
    /// <value>
    /// => <boolean>
    /// ```
    IsNot,
    /// Pop two values from the stack and test if they are both boolean true.
    ///
    /// # Operation
    ///
    /// ```text
    /// <boolean>
    /// <boolean>
    /// => <boolean>
    /// ```
    And,
    /// Pop two values from the stack and test if either of them are boolean
    /// true.
    ///
    /// # Operation
    ///
    /// ```text
    /// <boolean>
    /// <boolean>
    /// => <boolean>
    /// ```
    Or,
}

impl fmt::Display for InstOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Add => {
                write!(f, "+")?;
            }
            Self::Sub => {
                write!(f, "-")?;
            }
            Self::Mul => {
                write!(f, "*")?;
            }
            Self::Div => {
                write!(f, "/")?;
            }
            Self::Rem => {
                write!(f, "%")?;
            }
            Self::BitAnd => {
                write!(f, "&")?;
            }
            Self::BitXor => {
                write!(f, "^")?;
            }
            Self::BitOr => {
                write!(f, "|")?;
            }
            Self::Shl => {
                write!(f, "<<")?;
            }
            Self::Shr => {
                write!(f, ">>")?;
            }
            Self::Lt => {
                write!(f, "<")?;
            }
            Self::Gt => {
                write!(f, ">")?;
            }
            Self::Lte => {
                write!(f, "<=")?;
            }
            Self::Gte => {
                write!(f, ">=")?;
            }
            Self::Eq => {
                write!(f, "==")?;
            }
            Self::Neq => {
                write!(f, "!=")?;
            }
            Self::Is => {
                write!(f, "is")?;
            }
            Self::IsNot => {
                write!(f, "is not")?;
            }
            Self::And => {
                write!(f, "&&")?;
            }
            Self::Or => {
                write!(f, "||")?;
            }
        }

        Ok(())
    }
}

/// A literal value that can be pushed.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum InstValue {
    /// A unit.
    Unit,
    /// A boolean.
    Bool(bool),
    /// A byte.
    Byte(u8),
    /// A character.
    Char(char),
    /// An integer.
    Integer(i64),
    /// A float.
    Float(f64),
    /// A type hash.
    Type(Hash),
}

impl InstValue {
    /// Convert into a value that can be pushed onto the stack.
    pub(crate) fn into_value(self) -> Value {
        match self {
            Self::Unit => Value::Unit,
            Self::Bool(v) => Value::Bool(v),
            Self::Byte(v) => Value::Byte(v),
            Self::Char(v) => Value::Char(v),
            Self::Integer(v) => Value::Integer(v),
            Self::Float(v) => Value::Float(v),
            Self::Type(v) => Value::Type(v),
        }
    }
}

impl fmt::Display for InstValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unit => write!(f, "()")?,
            Self::Bool(v) => write!(f, "{}", v)?,
            Self::Byte(v) => {
                if v.is_ascii_graphic() {
                    write!(f, "b'{}'", *v as char)?
                } else {
                    write!(f, "b'\\x{:02x}'", v)?
                }
            }
            Self::Char(v) => write!(f, "{:?}", v)?,
            Self::Integer(v) => write!(f, "{}", v)?,
            Self::Float(v) => write!(f, "{}", v)?,
            Self::Type(v) => write!(f, "{}", v)?,
        }

        Ok(())
    }
}

/// A variant that can be constructed.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum InstVariant {
    /// `Option::Some`, which uses one value.
    Some,
    /// `Option::None`, which uses no values.
    None,
    /// `Result::Ok`, which uses one value.
    Ok,
    /// `Result::Err`, which uses one value.
    Err,
}

impl fmt::Display for InstVariant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Some => {
                write!(f, "Some")?;
            }
            Self::None => {
                write!(f, "None")?;
            }
            Self::Ok => {
                write!(f, "Ok")?;
            }
            Self::Err => {
                write!(f, "Err")?;
            }
        }

        Ok(())
    }
}
