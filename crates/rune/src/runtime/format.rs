//! Types for dealing with formatting specifications.

use core::fmt;
use core::fmt::Write;
use core::iter;
use core::mem::take;
use core::num::NonZeroUsize;
use core::str;

use crate::no_std::prelude::*;

use musli::{Decode, Encode};
use serde::{Deserialize, Serialize};

use crate::compile::Named;
use crate::module::InstallWith;
use crate::runtime::{FromValue, ProtocolCaller, RawStr, Value, VmErrorKind, VmResult};

/// Error raised when trying to parse a type string and it fails.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct TypeFromStrError;

impl fmt::Display for TypeFromStrError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Bad type string")
    }
}

/// Error raised when trying to parse an alignment string and it fails.
#[derive(Debug, Clone, Copy)]
pub struct AlignmentFromStrError;

impl fmt::Display for AlignmentFromStrError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Bad alignment string")
    }
}

/// A format specification, wrapping an inner value.
#[derive(Debug, Clone)]
pub struct Format {
    /// The value being formatted.
    pub(crate) value: Value,
    /// The specification.
    pub(crate) spec: FormatSpec,
}

impl Named for Format {
    const BASE_NAME: RawStr = RawStr::from_str("Format");
}

impl InstallWith for Format {}

impl FromValue for Format {
    #[inline]
    fn from_value(value: Value) -> VmResult<Self> {
        VmResult::Ok(*vm_try!(value.into_format()))
    }
}

/// A format specification.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Decode, Encode)]
#[non_exhaustive]
pub struct FormatSpec {
    /// Formatting flags.
    pub(crate) flags: Flags,
    /// The fill character.
    pub(crate) fill: char,
    /// The alignment specification.
    pub(crate) align: Alignment,
    /// Formatting width.
    pub(crate) width: Option<NonZeroUsize>,
    /// Formatting precision.
    pub(crate) precision: Option<NonZeroUsize>,
    /// The type specification.
    pub(crate) format_type: Type,
}

impl FormatSpec {
    /// Construct a new format specification.
    pub fn new(
        flags: Flags,
        fill: char,
        align: Alignment,
        width: Option<NonZeroUsize>,
        precision: Option<NonZeroUsize>,
        format_type: Type,
    ) -> Self {
        Self {
            flags,
            fill,
            align,
            width,
            precision,
            format_type,
        }
    }

    /// get traits out of a floating point number.
    fn float_traits(&self, n: f64) -> (f64, Alignment, char, Option<char>) {
        if self.flags.test(Flag::SignAwareZeroPad) {
            if n.is_sign_negative() {
                (-n, Alignment::Right, '0', Some('-'))
            } else {
                (n, Alignment::Right, '0', None)
            }
        } else if self.flags.test(Flag::SignPlus) && n.is_sign_positive() {
            (n, self.align, self.fill, Some('+'))
        } else {
            (n, self.align, self.fill, None)
        }
    }

    /// get traits out of an integer.
    fn int_traits(&self, n: i64) -> (i64, Alignment, char, Option<char>) {
        if self.flags.test(Flag::SignAwareZeroPad) {
            if n < 0 {
                (-n, Alignment::Right, '0', Some('-'))
            } else {
                (n, Alignment::Right, '0', None)
            }
        } else if self.flags.test(Flag::SignPlus) && n >= 0 {
            (n, self.align, self.fill, Some('+'))
        } else {
            (n, self.align, self.fill, None)
        }
    }

    /// Format the given number.
    fn format_number(&self, buf: &mut String, n: i64) {
        let mut buffer = itoa::Buffer::new();
        buf.push_str(buffer.format(n));
    }

    /// Format the given float.
    fn format_float(&self, buf: &mut String, n: f64) -> Result<(), VmErrorKind> {
        if let Some(precision) = self.precision {
            write!(buf, "{:.*}", precision.get(), n).map_err(|_| VmErrorKind::FormatError)?;
        } else {
            let mut buffer = ryu::Buffer::new();
            buf.push_str(buffer.format(n));
        }

        Ok(())
    }

    /// Format fill.
    fn format_fill(
        &self,
        out: &mut String,
        buf: &str,
        align: Alignment,
        fill: char,
        sign: Option<char>,
    ) {
        if let Some(sign) = sign {
            out.push(sign);
        }

        let mut w = self.width.map(|n| n.get()).unwrap_or_default();

        if w == 0 {
            out.push_str(buf);
            return;
        }

        w = w
            .saturating_sub(buf.chars().count())
            .saturating_sub(sign.map(|_| 1).unwrap_or_default());

        if w == 0 {
            out.push_str(buf);
            return;
        }

        let mut filler = iter::repeat(fill).take(w);

        match align {
            Alignment::Left => {
                out.push_str(buf);
                out.extend(filler);
            }
            Alignment::Center => {
                out.extend((&mut filler).take(w / 2));
                out.push_str(buf);
                out.extend(filler);
            }
            Alignment::Right => {
                out.extend(filler);
                out.push_str(buf);
            }
        }
    }

    fn format_display(
        &self,
        value: &Value,
        out: &mut String,
        buf: &mut String,
        caller: impl ProtocolCaller,
    ) -> VmResult<()> {
        match value {
            Value::Char(c) => {
                buf.push(*c);
                self.format_fill(out, buf, self.align, self.fill, None);
            }
            Value::String(s) => {
                buf.push_str(&vm_try!(s.borrow_ref()));
                self.format_fill(out, buf, self.align, self.fill, None);
            }
            Value::StaticString(s) => {
                buf.push_str(s.as_ref());
                self.format_fill(out, buf, self.align, self.fill, None);
            }
            Value::Integer(n) => {
                let (n, align, fill, sign) = self.int_traits(*n);
                self.format_number(buf, n);
                self.format_fill(out, buf, align, fill, sign);
            }
            Value::Float(n) => {
                let (n, align, fill, sign) = self.float_traits(*n);
                vm_try!(self.format_float(buf, n));
                self.format_fill(out, buf, align, fill, sign);
            }
            _ => {
                let result = vm_try!(value.string_display_with(out, buf, caller));
                vm_try!(result.map_err(|_| VmErrorKind::FormatError));
            }
        }

        VmResult::Ok(())
    }

    fn format_debug(
        &self,
        value: &Value,
        out: &mut String,
        buf: &mut String,
        caller: impl ProtocolCaller,
    ) -> VmResult<()> {
        match value {
            Value::String(s) => {
                let s = vm_try!(s.borrow_ref());
                vm_try!(write!(out, "{:?}", &*s).map_err(|_| VmErrorKind::FormatError));
            }
            Value::StaticString(s) => {
                vm_try!(write!(out, "{:?}", s.as_ref()).map_err(|_| VmErrorKind::FormatError));
            }
            Value::Integer(n) => {
                let (n, align, fill, sign) = self.int_traits(*n);
                self.format_number(buf, n);
                self.format_fill(out, buf, align, fill, sign);
            }
            Value::Float(n) => {
                let (n, align, fill, sign) = self.float_traits(*n);
                vm_try!(self.format_float(buf, n));
                self.format_fill(out, buf, align, fill, sign);
            }
            value => {
                let result = vm_try!(value.string_debug_with(out, caller));
                vm_try!(result.map_err(|_| VmErrorKind::FormatError));
            }
        }

        VmResult::Ok(())
    }

    fn format_upper_hex(
        &self,
        value: &Value,
        out: &mut String,
        buf: &mut String,
    ) -> Result<(), VmErrorKind> {
        match value {
            Value::Integer(n) => {
                let (n, align, fill, sign) = self.int_traits(*n);
                write!(buf, "{:X}", n).map_err(|_| VmErrorKind::FormatError)?;
                self.format_fill(out, buf, align, fill, sign);
            }
            _ => {
                return Err(VmErrorKind::FormatError);
            }
        }

        Ok(())
    }

    fn format_lower_hex(
        &self,
        value: &Value,
        out: &mut String,
        buf: &mut String,
    ) -> Result<(), VmErrorKind> {
        match value {
            Value::Integer(n) => {
                let (n, align, fill, sign) = self.int_traits(*n);
                write!(buf, "{:x}", n).map_err(|_| VmErrorKind::FormatError)?;
                self.format_fill(out, buf, align, fill, sign);
            }
            _ => {
                return Err(VmErrorKind::FormatError);
            }
        }

        Ok(())
    }

    fn format_binary(
        &self,
        value: &Value,
        out: &mut String,
        buf: &mut String,
    ) -> Result<(), VmErrorKind> {
        match value {
            Value::Integer(n) => {
                let (n, align, fill, sign) = self.int_traits(*n);
                write!(buf, "{:b}", n).map_err(|_| VmErrorKind::FormatError)?;
                self.format_fill(out, buf, align, fill, sign);
            }
            _ => {
                return Err(VmErrorKind::FormatError);
            }
        }

        Ok(())
    }

    fn format_pointer(
        &self,
        value: &Value,
        out: &mut String,
        buf: &mut String,
    ) -> Result<(), VmErrorKind> {
        match value {
            Value::Integer(n) => {
                let (n, align, fill, sign) = self.int_traits(*n);
                write!(buf, "{:p}", n as *const ()).map_err(|_| VmErrorKind::FormatError)?;
                self.format_fill(out, buf, align, fill, sign);
            }
            _ => {
                return Err(VmErrorKind::FormatError);
            }
        }

        Ok(())
    }

    /// Format the given value to the out buffer `out`, using `buf` for
    /// intermediate work if necessary.
    pub(crate) fn format(
        &self,
        value: &Value,
        out: &mut String,
        buf: &mut String,
        caller: impl ProtocolCaller,
    ) -> VmResult<()> {
        match self.format_type {
            Type::Display => vm_try!(self.format_display(value, out, buf, caller)),
            Type::Debug => vm_try!(self.format_debug(value, out, buf, caller)),
            Type::UpperHex => vm_try!(self.format_upper_hex(value, out, buf)),
            Type::LowerHex => vm_try!(self.format_lower_hex(value, out, buf)),
            Type::Binary => vm_try!(self.format_binary(value, out, buf)),
            Type::Pointer => vm_try!(self.format_pointer(value, out, buf)),
        }

        VmResult::Ok(())
    }
}

impl fmt::Display for FormatSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "format(fill = {fill:?}, align = {align}, flags = {flags:?}, width = {width}, precision = {precision}, format_type = {format_type})",
            fill = self.fill,
            align = self.align,
            flags = self.flags,
            width = OptionDebug(self.width.as_ref()),
            precision = OptionDebug(self.precision.as_ref()),
            format_type = self.format_type
        )
    }
}

struct OptionDebug<'a, T>(Option<&'a T>);

impl<T> fmt::Display for OptionDebug<'_, T>
where
    T: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            Some(value) => write!(f, "{}", value),
            None => write!(f, "?"),
        }
    }
}

/// The type of formatting requested.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Decode, Encode)]
#[non_exhaustive]
pub enum Type {
    /// Display type (default).
    Display,
    /// Debug type.
    Debug,
    /// Upper hex type.
    UpperHex,
    /// Upper hex type.
    LowerHex,
    /// Binary formatting type.
    Binary,
    /// Pointer formatting type.
    Pointer,
}

impl str::FromStr for Type {
    type Err = TypeFromStrError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "display" => Ok(Self::Display),
            "debug" => Ok(Self::Debug),
            "upper_hex" => Ok(Self::UpperHex),
            "lower_hex" => Ok(Self::LowerHex),
            "binary" => Ok(Self::Binary),
            "pointer" => Ok(Self::Pointer),
            _ => Err(TypeFromStrError),
        }
    }
}

impl Default for Type {
    fn default() -> Self {
        Self::Display
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Display => {
                write!(f, "display")?;
            }
            Self::Debug => {
                write!(f, "debug")?;
            }
            Self::UpperHex => {
                write!(f, "upper_hex")?;
            }
            Self::LowerHex => {
                write!(f, "lower_hex")?;
            }
            Self::Binary => {
                write!(f, "binary")?;
            }
            Self::Pointer => {
                write!(f, "pointer")?;
            }
        }

        Ok(())
    }
}

/// The alignment requested.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Decode, Encode)]
#[non_exhaustive]
pub enum Alignment {
    /// Left alignment.
    Left,
    /// Center alignment.
    Center,
    /// Right alignment.
    Right,
}

impl Default for Alignment {
    fn default() -> Self {
        Self::Left
    }
}

impl str::FromStr for Alignment {
    type Err = AlignmentFromStrError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "left" => Ok(Self::Left),
            "center" => Ok(Self::Center),
            "right" => Ok(Self::Right),
            _ => Err(AlignmentFromStrError),
        }
    }
}

impl fmt::Display for Alignment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Left => {
                write!(f, "left")?;
            }
            Self::Center => {
                write!(f, "center")?;
            }
            Self::Right => {
                write!(f, "right")?;
            }
        }

        Ok(())
    }
}

/// A single flag for format spec.
#[derive(Clone, Copy)]
#[repr(u32)]
#[non_exhaustive]
pub enum Flag {
    /// Plus sign `+`.
    SignPlus,
    /// Minus sign `-`.
    SignMinus,
    /// Alternate specifier `#`.
    Alternate,
    /// Sign-aware zero pad `0`.
    SignAwareZeroPad,
}

/// Format specification flags.
#[derive(Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, Decode, Encode)]
#[repr(transparent)]
#[musli(transparent)]
pub struct Flags(u32);

impl Flags {
    /// Check if the set of flags is empty.
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Get the flags as a number. This representation is not guaranteed to be
    /// stable.
    pub fn into_u32(self) -> u32 {
        self.0
    }

    /// Set the given flag.
    #[inline]
    pub fn set(&mut self, flag: Flag) {
        self.0 |= &(1 << flag as u32);
    }

    /// Test the given flag.
    #[inline]
    pub fn test(&self, flag: Flag) -> bool {
        (self.0 & (1 << flag as u32)) != 0
    }
}

impl From<u32> for Flags {
    fn from(flags: u32) -> Self {
        Self(flags)
    }
}

impl fmt::Debug for Flags {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        macro_rules! fmt_flag {
            ($flag:ident, $o:ident, $spec:literal) => {
                if self.test(Flag::$flag) {
                    if !take(&mut $o) {
                        write!(f, ", ")?;
                    }

                    write!(f, $spec)?;
                }
            };
        }

        let mut o = true;
        write!(f, "Flags{{")?;
        fmt_flag!(SignPlus, o, "+");
        fmt_flag!(SignMinus, o, "-");
        fmt_flag!(Alternate, o, "#");
        fmt_flag!(SignAwareZeroPad, o, "0");
        write!(f, "}}")?;
        Ok(())
    }
}
