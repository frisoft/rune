use crate::runtime::{Address, Stack, ToValue, Value, VmError};

/// Trait for converting arguments onto the stack.
pub trait Args {
    /// Encode arguments onto a stack at the given address.
    fn into_stack(self, at: Address, stack: &mut Stack) -> Result<Address, VmError>;

    /// Convert arguments into a vector.
    fn into_vec(self) -> Result<Vec<Value>, VmError>;

    /// The number of arguments.
    fn count(&self) -> usize;
}

macro_rules! impl_into_args {
    () => {
        impl_into_args!{@impl 0,}
    };

    ({$ty:ident, $value:ident, $count:expr}, $({$l_ty:ident, $l_value:ident, $l_count:expr},)*) => {
        impl_into_args!{@impl $count, {$ty, $value, $count}, $({$l_ty, $l_value, $l_count},)*}
        impl_into_args!{$({$l_ty, $l_value, $l_count},)*}
    };

    (@impl $count:expr, $({$ty:ident, $value:ident, $ignore_count:expr},)*) => {
        impl<$($ty,)*> Args for ($($ty,)*)
        where
            $($ty: ToValue,)*
        {
            #[allow(unused)]
            fn into_stack(self, mut at: Address, stack: &mut Stack) -> Result<Address, VmError> {
                let ($($value,)*) = self;

                $(
                    stack.store(at, $value.to_value()?);
                    at = at.step()?;
                )*

                Ok(at)
            }

            #[allow(unused)]
            fn into_vec(self) -> Result<Vec<Value>, VmError> {
                let ($($value,)*) = self;
                $(let $value = <$ty>::to_value($value)?;)*
                Ok(vec![$($value,)*])
            }

            fn count(&self) -> usize {
                $count
            }
        }
    };
}

repeat_macro!(impl_into_args);

impl Args for Vec<Value> {
    fn into_stack(self, mut at: Address, stack: &mut Stack) -> Result<Address, VmError> {
        for value in self {
            stack.store(at, value)?;
            at = at.step()?;
        }

        Ok(at)
    }

    fn into_vec(self) -> Result<Vec<Value>, VmError> {
        Ok(self)
    }

    fn count(&self) -> usize {
        self.len()
    }
}
