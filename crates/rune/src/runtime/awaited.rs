use crate::runtime::{Address, Future, Select, Shared, ToValue, Vm, VmError};

/// A stored await task.
#[derive(Debug)]
pub(crate) enum Awaited {
    /// A future to be awaited.
    Future(Shared<Future>, Address),
    /// A select to be awaited.
    Select(Select, Address, Address),
}

impl Awaited {
    /// Wait for the given awaited into the specified virtual machine.
    pub(crate) async fn into_vm(self, vm: &mut Vm) -> Result<(), VmError> {
        match self {
            Self::Future(future, output) => {
                let value = future.borrow_mut()?.await?;
                vm.stack_mut().store(output, value)?;
                vm.advance();
            }
            Self::Select(select, output, branch_output) => {
                let (branch, value) = select.await?;
                let mut pair = vm.stack_mut().interleaved_pair(output, branch_output)?;
                *pair.first_mut() = value;
                *pair.second_mut() = ToValue::to_value(branch)?;
                vm.advance();
            }
        }

        Ok(())
    }
}
