//! The `std::io` module.

use crate::runtime::{Address, Panic, Protocol, Stack, VmError};
use crate::{ContextError, Module};
use std::fmt::{self, Write as _};
use std::io::{self, Write as _};

/// Construct the `std::io` module.
pub fn module(stdio: bool) -> Result<Module, ContextError> {
    let mut module = Module::with_crate_item("std", &["io"]);

    module.ty::<io::Error>()?;
    module.inst_fn(Protocol::STRING_DISPLAY, format_io_error)?;

    if stdio {
        module.function(&["print"], print_impl)?;
        module.function(&["println"], println_impl)?;
        module.raw_fn(&["dbg"], dbg_impl)?;
    }

    Ok(module)
}

fn format_io_error(error: &std::io::Error, buf: &mut String) -> fmt::Result {
    write!(buf, "{}", error)
}

fn dbg_impl(
    stack: &mut Stack,
    address: Address,
    args: usize,
    output: Address,
) -> Result<(), VmError> {
    let stdout = io::stdout();
    let mut stdout = stdout.lock();

    for value in stack.drain_at(address, args)? {
        writeln!(stdout, "{:?}", value).map_err(VmError::panic)?;
    }

    stack.store(output, ())?;
    Ok(())
}

fn print_impl(m: &str) -> Result<(), Panic> {
    let stdout = io::stdout();
    let mut stdout = stdout.lock();
    write!(stdout, "{}", m).map_err(Panic::custom)
}

fn println_impl(m: &str) -> Result<(), Panic> {
    let stdout = io::stdout();
    let mut stdout = stdout.lock();
    writeln!(stdout, "{}", m).map_err(Panic::custom)
}
