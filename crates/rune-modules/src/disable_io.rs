//! I/O module ignoring everything written to output.
//!
//! ```
//! use rune::{Context, ContextError};
//! use rune_modules::disable_io;
//!
//! # fn main() -> Result<(), ContextError> {
//! let mut c = rune_modules::with_config(false)?;
//! c.install(&disable_io::module()?)?;
//! # Ok(()) }
//! ```

use rune::runtime::{Address, Arguments, Stack};
use rune::{ContextError, Module};

/// Provide a bunch of `std::io` functions which will cause any output to be ignored.
pub fn module() -> Result<Module, ContextError> {
    let mut module = Module::with_crate_item("std", &["io"]);

    module.function(&["print"], move |_: &str| {})?;

    module.function(&["println"], move |_: &str| {})?;

    module.raw_fn(
        &["dbg"],
        move |stack: &mut Stack, arguments: &mut dyn Arguments, output: Address| {
            while let Some(_) = arguments.try_take_next(stack)? {}
            stack.store(output, ())?;
            Ok(())
        },
    )?;

    Ok(module)
}
