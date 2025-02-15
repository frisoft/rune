//! The `std::ops` module.

use crate::runtime::{Function, Protocol, Range, Value};
use crate::{ContextError, Module};

/// Construct the `std::ops` module.
pub fn module() -> Result<Module, ContextError> {
    let mut module = Module::with_crate_item("std", ["ops"]);

    module.ty::<Range>()?;

    module
        .type_meta::<Range>()?
        .make_named_struct(&["start", "end"])?
        .docs([
            "Type for a range expression.",
            "",
            "A range expression is one of:",
            "* `..<value>`",
            "* `<value>..<value>`",
            "* `<value>..`",
            "* `..=<value>`",
            "* `<value>..=<value>`",
        ]);

    module.field_function(Protocol::GET, "start", |r: &Range| r.start.clone())?;
    module.field_function(Protocol::SET, "start", range_set_start)?;

    module.field_function(Protocol::GET, "end", |r: &Range| r.end.clone())?;
    module.field_function(Protocol::SET, "end", range_set_end)?;
    module.associated_function(Protocol::INTO_ITER, Range::into_iterator)?;

    module.function_meta(Range::contains_int__meta)?;

    module
        .associated_function("iter", Range::into_iterator)?
        .docs([
            "Iterate over the range.",
            "",
            "This panics if the range is not a well-defined range.",
        ]);

    module.ty::<Function>()?.docs([
        "The type of a function in Rune.",
        "",
        "Functions can be called using call expression syntax, such as `<expr>()`.",
        "",
        "There are multiple different kind of things which can be coerced into a function in Rune:",
        "* Regular functions.",
        "* Closures (which might or might not capture their environment).",
        "* Built-in constructors for tuple types (tuple structs, tuple variants).",
        "",
        "# Examples",
        "",
        "```rune",
        "// Captures the constructor for the `Some(<value>)` tuple variant.",
        "let build_some = Some;",
        "assert_eq!(build_some(42), Some(42));",
        "",
        "fn build(value) {",
        "    Some(value)",
        "}",
        "",
        "// Captures the function previously defined.",
        "let build_some = build;",
        "assert_eq!(build_some(42), Some(42));",
        "```",
    ]);
    Ok(module)
}

fn range_set_start(range: &mut Range, start: Option<Value>) {
    range.start = start;
}

fn range_set_end(range: &mut Range, end: Option<Value>) {
    range.end = end;
}
