use rune::runtime::VecTuple;
use rune::termcolor::{ColorChoice, StandardStream};
use rune::{Diagnostics, Vm};
use std::sync::Arc;

fn main() -> rune::Result<()> {
    let context = rune_modules::default_context()?;
    let runtime = Arc::new(context.runtime());

    let mut sources = rune::sources! {
        entry => {
            pub fn calc(input) {
                let a = input[0] + 1;
                let b = format!("{} World", input[1]);
                [a, b]
            }
        }
    };

    let mut diagnostics = Diagnostics::new();

    let result = rune::prepare(&mut sources)
        .with_context(&context)
        .with_diagnostics(&mut diagnostics)
        .build();

    if !diagnostics.is_empty() {
        let mut writer = StandardStream::stderr(ColorChoice::Always);
        diagnostics.emit(&mut writer, &sources)?;
    }

    let unit = result?;
    let mut vm = Vm::new(runtime, Arc::new(unit));

    let input: VecTuple<(i64, String)> = VecTuple::new((1, String::from("Hello")));
    let output = vm.call(["calc"], (input,))?;
    let VecTuple((a, b)): VecTuple<(u32, String)> = rune::from_value(output)?;

    println!("({:?}, {:?})", a, b);
    Ok(())
}
