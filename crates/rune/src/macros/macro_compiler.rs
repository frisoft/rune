//! Macro compiler.

use crate::arena::Arena;
use crate::ast::{self, Spanned, SpannedError};
use crate::compile::{CompileError, CompileErrorKind, IrError, ItemMeta, Options};
use crate::macros::MacroContext;
use crate::parse::{Parse, ParseError, Parser};
use crate::query::Query;
use crate::Context;

pub(crate) struct MacroCompiler<'a> {
    pub(crate) item_meta: ItemMeta,
    pub(crate) options: &'a Options,
    pub(crate) context: &'a Context,
    pub(crate) query: Query<'a>,
}

impl MacroCompiler<'_> {
    /// Compile the given macro into the given output type.
    pub(crate) fn eval_macro<T>(&mut self, macro_call: &ast::MacroCall) -> Result<T, CompileError>
    where
        T: Parse,
    {
        let span = macro_call.span();

        if !self.options.macros {
            return Err(CompileError::experimental(
                span,
                "macros must be enabled with `-O macros=true`",
            ));
        }

        // TODO: include information on the module the macro is being called
        // from.
        //
        // TODO: Figure out how to avoid performing ad-hoc lowering here.
        let arena = Arena::new();

        let path = {
            let mut cx = crate::hir::lowering::Ctxt::new(
                self.item_meta.location.source_id,
                self.query.borrow(),
                &arena,
            );
            crate::hir::lowering::path(&mut cx, &macro_call.path)?
        };
        let named = self.query.convert_path(self.context, &path)?;

        let hash = self.query.pool.item_type_hash(named.item);

        let handler = match self.context.lookup_macro(hash) {
            Some(handler) => handler,
            None => {
                return Err(CompileError::new(
                    span,
                    CompileErrorKind::MissingMacro {
                        item: self.query.pool.item(named.item).to_owned(),
                    },
                ));
            }
        };

        let input_stream = &macro_call.stream;

        // SAFETY: Macro context only needs to live for the duration of the
        // `handler` call.
        let result = {
            let mut macro_context = MacroContext {
                macro_span: macro_call.span(),
                stream_span: macro_call.stream_span(),
                item_meta: self.item_meta,
                q: self.query.borrow(),
            };

            handler(&mut macro_context, input_stream)
        };

        let token_stream = match result {
            Ok(output) => output,
            Err(error) => {
                let error = match error.downcast::<ParseError>() {
                    Ok(error) => return Err(CompileError::from(error)),
                    Err(error) => error,
                };

                let error = match error.downcast::<IrError>() {
                    Ok(error) => return Err(CompileError::from(error)),
                    Err(error) => error,
                };

                let error = match error.downcast::<CompileError>() {
                    Ok(error) => return Err(error),
                    Err(error) => error,
                };

                let error = match error.downcast::<SpannedError>() {
                    Ok(error) => {
                        return Err(CompileError::new(
                            error.span(),
                            CompileErrorKind::CallMacroError {
                                item: self.query.pool.item(named.item).to_owned(),
                                error: error.into_inner(),
                            },
                        ));
                    }
                    Err(error) => error,
                };

                return Err(CompileError::new(
                    span,
                    CompileErrorKind::CallMacroError {
                        item: self.query.pool.item(named.item).to_owned(),
                        error,
                    },
                ));
            }
        };

        let mut parser = Parser::from_token_stream(&token_stream, span);
        let output = parser.parse::<T>()?;
        parser.eof()?;

        Ok(output)
    }
}
