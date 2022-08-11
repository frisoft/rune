use proc_macro2::TokenStream;
use quote::quote_spanned;
use syn::spanned::Spanned;

/// An internal call to the macro.
pub struct Expander {
    f: syn::ItemFn,
}

impl syn::parse::Parse for Expander {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let f: syn::ItemFn = input.parse()?;
        Ok(Self { f })
    }
}

impl Expander {
    pub fn expand(self, hir: bool) -> Result<TokenStream, Vec<syn::Error>> {
        let f = self.f;

        let mut it = f.sig.inputs.iter();

        let first = match it.next() {
            Some(syn::FnArg::Typed(ty)) => match &*ty.pat {
                syn::Pat::Ident(ident) => Some(&ident.ident),
                _ => None,
            },
            _ => None,
        };

        let second = if hir {
            first
        } else {
            match it.next() {
                Some(syn::FnArg::Typed(ty)) => match &*ty.pat {
                    syn::Pat::Ident(ident) => Some(&ident.ident),
                    _ => None,
                },
                _ => None,
            }
        };

        let ident = &f.sig.ident;

        let log = match (first, second) {
            (Some(a), Some(b)) => {
                let ident = syn::LitStr::new(&ident.to_string(), ident.span());

                Some(quote_spanned! {
                    ident.span() =>
                    let _instrument_span = ::tracing::span!(::tracing::Level::TRACE, #ident);
                    let _instrument_enter = _instrument_span.enter();

                    if let Some(source) = #b.q.sources.source(#b.source_id, crate::ast::Spanned::span(&#a)) {
                        ::tracing::trace!("{:?}", source);
                    }
                })
            }
            _ => None,
        };

        let span = f.span();
        let vis = &f.vis;
        let stmts = &f.block.stmts;
        let sig = &f.sig;

        Ok(quote_spanned! {
            span =>
            #vis #sig {
                #log
                { #(#stmts)* }
            }
        })
    }
}
