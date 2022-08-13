use proc_macro2::TokenStream;
use quote::quote_spanned;
use syn::parse::ParseStream;
use syn::spanned::Spanned;

use crate::context::{Context, Tokens};

/// An internal call to the macro.
pub struct Derive {
    input: syn::DeriveInput,
}

impl syn::parse::Parse for Derive {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        Ok(Self {
            input: input.parse()?,
        })
    }
}

impl Derive {
    pub(crate) fn expand(self) -> Result<TokenStream, Vec<syn::Error>> {
        let mut cx = Context::new();

        let attrs = match cx.type_attrs(&self.input.attrs) {
            Ok(attrs) => attrs,
            Err(()) => return Err(cx.errors),
        };

        let tokens = cx.tokens_with_module(attrs.module.as_ref());

        match self.expand_inner(&mut cx, &tokens) {
            Ok(stream) => {
                if !cx.errors.is_empty() {
                    return Err(cx.errors);
                }

                Ok(stream)
            }
            Err(()) => Err(cx.errors),
        }
    }

    fn expand_inner(self, cx: &mut Context, tokens: &Tokens) -> Result<TokenStream, ()> {
        let data = match &self.input.data {
            syn::Data::Enum(data) => data,
            _ => {
                cx.errors.push(syn::Error::new_spanned(
                    &self.input,
                    "only enums not supported",
                ));
                return Err(());
            }
        };

        let type_ident = &self.input.ident;
        let label_ident = &syn::Ident::new("label", self.input.span());

        let mut variants = Vec::new();
        let mut matches = Vec::new();
        let mut format_branches = Vec::new();

        for variant in &data.variants {
            let _ = cx.variant_attrs(&variant.attrs)?;

            let ident = &variant.ident;

            let mut decls = Vec::new();
            let mut pattern = Vec::new();
            let mut fmt_patterns = Vec::new();
            let mut decl = Vec::new();
            let mut translate = Vec::new();
            let mut format_string = to_dashes(ident.to_string());

            let mut it = variant.fields.iter().peekable();

            while let Some(field) = it.next() {
                let attr = cx.field_attrs(&field.attrs)?;

                let ident = match &field.ident {
                    Some(ident) => ident,
                    None => {
                        cx.errors.push(syn::Error::new_spanned(
                            field,
                            "unnamed variants are not supported",
                        ));
                        break;
                    }
                };

                let f = if attr.debug.is_some() { "{:?}" } else { "{}" };

                format_string.push_str(&format!(" {ident}={f}"));

                if it.peek().is_some() {
                    format_string.push(',');
                }

                if attr.address.is_some() {
                    let assembly_address = &tokens.assembly_address;

                    match &field.ty {
                        syn::Type::Array(array) => {
                            let len: usize = match array_len(cx, array) {
                                Ok(len) => len,
                                Err(()) => continue,
                            };

                            let mut elements = Vec::with_capacity(len);

                            for index in 0..len {
                                elements.push(quote_spanned!(field.span() => allocator.translate(span, #ident[#index])?));
                            }

                            translate.push(quote_spanned! {
                                variant.span() =>
                                let #ident = [#(#elements),*];
                            });

                            decls.push(
                                quote_spanned!(field.span() => #ident: [#assembly_address; #len]),
                            );
                        }
                        _ => {
                            translate.push(quote_spanned! {
                                variant.span() =>
                                let #ident = allocator.translate(span, #ident)?;
                            });

                            decls.push(quote_spanned!(field.span() => #ident: #assembly_address));
                        }
                    }

                    pattern.push(ident);
                } else if attr.label.is_some() {
                    let label = &tokens.label;

                    translate.push(quote_spanned! {
                        variant.span() =>
                        let #ident = translate_label(#label_ident)?;
                    });

                    decls.push(quote_spanned!(field.span() => label: #label));
                    pattern.push(label_ident);
                } else {
                    let ty = &field.ty;
                    decls.push(quote_spanned!(field.span() => #ident: #ty));
                    pattern.push(ident);
                }

                decl.push(ident);
                fmt_patterns.push(ident);
            }

            match &variant.fields {
                syn::Fields::Named(_) => {
                    variants.push(quote_spanned! {
                        variant.span() =>
                        #ident { #(#decls),* }
                    });
                }
                syn::Fields::Unnamed(unnamed) => {
                    cx.errors.push(syn::Error::new_spanned(
                        unnamed,
                        "unnamed variants are not supported",
                    ));
                    continue;
                }
                syn::Fields::Unit => {
                    variants.push(quote_spanned! {
                        variant.span() => #ident
                    });
                }
            }

            matches.push(quote_spanned! {
                variant.span() =>
                Self::#ident { #(#pattern),* } => {
                    #(#translate)*
                    Ok(#type_ident::#ident {
                        #(#decl),*
                    })
                }
            });

            format_branches.push(quote_spanned! {
                variant.span() =>
                Self::#ident { #(#fmt_patterns),* } => {
                    write!(f, #format_string, #(#fmt_patterns),*)
                }
            });
        }

        let compile_error = &tokens.compile_error;
        let allocator = &tokens.allocator;
        let span = &tokens.span;
        let label = &tokens.label;
        let fmt_display = &tokens.fmt_display;
        let fmt_result = &tokens.fmt_result;
        let fmt_formatter = &tokens.fmt_formatter;

        Ok(quote_spanned! {
            self.input.span() =>
            #[derive(Debug, Clone, Copy)]
            pub(crate) enum AssemblyInst {
                #(#variants,)*
            }

            impl AssemblyInst {
                pub(crate) fn translate<T>(self, span: #span, allocator: &#allocator, translate_label: T) -> Result<#type_ident, #compile_error>
                where
                    T: Fn(#label) -> Result<isize, #compile_error>
                {
                    match self {
                        #(#matches,)*
                    }
                }
            }

            impl #fmt_display for #type_ident {
                fn fmt(&self, f: &mut #fmt_formatter<'_>) -> #fmt_result {
                    match self {
                        #(#format_branches)*
                    }
                }
            }
        })
    }
}

/// Array length.
fn array_len(cx: &mut Context, array: &syn::TypeArray) -> Result<usize, ()> {
    match &array.len {
        syn::Expr::Lit(syn::ExprLit {
            lit: syn::Lit::Int(n),
            ..
        }) => match n.base10_parse() {
            Ok(n) => Ok(n),
            Err(e) => {
                cx.errors.push(e);
                Err(())
            }
        },
        expr => {
            cx.errors
                .push(syn::Error::new_spanned(expr, "unsupported array length"));
            Err(())
        }
    }
}

fn to_dashes(string: String) -> String {
    let mut out = String::new();

    let mut it = string.chars();

    if let Some(c) = it.next() {
        out.extend(c.to_lowercase());
    }

    for c in it {
        if c.is_uppercase() {
            out.push('-');
            out.extend(c.to_lowercase());
        } else {
            out.push(c);
        }
    }

    out
}
