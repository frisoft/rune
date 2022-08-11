use crate::context::{Context, Tokens};
use proc_macro2::TokenStream;
use quote::{quote, quote_spanned};
use syn::spanned::Spanned as _;

struct Expander {
    ctx: Context,
    tokens: Tokens,
}

impl Expander {
    /// Expand on a struct.
    fn expand_struct(
        &mut self,
        input: &syn::DeriveInput,
        st: &syn::DataStruct,
    ) -> Result<TokenStream, ()> {
        let inner = self.expand_fields(&st.fields)?;

        let ident = &input.ident;
        let value = &self.tokens.value;
        let vm_error = &self.tokens.vm_error;
        let to_value = &self.tokens.to_value;

        Ok(quote! {
            impl #to_value for #ident {
                fn to_value(self) -> ::std::result::Result<#value, #vm_error> {
                    #inner
                }
            }
        })
    }

    /// Expand field decoding.
    fn expand_fields(&mut self, fields: &syn::Fields) -> Result<TokenStream, ()> {
        match fields {
            syn::Fields::Unnamed(named) => self.expand_unnamed(named),
            syn::Fields::Named(named) => self.expand_named(named),
            syn::Fields::Unit => {
                self.ctx.errors.push(syn::Error::new_spanned(
                    fields,
                    "unit structs are not supported",
                ));
                Err(())
            }
        }
    }

    /// Get a field identifier.
    fn field_ident<'a>(&mut self, field: &'a syn::Field) -> Result<&'a syn::Ident, ()> {
        match &field.ident {
            Some(ident) => Ok(ident),
            None => {
                self.ctx.errors.push(syn::Error::new_spanned(
                    field,
                    "unnamed fields are not supported",
                ));
                Err(())
            }
        }
    }

    /// Expand unnamed fields.
    fn expand_unnamed(&mut self, unnamed: &syn::FieldsUnnamed) -> Result<TokenStream, ()> {
        let mut to_values = Vec::new();

        for (index, field) in unnamed.unnamed.iter().enumerate() {
            let _ = self.ctx.field_attrs(&field.attrs)?;

            let index = syn::Index::from(index);

            let to_value = &self.tokens.to_value;

            to_values.push(quote_spanned! {
                field.span() =>
                tuple.push(#to_value::to_value(self.#index)?);
            });
        }

        let cap = unnamed.unnamed.len();
        let value = &self.tokens.value;
        let tuple = &self.tokens.tuple;

        Ok(quote_spanned! {
            unnamed.span() =>
            let mut tuple = Vec::with_capacity(#cap);
            #(#to_values)*
            Ok(#value::from(#tuple::from(tuple)))
        })
    }

    /// Expand named fields.
    fn expand_named(&mut self, named: &syn::FieldsNamed) -> Result<TokenStream, ()> {
        let mut to_values = Vec::new();

        for field in &named.named {
            let ident = self.field_ident(field)?;
            let _ = self.ctx.field_attrs(&field.attrs)?;

            let name = &syn::LitStr::new(&ident.to_string(), ident.span());

            let to_value = &self.tokens.to_value;

            to_values.push(quote_spanned! {
                field.span() =>
                object.insert(String::from(#name), #to_value::to_value(self.#ident)?);
            });
        }

        let value = &self.tokens.value;
        let object = &self.tokens.object;

        Ok(quote_spanned! {
            named.span() =>
            let mut object = <#object>::new();
            #(#to_values)*
            Ok(#value::from(object))
        })
    }
}

pub(super) fn expand(input: &syn::DeriveInput) -> Result<TokenStream, Vec<syn::Error>> {
    let mut ctx = Context::new();

    let attrs = match ctx.type_attrs(&input.attrs) {
        Ok(attrs) => attrs,
        Err(()) => {
            return Err(ctx.errors);
        }
    };

    let tokens = ctx.tokens_with_module(attrs.module.as_ref());

    let mut expander = Expander {
        ctx: Context::new(),
        tokens,
    };

    match &input.data {
        syn::Data::Struct(st) => {
            if let Ok(expanded) = expander.expand_struct(input, st) {
                return Ok(expanded);
            }
        }
        syn::Data::Enum(en) => {
            expander.ctx.errors.push(syn::Error::new_spanned(
                en.enum_token,
                "not supported on enums",
            ));
        }
        syn::Data::Union(un) => {
            expander.ctx.errors.push(syn::Error::new_spanned(
                un.union_token,
                "not supported on unions",
            ));
        }
    }

    Err(expander.ctx.errors)
}
