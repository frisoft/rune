use rune_macros::__instrument_hir as instrument;

use crate::compile::assemble::Ctxt;
use crate::compile::{PrivMeta, PrivMetaKind, PrivStructMeta, PrivVariantMeta};
use crate::hash::Hash;
use crate::runtime::TypeCheck;

#[derive(Debug, Clone, Copy)]
pub(crate) enum TypeMatch {
    BuiltIn {
        type_check: TypeCheck,
    },
    Type {
        type_hash: Hash,
    },
    Variant {
        variant_hash: Hash,
        enum_hash: Hash,
        index: usize,
    },
}

/// Construct the appropriate match instruction for the given [PrivMeta].
#[instrument]
pub(crate) fn tuple_match_for<'hir>(
    cx: &mut Ctxt<'_, 'hir>,
    meta: &PrivMeta,
) -> Option<(usize, TypeMatch)> {
    match &meta.kind {
        PrivMetaKind::Struct {
            type_hash,
            variant: PrivVariantMeta::Unit,
            ..
        } => Some((
            0,
            TypeMatch::Type {
                type_hash: *type_hash,
            },
        )),
        PrivMetaKind::Struct {
            type_hash,
            variant: PrivVariantMeta::Tuple(tuple),
            ..
        } => Some((
            tuple.args,
            TypeMatch::Type {
                type_hash: *type_hash,
            },
        )),
        PrivMetaKind::Variant {
            enum_hash,
            type_hash,
            index,
            variant,
            ..
        } => {
            let args = match variant {
                PrivVariantMeta::Tuple(tuple) => tuple.args,
                PrivVariantMeta::Unit => 0,
                _ => return None,
            };

            let struct_match = if let Some(type_check) = cx.context.type_check_for(*type_hash) {
                TypeMatch::BuiltIn { type_check }
            } else {
                TypeMatch::Variant {
                    enum_hash: *enum_hash,
                    variant_hash: *type_hash,
                    index: *index,
                }
            };

            Some((args, struct_match))
        }
        meta => {
            tracing::trace!(?meta, "no match");
            None
        }
    }
}

/// Construct the appropriate match instruction for the given [PrivMeta].
#[instrument]
pub(crate) fn struct_match_for<'a>(
    cx: &mut Ctxt<'_, '_>,
    meta: &'a PrivMeta,
) -> Option<(&'a PrivStructMeta, TypeMatch)> {
    Some(match &meta.kind {
        PrivMetaKind::Struct {
            type_hash,
            variant: PrivVariantMeta::Struct(st),
            ..
        } => (
            st,
            TypeMatch::Type {
                type_hash: *type_hash,
            },
        ),
        PrivMetaKind::Variant {
            type_hash,
            enum_hash,
            index,
            variant: PrivVariantMeta::Struct(st),
            ..
        } => {
            let type_check = if let Some(type_check) = cx.context.type_check_for(*type_hash) {
                TypeMatch::BuiltIn { type_check }
            } else {
                TypeMatch::Variant {
                    variant_hash: *type_hash,
                    enum_hash: *enum_hash,
                    index: *index,
                }
            };

            (st, type_check)
        }
        _ => {
            return None;
        }
    })
}
