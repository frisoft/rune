use rune_macros::__instrument_hir as instrument;

use crate::ast::{self, Span};
use crate::collections::{HashMap, HashSet};
use crate::compile::assemble::type_match::tuple_match_for;
use crate::compile::assemble::{
    BoundPat, BoundPatKind, Ctxt, Expr, ExprKind, ExprStructKind, Result, TypeMatch,
};
use crate::compile::{
    AssemblyAddress, BindingName, CompileError, CompileErrorKind, PrivMeta, UnitBuilder,
};
use crate::hash::Hash;
use crate::parse::Resolve;

/// A struct pattern.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Pat<'hir> {
    span: Span,
    kind: PatKind<'hir>,
}

impl<'hir> Pat<'hir> {
    /// Construct a new pattern.
    pub(crate) fn new(span: Span, kind: PatKind<'hir>) -> Self {
        Self { span, kind }
    }

    /// Bind a pattern to an expression.
    pub(crate) fn bind(self, cx: &mut Ctxt<'_, 'hir>, expr: Expr<'hir>) -> Result<BoundPat<'hir>> {
        cx.with_span(self.span, |cx| {
            let scope = cx.scopes.clone_scope(cx.span, cx.scope)?;
            let mut removed = HashMap::new();

            let out = cx.with_scope(scope, |cx| self.bind_inner(cx, expr, &mut removed))?;

            // Free each *unique* binding that was replaced, in case the same binding
            for (_, (address, span)) in removed {
                if let AssemblyAddress::Slot(slot) = address {
                    cx.scopes.free_user(span, slot)?;
                }
            }

            // Replace the new updated scope with the old scope, allowing the new
            // set of variables to be visible from now on.
            cx.scopes.replace_scope(cx.span, cx.scope, scope)?;
            Ok(out)
        })
    }

    /// Unconditionally bind the given expression to the current pattern.
    fn bind_inner(
        self,
        cx: &mut Ctxt<'_, 'hir>,
        expr: Expr<'hir>,
        removed: &mut HashMap<BindingName, (AssemblyAddress, Span)>,
    ) -> Result<BoundPat<'hir>> {
        cx.with_span(self.span, |cx| {
            match self.kind {
                PatKind::Ignore => Ok(cx.bound_pat(BoundPatKind::Irrefutable)),
                PatKind::Lit { lit } => bind_pat_lit(cx, lit, expr),
                PatKind::Path {
                    path: PatPath::Ident { ident },
                } => {
                    let (address, expr) = match expr.kind {
                        ExprKind::Address { address, .. } => {
                            cx.scopes.retain(expr.span, address)?;
                            expr.scope.free(cx)?;
                            (address, cx.expr(ExprKind::Empty))
                        }
                        _ => {
                            let address = cx.scopes.alloc();
                            (address, expr)
                        }
                    };

                    // Handle syntactical reassignment of the given expression.
                    let ident = ident.resolve(resolve_context!(cx.q))?;
                    let binding_name = cx.scopes.binding_name(ident);

                    tracing::trace!(?ident, ?address, "reassign");

                    let (binding, replaced_address) =
                        cx.scopes
                            .declare_as(self.span, cx.scope, binding_name, address)?;

                    if let Some(address) = replaced_address {
                        if let Some((_, span)) = removed.insert(binding.name, (address, self.span))
                        {
                            return Err(CompileError::new(
                                self.span,
                                CompileErrorKind::DuplicateBinding {
                                    previous_span: span,
                                },
                            ));
                        }
                    }

                    Ok(cx.bound_pat(BoundPatKind::Expr {
                        address,
                        expr: alloc!(cx; expr),
                    }))
                }
                PatKind::Path {
                    path: PatPath::Meta { meta, .. },
                } => {
                    let type_match = match tuple_match_for(cx, meta) {
                        Some((args, inst)) if args == 0 => inst,
                        _ => return Err(cx.error(CompileErrorKind::UnsupportedPattern)),
                    };

                    let address = expr.as_address(cx)?;

                    Ok(cx.bound_pat(BoundPatKind::TypedSequence {
                        type_match,
                        expr: alloc!(cx; expr),
                        address,
                        items: &[],
                    }))
                }
                PatKind::Vec { items, is_open } => bind_pat_vec(cx, items, is_open, expr, removed),
                PatKind::Tuple {
                    kind,
                    patterns,
                    is_open,
                } => bind_pat_tuple(cx, kind, patterns, is_open, expr, removed),
                PatKind::Object { kind, patterns } => {
                    bind_pat_object(cx, kind, patterns, expr, removed)
                }
            }
        })
    }
}

/// A path that has been evaluated.
#[derive(Debug, Clone, Copy)]
pub(crate) enum PatPath<'hir> {
    /// An identifier as a pattern.
    Ident { ident: &'hir ast::Ident },
    /// A meta item as a pattern.
    Meta { meta: &'hir PrivMeta },
}

/// The kind of a pattern.
#[derive(Debug, Clone, Copy)]
pub(crate) enum PatKind<'hir> {
    /// An ignore pattern.
    Ignore,
    /// A literal value.
    Lit { lit: &'hir Expr<'hir> },
    /// A path pattern.
    Path { path: &'hir PatPath<'hir> },
    /// A vector pattern.
    Vec {
        items: &'hir [Pat<'hir>],
        is_open: bool,
    },
    /// A tuple pattern.
    Tuple {
        kind: PatTupleKind,
        patterns: &'hir [Pat<'hir>],
        is_open: bool,
    },
    /// An object pattern.
    Object {
        kind: PatObjectKind<'hir>,
        patterns: &'hir [Pat<'hir>],
    },
}

/// The type of a pattern object.
#[derive(Debug, Clone, Copy)]
pub(crate) enum PatObjectKind<'hir> {
    Typed {
        type_match: TypeMatch,
        keys: &'hir [(usize, Hash)],
    },
    Anonymous {
        slot: usize,
        is_open: bool,
    },
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum PatTupleKind {
    Typed { type_match: TypeMatch },
    Anonymous,
}

/// The outcome of a bind operation.
#[derive(Debug)]
pub(crate) enum PatOutcome {
    Irrefutable,
    Refutable,
}

impl PatOutcome {
    /// Combine this outcome with another.
    pub(crate) fn combine(self, other: Self) -> Self {
        match (self, other) {
            (PatOutcome::Irrefutable, PatOutcome::Irrefutable) => PatOutcome::Irrefutable,
            _ => PatOutcome::Refutable,
        }
    }
}

/// Bind a literal pattern.
#[instrument]
fn bind_pat_lit<'hir>(
    cx: &mut Ctxt<'_, 'hir>,
    lit: &'hir Expr<'hir>,
    expr: Expr<'hir>,
) -> Result<BoundPat<'hir>> {
    // Match irrefutable patterns.
    match (lit.kind, expr.kind) {
        (ExprKind::Store { value: a }, ExprKind::Store { value: b }) if a == b => {
            expr.scope.free(cx)?;
            return Ok(cx.bound_pat(BoundPatKind::Irrefutable));
        }
        (ExprKind::String { string: a }, ExprKind::String { string: b }) if a == b => {
            expr.scope.free(cx)?;
            return Ok(cx.bound_pat(BoundPatKind::Irrefutable));
        }
        (ExprKind::Bytes { bytes: a }, ExprKind::Bytes { bytes: b }) if a == b => {
            expr.scope.free(cx)?;
            return Ok(cx.bound_pat(BoundPatKind::Irrefutable));
        }
        _ => {}
    }

    Ok(cx.bound_pat(BoundPatKind::Lit {
        lit,
        expr: alloc!(cx; expr),
    }))
}

/// Bind a vector pattern.
#[instrument]
fn bind_pat_vec<'hir>(
    cx: &mut Ctxt<'_, 'hir>,
    patterns: &'hir [Pat<'hir>],
    is_open: bool,
    expr: Expr<'hir>,
    removed: &mut HashMap<BindingName, (AssemblyAddress, Span)>,
) -> Result<BoundPat<'hir>> {
    // Try a simpler form of pattern matching through syntactical reassignment.
    match expr.kind {
        ExprKind::Vec { items: expr_items }
            if expr_items.len() == patterns.len()
                || expr_items.len() >= patterns.len() && is_open =>
        {
            let items = iter!(cx; patterns.into_iter().zip(expr_items), |(pat, expr)| {
                pat.bind_inner(cx, *expr, removed)?
            });

            expr.scope.free(cx)?;

            return Ok(cx.bound_pat(BoundPatKind::IrrefutableSequence { items }));
        }
        _ => {}
    }

    let address = cx.scopes.array_index();

    cx.scopes.alloc_array_items(patterns.len());

    let items = iter!(cx; patterns, |pat| {
        cx.scopes.free_array_item(cx.span)?;

        let expr = cx.expr(ExprKind::Address {
            address: cx.scopes.array_index(),
            binding: None,
        });

        pat.bind_inner(
            cx,
            expr,
            removed,
        )?
    });

    Ok(cx.bound_pat(BoundPatKind::Vec {
        expr: alloc!(cx; expr),
        address,
        is_open,
        items,
    }))
}

/// Bind a tuple pattern.
#[instrument]
fn bind_pat_tuple<'hir>(
    cx: &mut Ctxt<'_, 'hir>,
    kind: PatTupleKind,
    patterns: &'hir [Pat<'hir>],
    is_open: bool,
    expr: Expr<'hir>,
    removed: &mut HashMap<BindingName, (AssemblyAddress, Span)>,
) -> Result<BoundPat<'hir>> {
    match kind {
        PatTupleKind::Typed { type_match } => {
            let address = match expr.kind {
                ExprKind::Address { address, .. } => {
                    cx.scopes.retain(cx.span, address)?;
                    address
                }
                _ => cx.scopes.alloc(),
            };

            let items = iter!(cx; patterns.iter().enumerate().rev(), |(index, pat)| {
                let expr = cx.expr(ExprKind::TupleFieldAccess {
                    lhs: alloc!(cx; cx.expr(ExprKind::Address { address, binding: None })),
                    index,
                });

                pat.bind(cx, expr)?
            });

            Ok(cx.bound_pat(BoundPatKind::TypedSequence {
                type_match,
                expr: alloc!(cx; expr),
                address,
                items,
            }))
        }
        PatTupleKind::Anonymous => {
            // Try a simpler form of pattern matching through syntactical
            // reassignment.
            match expr.kind {
                ExprKind::Tuple { items: tuple_items }
                    if tuple_items.len() == patterns.len()
                        || is_open && tuple_items.len() >= patterns.len() =>
                {
                    let items = iter!(cx; patterns.iter().zip(tuple_items), |(pat, expr)| {
                        pat.bind_inner(cx, *expr, removed)?
                    });

                    expr.scope.free(cx)?;
                    return Ok(cx.bound_pat(BoundPatKind::IrrefutableSequence { items }));
                }
                _ => {}
            }

            let address = cx.scopes.array_index();

            cx.scopes.alloc_array_items(patterns.len());

            let items = iter!(cx; patterns, |pat| {
                cx.scopes.free_array_item(cx.span)?;

                let expr = cx.expr(ExprKind::Address {
                    address: cx.scopes.array_index(),
                    binding: None,
                });

                pat.bind_inner(cx, expr, removed)?
            });

            Ok(cx.bound_pat(BoundPatKind::AnonymousTuple {
                expr: alloc!(cx; expr),
                address,
                is_open,
                items,
            }))
        }
    }
}

/// Set up binding for pattern objects.
#[instrument]
fn bind_pat_object<'hir>(
    cx: &mut Ctxt<'_, 'hir>,
    kind: PatObjectKind<'hir>,
    patterns: &'hir [Pat<'hir>],
    expr: Expr<'hir>,
    removed: &mut HashMap<BindingName, (AssemblyAddress, Span)>,
) -> Result<BoundPat<'hir>> {
    match kind {
        PatObjectKind::Typed { type_match, keys } => {
            let address = expr.as_address(cx)?;

            let items = iter!(cx; keys.iter().zip(patterns), |(&(slot, hash), pat)| {
                let expr = cx.expr(ExprKind::StructFieldAccess {
                    lhs: alloc!(cx; cx.expr(ExprKind::Address { address, binding: None })),
                    slot,
                    hash,
                });

                pat.bind_inner(cx, expr, removed)?
            });

            Ok(cx.bound_pat(BoundPatKind::TypedSequence {
                type_match,
                expr: alloc!(cx; expr),
                address,
                items,
            }))
        }
        PatObjectKind::Anonymous { slot, is_open } => {
            // Try a simpler form of pattern matching through syntactical
            // reassignment.
            match expr.kind {
                ExprKind::Struct {
                    kind: ExprStructKind::Anonymous { slot: expr_slot },
                    exprs,
                } if object_keys_match(&cx.q.unit, expr_slot, slot, is_open)
                    .unwrap_or_default() =>
                {
                    let items = iter!(cx; patterns.iter().zip(exprs), |(pat, expr)| {
                        pat.bind_inner(cx, *expr, removed)?
                    });

                    expr.scope.free(cx)?;
                    return Ok(cx.bound_pat(BoundPatKind::IrrefutableSequence { items }));
                }
                _ => {}
            }

            let address = cx.scopes.array_index();

            cx.scopes.alloc_array_items(patterns.len());

            let items = iter!(cx; patterns, |pat| {
                cx.scopes.free_array_item(cx.span)?;

                let expr = cx.expr(ExprKind::Address {
                    address: cx.scopes.array_index(),
                    binding: None,
                });

                pat.bind_inner(cx, expr, removed)?
            });

            Ok(cx.bound_pat(BoundPatKind::AnonymousObject {
                expr: alloc!(cx; expr),
                address,
                slot,
                is_open,
                items,
            }))
        }
    }
}

/// Test if object keys match.
fn object_keys_match(
    unit: &UnitBuilder,
    from_slot: usize,
    to_slot: usize,
    is_open: bool,
) -> Option<bool> {
    let from_keys = unit.get_static_object_keys(from_slot)?;
    let to_keys = unit.get_static_object_keys(to_slot)?;

    let mut from_keys = from_keys.iter().cloned().collect::<HashSet<_>>();

    for key in to_keys {
        if !from_keys.remove(key) {
            return Some(false);
        }
    }

    Some(is_open || from_keys.is_empty())
}
