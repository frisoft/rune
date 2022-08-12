use crate::ast::Span;
use crate::compile::assemble::{Ctxt, Expr, ExprKind, MaybeAlloc, PatOutcome, Result, TypeMatch};
use crate::compile::{CompileErrorKind, Label};
use crate::runtime::{AssemblyInst as Inst, PanicReason, TypeCheck};

/// A pattern that has been bound with an expression.
///
/// Think of this as the state of an expression like `let Some(value) = var`.
/// Where `Some(value)` is the pattern and `var` is the expression that has been
/// bound into the pattern.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BoundPat<'hir> {
    span: Span,
    kind: BoundPatKind<'hir>,
}

/// The kind of a [BoundPat].
#[derive(Debug, Clone, Copy)]
pub(crate) enum BoundPatKind<'hir> {
    Irrefutable,
    IrrefutableSequence {
        items: &'hir [BoundPat<'hir>],
    },
    Lit {
        lit: &'hir Expr<'hir>,
        expr: &'hir Expr<'hir>,
    },
    Vec {
        address: MaybeAlloc,
        is_open: bool,
        items: &'hir [BoundPat<'hir>],
    },
    AnonymousTuple {
        address: MaybeAlloc,
        is_open: bool,
        items: &'hir [BoundPat<'hir>],
    },
    AnonymousObject {
        address: MaybeAlloc,
        slot: usize,
        is_open: bool,
        items: &'hir [BoundPat<'hir>],
    },
    TypedSequence {
        type_match: TypeMatch,
        address: MaybeAlloc,
        items: &'hir [BoundPat<'hir>],
    },
}

impl<'hir> BoundPat<'hir> {
    /// Construct a new bound pattern.
    pub(crate) const fn new(span: Span, kind: BoundPatKind<'hir>) -> Self {
        Self { span, kind }
    }

    /// Compile a pattern expression that panics in case it doesn't match.
    pub(crate) fn compile_or_panic(self, cx: &mut Ctxt<'_, 'hir>) -> Result<()> {
        cx.with_span(self.span, |cx| {
            let panic_label = cx.new_label("pat_panic");

            if let PatOutcome::Refutable = self.compile_inner(cx, panic_label)? {
                let end = cx.new_label("pat_end");
                cx.push(Inst::Jump { label: end });
                cx.label(panic_label)?;
                cx.push(Inst::Panic {
                    reason: PanicReason::UnmatchedPattern,
                });
                cx.label(end)?;
            }

            Ok(())
        })
    }

    /// Compile a pattern that jumps to the given label if it doesn't match.
    pub(crate) fn compile(self, cx: &mut Ctxt<'_, 'hir>, label: Label) -> Result<PatOutcome> {
        cx.with_span(self.span, |cx| self.compile_inner(cx, label))
    }

    fn compile_inner(self, cx: &mut Ctxt<'_, 'hir>, label: Label) -> Result<PatOutcome> {
        match self.kind {
            BoundPatKind::Irrefutable => Ok(PatOutcome::Irrefutable),
            BoundPatKind::IrrefutableSequence { items } => {
                let mut outcome = PatOutcome::Irrefutable;

                for pat in items {
                    outcome = outcome.combine(pat.compile(cx, label)?);
                }

                Ok(outcome)
            }
            BoundPatKind::Lit { lit, expr } => {
                let output = expr.compile(cx, None)?.ensure_address(cx)?;

                match lit.kind {
                    ExprKind::Store { value } => {
                        cx.push(Inst::MatchValue {
                            address: *output,
                            value,
                            label,
                        });
                    }
                    ExprKind::String { string } => {
                        let slot = cx.q.unit.new_static_string(cx.span, string)?;
                        cx.push(Inst::MatchString {
                            address: *output,
                            slot,
                            label,
                        });
                    }
                    ExprKind::Bytes { bytes } => {
                        let slot = cx.q.unit.new_static_bytes(cx.span, bytes)?;
                        cx.push(Inst::MatchBytes {
                            address: *output,
                            slot,
                            label,
                        });
                    }
                    _ => {
                        return Err(cx.error(CompileErrorKind::UnsupportedPattern));
                    }
                }

                output.free(cx)?;
                Ok(PatOutcome::Refutable)
            }
            BoundPatKind::Vec {
                address,
                is_open,
                items,
            } => {
                cx.push(Inst::MatchSequence {
                    address: *address,
                    type_check: TypeCheck::Vec,
                    len: items.len(),
                    exact: !is_open,
                    label,
                    output: *address,
                });

                address.free(cx)?;

                for pat in items {
                    pat.compile(cx, label)?;
                }

                Ok(PatOutcome::Refutable)
            }
            BoundPatKind::AnonymousTuple {
                address,
                is_open,
                items,
            } => {
                cx.push(Inst::MatchSequence {
                    address: *address,
                    type_check: TypeCheck::Tuple,
                    len: items.len(),
                    exact: !is_open,
                    label,
                    output: *address,
                });

                address.free(cx)?;

                for pat in items {
                    pat.compile(cx, label)?;
                }

                Ok(PatOutcome::Refutable)
            }
            BoundPatKind::AnonymousObject {
                address,
                slot,
                is_open,
                items,
            } => {
                cx.push(Inst::MatchObject {
                    address: *address,
                    slot,
                    exact: !is_open,
                    label,
                    output: *address,
                });

                address.free(cx)?;

                for pat in items {
                    pat.compile(cx, label)?;
                }

                Ok(PatOutcome::Refutable)
            }
            BoundPatKind::TypedSequence {
                type_match,
                address,
                items,
            } => {
                match type_match {
                    TypeMatch::BuiltIn { type_check } => cx.push(Inst::MatchBuiltIn {
                        address: *address,
                        type_check,
                        label,
                    }),
                    TypeMatch::Type { type_hash } => cx.push(Inst::MatchType {
                        address: *address,
                        type_hash,
                        label,
                    }),
                    TypeMatch::Variant {
                        variant_hash,
                        enum_hash,
                        index,
                    } => {
                        let output = cx.scopes.temporary();

                        cx.push(Inst::MatchVariant {
                            address: *address,
                            variant_hash,
                            enum_hash,
                            index,
                            label,
                            output,
                        });
                    }
                }

                address.free(cx)?;

                for pat in items {
                    pat.compile(cx, label)?;
                }

                Ok(PatOutcome::Refutable)
            }
        }
    }
}
