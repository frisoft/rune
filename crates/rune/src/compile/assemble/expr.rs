use std::mem;

use rune_macros::__instrument_hir as instrument;

use crate::ast::{self, Span};
use crate::compile::assemble::{
    assemble_block, assemble_expr_const, assemble_expr_value, assemble_pat, generics_parameters,
    ConditionBranch, Ctxt, CtxtState, MatchBranch, MaybeAlloc, Needs, Pat, PatKind, PatOutcome,
    Result,
};
use crate::compile::{
    AssemblyAddress, Binding, BindingName, CompileError, CompileErrorKind, ControlFlow, PrivMeta,
    PrivMetaKind, PrivVariantMeta, ScopeId,
};
use crate::hash::Hash;
use crate::hir;
use crate::parse::Resolve;
use crate::query::Named;
use crate::runtime::{
    AssemblyInst as Inst, FormatSpec, InstAssignOp, InstOp, InstRangeLimits, InstTarget, InstValue,
    InstVariant, Protocol,
};

/// An expression that can be assembled.
#[must_use]
#[derive(Debug, Clone, Copy)]
pub(crate) struct Expr<'hir> {
    /// The span of the assembled expression.
    pub(crate) span: Span,
    /// The scope of the expression.
    pub(crate) scope: ExprScope,
    /// The kind of the expression.
    pub(crate) kind: ExprKind<'hir>,
}

impl<'hir> Expr<'hir> {
    pub(crate) const fn new(span: Span, scope: ScopeId, kind: ExprKind<'hir>) -> Self {
        Self {
            span,
            scope: ExprScope {
                id: scope,
                free: false,
            },
            kind,
        }
    }

    /// Wrap the curent expression as a return expression.
    pub(crate) fn into_return(self, cx: &mut Ctxt<'_, 'hir>) -> Result<Expr<'hir>> {
        let expr = alloc!(cx; self);

        Ok(Expr::new(
            expr.span,
            expr.scope.id,
            ExprKind::Return { expr },
        ))
    }

    /// Try to coerce the current expression into its kind.
    pub(crate) fn into_kind(self) -> Result<ExprKind<'hir>> {
        if self.scope.id != ScopeId::CONST {
            return Err(CompileError::msg(
                self.span,
                "cannot coerce non-const expression into kind",
            ));
        }

        Ok(self.kind)
    }

    /// Mark the current expression as responsible for freeing its scope.
    pub(crate) fn free_scope(mut self) -> Self {
        self.scope.free = true;
        self
    }

    /// Perform custom mapping over the current expression kind, preserving any
    /// scope cleanup that might be required.
    pub(crate) fn map<T>(self, custom: T) -> Result<Self>
    where
        T: FnOnce(ExprKind<'hir>) -> Result<ExprKind<'hir>>,
    {
        Ok(Self {
            kind: custom(self.kind)?,
            ..self
        })
    }

    /// Compile the expression into a sequence of instructions.
    pub(crate) fn compile(
        self,
        cx: &mut Ctxt<'_, 'hir>,
        output: Option<AssemblyAddress>,
    ) -> Result<ExprOutcome> {
        if matches!(cx.state, CtxtState::Unreachable { reported: false }) {
            cx.q.diagnostics
                .unreachable(cx.source_id, self.span, Some(cx.context()));
            cx.state = CtxtState::Unreachable { reported: true };
        }

        let outcome = cx.with_span(self.span, |cx| {
            cx.with_scope(self.scope.id, |cx| self.compile_inner(cx, output))
        })?;
        self.scope.free(cx)?;

        let outcome = match (cx.state, outcome) {
            (CtxtState::Default, ExprOutcome::Unreachable) => {
                // Suppresses code generation from this point on.
                cx.state = CtxtState::Unreachable { reported: false };
                ExprOutcome::Unreachable
            }
            (CtxtState::Unreachable { .. }, outcome) => {
                outcome.free(cx)?;
                ExprOutcome::Unreachable
            }
            (_, outcome) => outcome,
        };

        Ok(outcome)
    }

    #[instrument]
    fn compile_inner(
        self,
        cx: &mut Ctxt<'_, 'hir>,
        output: Option<AssemblyAddress>,
    ) -> Result<ExprOutcome> {
        let outcome = match self.kind {
            ExprKind::Function { hash } => {
                let output = cx.alloc_or(output, |cx, output| {
                    cx.push(Inst::LoadFn { hash, output });
                    Ok(())
                })?;

                ExprOutcome::Output(output)
            }
            ExprKind::Closure { hash, captures } => {
                let output = cx.alloc_or(output, |cx, output| {
                    cx.array(captures, |cx, address, count| {
                        cx.push(Inst::Closure {
                            hash,
                            address,
                            count,
                            output,
                        });
                        Ok(())
                    })
                })?;

                ExprOutcome::Output(output)
            }
            ExprKind::Empty => ExprOutcome::Empty,
            ExprKind::Return { expr } => {
                let output = cx.alloc_or(output, |cx, output| {
                    match expr.kind {
                        ExprKind::Store {
                            value: InstValue::Unit,
                        } => {
                            cx.push(Inst::ReturnUnit);
                            expr.scope.free(cx)?;
                        }
                        _ => cx.addresses([*expr], [output], |cx, [address]| {
                            cx.push(Inst::Return { address });
                            Ok(())
                        })?,
                    }

                    Ok(())
                })?;

                output.free(cx)?;
                ExprOutcome::Unreachable
            }
            ExprKind::Assign { address, rhs } => {
                rhs.compile(cx, Some(address))?.free(cx)?;
                ExprOutcome::Empty
            }
            ExprKind::Let { pat, expr } => {
                pat.bind(cx, *expr)?.compile_or_panic(cx)?;
                ExprOutcome::Empty
            }
            ExprKind::Conditions { branches } => compile_expr_conditions(cx, branches, output)?,
            ExprKind::Matches {
                expr,
                address,
                branches,
            } => compile_expr_matches(cx, expr, address, branches, output)?,
            ExprKind::Address {
                binding, address, ..
            } => {
                if let Some(output) = output {
                    if address == output {
                        // No copy necessary.
                        return Ok(ExprOutcome::Output(MaybeAlloc::temporary(address)));
                    }
                }

                let address = cx.alloc_or(output, |cx, output| {
                    let comment = match &binding {
                        Some(binding) => {
                            let name = cx.scopes.name(cx.span, binding.name)?;
                            format!("copy `{}` (scope: {:?})", name, binding.scope)
                        }
                        None => format!("copy from anonymous binding"),
                    };

                    cx.push_with_comment(Inst::Copy { address, output }, comment);
                    Ok(())
                })?;

                ExprOutcome::Output(address)
            }
            ExprKind::Option { value } => {
                let address = cx.alloc_or(output, |cx, output| {
                    let variant = match value {
                        Some(&value) => {
                            value.compile(cx, Some(output))?.free(cx)?;
                            InstVariant::Some
                        }
                        None => InstVariant::None,
                    };

                    cx.push(Inst::Variant {
                        address: output,
                        variant,
                        output,
                    });
                    Ok(())
                })?;

                ExprOutcome::Output(address)
            }
            ExprKind::Yield { expr } => {
                let address = cx.alloc_or(output, |cx, output| {
                    match expr {
                        Some(expr) => {
                            cx.addresses([*expr], [output], |cx, [address]| {
                                cx.push(Inst::Yield { address, output });
                                Ok(())
                            })?;
                        }
                        None => {
                            cx.push(Inst::YieldUnit { output });
                        }
                    }

                    Ok(())
                })?;

                ExprOutcome::Output(address)
            }
            ExprKind::Await { expr } => {
                let address = cx.alloc_or(output, |cx, output| {
                    cx.addresses([*expr], [output], |cx, [address]| {
                        cx.push(Inst::Await { address, output });
                        Ok(())
                    })
                })?;

                ExprOutcome::Output(address)
            }
            ExprKind::Try { expr } => {
                let address = cx.alloc_or(output, |cx, output| {
                    cx.addresses([*expr], [output], |cx, [address]| {
                        cx.push(Inst::Try { address, output });
                        Ok(())
                    })
                })?;

                ExprOutcome::Output(address)
            }
            ExprKind::Store { value } => {
                let address = cx.alloc_or(output, |cx, output| {
                    cx.push(Inst::Store { value, output });
                    Ok(())
                })?;

                ExprOutcome::Output(address)
            }
            ExprKind::String { string } => {
                let address = cx.alloc_or(output, |cx, output| {
                    let slot = cx.q.unit.new_static_string(cx.span, string)?;
                    cx.push(Inst::String { slot, output });
                    Ok(())
                })?;

                ExprOutcome::Output(address)
            }
            ExprKind::Bytes { bytes } => {
                let address = cx.alloc_or(output, |cx, output| {
                    let slot = cx.q.unit.new_static_bytes(cx.span, bytes)?;
                    cx.push(Inst::Bytes { slot, output });
                    Ok(())
                })?;

                ExprOutcome::Output(address)
            }
            ExprKind::Vec { items: args } => {
                let address = cx.alloc_or(output, |cx, output| {
                    cx.array(args, |cx, address, count| {
                        cx.push(Inst::Vec {
                            address,
                            count,
                            output,
                        });
                        Ok(())
                    })
                })?;

                ExprOutcome::Output(address)
            }
            ExprKind::Tuple { items: args } => {
                let output = cx.alloc_or(output, |cx, output| match args {
                    &[a] => cx.addresses([a], [output], |cx, [a]| {
                        cx.push(Inst::Tuple1 { args: [a], output });
                        Ok(())
                    }),
                    &[a, b] => cx.addresses([a, b], [output], |cx, [a, b]| {
                        cx.push(Inst::Tuple2 {
                            args: [a, b],
                            output,
                        });
                        Ok(())
                    }),
                    &[a, b, c] => cx.addresses([a, b, c], [output], |cx, [a, b, c]| {
                        cx.push(Inst::Tuple3 {
                            args: [a, b, c],
                            output,
                        });
                        Ok(())
                    }),
                    &[a, b, c, d] => cx.addresses([a, b, c, d], [output], |cx, [a, b, c, d]| {
                        cx.push(Inst::Tuple4 {
                            args: [a, b, c, d],
                            output,
                        });
                        Ok(())
                    }),
                    args => cx.array(args, |cx, address, count| {
                        cx.push(Inst::Tuple {
                            address,
                            count,
                            output,
                        });
                        Ok(())
                    }),
                })?;

                ExprOutcome::Output(output)
            }
            ExprKind::Object { slot, args } => {
                let address = cx.alloc_or(output, |cx, output| {
                    cx.array(args, |cx, address, _| {
                        cx.push(Inst::Object {
                            address,
                            slot,
                            output,
                        });
                        Ok(())
                    })
                })?;

                ExprOutcome::Output(address)
            }
            ExprKind::AssignStructField { lhs, slot, rhs } => {
                let address = cx.alloc_or(output, |cx, output| {
                    cx.addresses([*lhs, *rhs], [output], |cx, [address, value]| {
                        cx.push(Inst::ObjectIndexSet {
                            address,
                            value,
                            slot,
                            output,
                        });
                        Ok(())
                    })
                })?;

                ExprOutcome::Output(address)
            }
            ExprKind::AssignTupleField { lhs, index, rhs } => {
                let address = cx.alloc_or(output, |cx, output| {
                    cx.addresses([*lhs, *rhs], [output], |cx, [address, value]| {
                        cx.push(Inst::TupleIndexSet {
                            address,
                            value,
                            index,
                            output,
                        });
                        Ok(())
                    })
                })?;

                ExprOutcome::Output(address)
            }
            ExprKind::Meta { meta, needs, named } => {
                compile_expr_meta(cx, meta, needs, named, output)?
            }
            ExprKind::Unary { op, expr } => {
                let address = cx.alloc_or(output, |cx, output| {
                    cx.addresses([*expr], [output], |cx, [address]| {
                        match op {
                            ExprUnOp::Neg => {
                                cx.push(Inst::Neg { address, output });
                            }
                            ExprUnOp::Not => {
                                cx.push(Inst::Not { address, output });
                            }
                        }

                        Ok(())
                    })
                })?;
                ExprOutcome::Output(address)
            }
            ExprKind::Binary { lhs, op, rhs } => compile_expr_binary(cx, lhs, op, rhs, output)?,
            ExprKind::Loop { hir } => compile_expr_loop(cx, hir, output)?,
            ExprKind::TupleFieldAccess { lhs, index } => {
                let address = cx.alloc_or(output, |cx, output| {
                    cx.addresses([*lhs], [output], |cx, [address]| {
                        cx.push(Inst::TupleIndexGet {
                            address,
                            index,
                            output,
                        });

                        Ok(())
                    })
                })?;

                ExprOutcome::Output(address)
            }
            ExprKind::StructFieldAccess { lhs, slot, .. } => {
                let address = cx.alloc_or(output, |cx, output| {
                    cx.addresses([*lhs], [output], |cx, [address]| {
                        cx.push(Inst::ObjectIndexGet {
                            address,
                            slot,
                            output,
                        });
                        Ok(())
                    })
                })?;

                ExprOutcome::Output(address)
            }
            ExprKind::CallAddress {
                address: function,
                args,
            } => {
                let address = cx.alloc_or(output, |cx, output| {
                    cx.array(args, |cx, address, count| {
                        cx.push(Inst::CallFn {
                            function,
                            address,
                            count,
                            output,
                        });
                        Ok(())
                    })
                })?;

                ExprOutcome::Output(address)
            }
            ExprKind::CallHash { hash, args } => compile_expr_call_hash(cx, hash, args, output)?,
            ExprKind::CallInstance { lhs, hash, args } => {
                compile_expr_call_instance(cx, lhs, hash, args, output)?
            }
            ExprKind::CallExpr { expr, args } => {
                let address = cx.alloc_or(output, |cx, output| {
                    expr.compile(cx, Some(output))?.free(cx)?;

                    cx.array(args, |cx, address, count| {
                        cx.push(Inst::CallFn {
                            function: output,
                            address,
                            count,
                            output,
                        });
                        Ok(())
                    })
                })?;

                ExprOutcome::Output(address)
            }
            ExprKind::StructFieldAccessGeneric { .. } => {
                return Err(cx.error(CompileErrorKind::ExpectedExpr));
            }
            ExprKind::Index { target, index } => {
                let address = cx.alloc_or(output, |cx, output| {
                    cx.addresses([*target, *index], [output], |cx, [address, index]| {
                        cx.push(Inst::IndexGet {
                            address,
                            index,
                            output,
                        });
                        Ok(())
                    })
                })?;

                ExprOutcome::Output(address)
            }
            ExprKind::Struct { kind, exprs } => {
                let output = cx.alloc_or(output, |cx, output| {
                    cx.array(exprs, |cx, address, _| {
                        match kind {
                            ExprStructKind::Anonymous { slot } => {
                                cx.push(Inst::Object {
                                    address,
                                    slot,
                                    output,
                                });
                            }
                            ExprStructKind::Unit { hash } => {
                                cx.push(Inst::UnitStruct { hash, output });
                            }
                            ExprStructKind::Struct { hash, slot } => {
                                cx.push(Inst::Struct {
                                    hash,
                                    address,
                                    slot,
                                    output,
                                });
                            }
                            ExprStructKind::StructVariant { hash, slot } => {
                                cx.push(Inst::StructVariant {
                                    hash,
                                    address,
                                    slot,
                                    output,
                                });
                            }
                        }

                        Ok(())
                    })
                })?;

                ExprOutcome::Output(output)
            }
            ExprKind::Range { from, limits, to } => {
                let output = cx.alloc_or(output, |cx, output| {
                    cx.addresses([*from, *to], [output], |cx, [from, to]| {
                        cx.push(Inst::Range {
                            from,
                            to,
                            limits,
                            output,
                        });
                        Ok(())
                    })
                })?;
                ExprOutcome::Output(output)
            }
            ExprKind::StringConcat { exprs } => {
                let address = cx.alloc_or(output, |cx, output| {
                    cx.array(exprs, |cx, address, count| {
                        cx.push(Inst::StringConcat {
                            address,
                            count,
                            size_hint: 0,
                            output,
                        });
                        Ok(())
                    })
                })?;

                ExprOutcome::Output(address)
            }
            ExprKind::Format { spec, expr } => {
                let address = cx.alloc_or(output, |cx, output| {
                    cx.addresses([*expr], [output], |cx, [address]| {
                        cx.push(Inst::Format {
                            address,
                            spec: *spec,
                            output,
                        });
                        Ok(())
                    })
                })?;

                ExprOutcome::Output(address)
            }
            ExprKind::Break { value } => {
                cx.with_scope(self.scope.id, |cx| compile_expr_break(cx, value))?
            }
            ExprKind::Continue { label } => {
                cx.with_scope(self.scope.id, |cx| compile_expr_continue(cx, label))?
            }
        };

        Ok(outcome)
    }
}

/// An expression kind.
#[derive(Debug, Clone, Copy)]
pub(crate) enum ExprKind<'hir> {
    /// An empty expression.
    Empty,
    /// A collection of conditions.
    Conditions {
        branches: &'hir [ConditionBranch<'hir>],
    },
    /// Matches.
    Matches {
        /// The expression being matched over.
        expr: &'hir Expr<'hir>,
        /// The address where the expression must be assembled.
        address: AssemblyAddress,
        /// A collection of matches.
        branches: &'hir [MatchBranch<'hir>],
    },
    /// Load a function.
    Function {
        hash: Hash,
    },
    /// Load a closure.
    Closure {
        /// The hash of the closure function to load.
        hash: Hash,
        /// Captures to this closure.
        captures: &'hir [Expr<'hir>],
    },
    /// Expressions that must be used to unpack the local value.
    Let {
        pat: &'hir Pat<'hir>,
        expr: &'hir Expr<'hir>,
    },
    /// An address expression.
    Address {
        binding: Option<Binding>,
        address: AssemblyAddress,
    },
    /// Allocate an optional value.
    Option {
        /// The value to allocate.
        value: Option<&'hir Expr<'hir>>,
    },
    /// Yield the given value.
    Yield {
        /// Yield the given expression.
        expr: Option<&'hir Expr<'hir>>,
    },
    /// Perform an await operation.
    Await {
        /// The expression to await.
        expr: &'hir Expr<'hir>,
    },
    /// Perform a try operation.
    Try {
        /// The expression to try.
        expr: &'hir Expr<'hir>,
    },
    /// Store a literal value.
    Store {
        value: InstValue,
    },
    /// Allocate a string from a slot.
    String {
        string: &'hir str,
    },
    /// Allocate bytes from a slot.
    Bytes {
        bytes: &'hir [u8],
    },
    /// Allocate a vector.
    Vec {
        items: &'hir [Expr<'hir>],
    },
    /// An anonymous tuple.
    Tuple {
        items: &'hir [Expr<'hir>],
    },
    /// Allocate an object.
    Object {
        slot: usize,
        args: &'hir [Expr<'hir>],
    },
    /// A tuple field access.
    TupleFieldAccess {
        lhs: &'hir Expr<'hir>,
        index: usize,
    },
    /// A struct field access where the index is the slot used.
    StructFieldAccess {
        lhs: &'hir Expr<'hir>,
        slot: usize,
        hash: Hash,
    },
    StructFieldAccessGeneric {
        lhs: &'hir Expr<'hir>,
        hash: Hash,
        generics: Option<(Span, &'hir [hir::Expr<'hir>])>,
    },
    /// The `<target>[<value>]` operation.
    Index {
        target: &'hir Expr<'hir>,
        index: &'hir Expr<'hir>,
    },
    /// An address assignment.
    Assign {
        /// Address to assign to.
        address: AssemblyAddress,
        /// The expression to assign.
        rhs: &'hir Expr<'hir>,
    },
    AssignStructField {
        lhs: &'hir Expr<'hir>,
        slot: usize,
        rhs: &'hir Expr<'hir>,
    },
    AssignTupleField {
        lhs: &'hir Expr<'hir>,
        index: usize,
        rhs: &'hir Expr<'hir>,
    },
    Meta {
        meta: &'hir PrivMeta,
        needs: Needs,
        named: &'hir Named<'hir>,
    },
    Loop {
        /// The loop expression to assemble.
        hir: &'hir hir::ExprLoop<'hir>,
    },
    /// A partial translated unary expression. We perform partial translation in
    /// phase one since unary expressions are also valid literals, like `-1`.
    Unary {
        op: ExprUnOp,
        expr: &'hir Expr<'hir>,
    },
    /// A binary expression.
    Binary {
        /// The left-hand side of a binary operation.
        lhs: &'hir Expr<'hir>,
        /// The operator.
        op: ast::BinOp,
        /// The right-hand side of a binary operation.
        rhs: &'hir Expr<'hir>,
    },
    /// Return a kind.
    Return {
        expr: &'hir Expr<'hir>,
    },
    CallAddress {
        address: AssemblyAddress,
        args: &'hir [Expr<'hir>],
    },
    CallHash {
        hash: Hash,
        args: &'hir [Expr<'hir>],
    },
    CallInstance {
        lhs: &'hir Expr<'hir>,
        hash: Hash,
        args: &'hir [Expr<'hir>],
    },
    CallExpr {
        expr: &'hir Expr<'hir>,
        args: &'hir [Expr<'hir>],
    },
    Struct {
        kind: ExprStructKind,
        exprs: &'hir [Expr<'hir>],
    },
    Range {
        from: &'hir Expr<'hir>,
        limits: InstRangeLimits,
        to: &'hir Expr<'hir>,
    },
    StringConcat {
        exprs: &'hir [Expr<'hir>],
    },
    Format {
        spec: &'hir FormatSpec,
        expr: &'hir Expr<'hir>,
    },
    Break {
        value: &'hir ExprBreakValue<'hir>,
    },
    Continue {
        label: Option<BindingName>,
    },
}

/// Helper wrapper around a scope ensuring that it can be correctly retained
/// while an address in that scope might still be in use.
#[derive(Debug, Clone, Copy, Default)]
#[must_use = "must be freed with ExprScope::free"]
pub(crate) struct ExprScope {
    id: ScopeId,
    free: bool,
}

impl ExprScope {
    /// Free the current scope if its intended to be freed.
    pub(crate) fn free(self, cx: &mut Ctxt<'_, '_>) -> Result<()> {
        if self.free {
            tracing::trace!(?self.id, "freeing scope");
            cx.scopes.pop(cx.span, self.id)?;
        }

        Ok(())
    }
}

// #[must_use = "Code generation can be elided if reachability is taken into account"]
#[derive(Debug, Clone)]
#[must_use = "ExprOutcome::free must be called"]
pub(crate) enum ExprOutcome {
    /// The expression produced no value.
    Empty,
    /// The expression produced output.
    Output(MaybeAlloc),
    /// The expression produced a value which is unreachable.
    Unreachable,
}

impl ExprOutcome {
    /// Free the current outcome.
    pub(crate) fn free(self, cx: &mut Ctxt<'_, '_>) -> Result<()> {
        if let ExprOutcome::Output(output) = self {
            output.free(cx)?;
        }

        Ok(())
    }

    /// Expect an address out of an outcome.
    pub(crate) fn ensure_address(self, cx: &mut Ctxt<'_, '_>) -> Result<MaybeAlloc> {
        if let ExprOutcome::Output(output) = self {
            Ok(output)
        } else {
            Ok(MaybeAlloc::allocated(cx.scopes.alloc()))
        }
    }
}

/// Things that we can break on.
#[derive(Debug, Clone, Copy)]
pub(crate) enum ExprBreakValue<'hir> {
    /// Empty break value.
    None,
    /// Breaking a value out of a loop.
    Expr(&'hir Expr<'hir>),
    /// Break and jump to the given label.
    Label(BindingName),
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum ExprStructKind {
    /// An anonymous struct.
    Anonymous { slot: usize },
    /// A unit struct.
    Unit { hash: Hash },
    /// A struct with named fields.
    Struct { hash: Hash, slot: usize },
    /// A variant struct with named fields.
    StructVariant { hash: Hash, slot: usize },
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum ExprUnOp {
    Neg,
    Not,
}

/// Compile a match expression.
#[instrument]
fn compile_expr_conditions<'hir>(
    cx: &mut Ctxt<'_, 'hir>,
    branches: &'hir [ConditionBranch<'hir>],
    output: Option<AssemblyAddress>,
) -> Result<ExprOutcome> {
    let address = cx.alloc_or(output, |cx, output| {
        let end = cx.new_label("condition_end");
        let mut checkpoint = Some(cx.state);

        let mut it = branches.iter().peekable();
        let mut first = true;

        while let Some(branch) = it.next() {
            let first = mem::take(&mut first);

            let (pat_outcome, expr_outcome) = cx.with_span(branch.span, |cx| {
                let label = if it.peek().is_none() {
                    end
                } else {
                    cx.new_label("condition_branch_end")
                };

                let pat_outcome = branch.pat.compile(cx, label)?;
                let expr_outcome =
                    cx.with_state_checkpoint(|cx| branch.body.compile(cx, Some(output)))?;

                if matches!(pat_outcome, PatOutcome::Refutable) && end != label {
                    cx.push(Inst::Jump { label: end });
                }

                if label != end {
                    cx.label(label)?;
                }

                Ok((pat_outcome, expr_outcome))
            })?;

            if let PatOutcome::Irrefutable = pat_outcome {
                // NB: End label *might* be used, so before we mark as
                // unreachable we need to provide it here since the last end
                // label might otherwise be supressed.
                cx.label(end)?;
                cx.state = CtxtState::Unreachable { reported: false };

                // If the *expression* was unreachable and this is the first
                // (and only reachable) expression mark this control flow as
                // unreachable as well.
                if first && matches!(expr_outcome, ExprOutcome::Unreachable) {
                    checkpoint = None;
                }
            }

            expr_outcome.free(cx)?;
        }

        cx.label(end)?;

        if let Some(state) = checkpoint {
            cx.state = state;
        }

        Ok(())
    })?;

    Ok(ExprOutcome::Output(address))
}

/// Compile a match expression.
#[instrument]
fn compile_expr_matches<'hir>(
    cx: &mut Ctxt<'_, 'hir>,
    expr: &'hir Expr<'hir>,
    address: AssemblyAddress,
    branches: &'hir [MatchBranch<'hir>],
    output: Option<AssemblyAddress>,
) -> Result<ExprOutcome> {
    let address = cx.alloc_or(output, |cx, output| {
        let end = cx.new_label("match_end");
        let mut checkpoint = Some(cx.state);

        let mut it = branches.iter().peekable();
        let mut first = true;

        let match_expr_outcome = expr.compile(cx, Some(address))?;

        while let Some(branch) = it.next() {
            let first = mem::take(&mut first);

            let (pat_outcome, expr_outcome) = cx.with_span(branch.span, |cx| {
                let label = if it.peek().is_none() {
                    end
                } else {
                    cx.new_label("match_branch_end")
                };

                let mut pat_outcome = branch.pat.compile(cx, label)?;
                let expr_outcome = branch.condition.compile(cx, None)?;

                if let ExprOutcome::Output(address) = expr_outcome {
                    cx.push(Inst::JumpIfNot {
                        address: *address,
                        label,
                    });
                    pat_outcome = PatOutcome::Refutable;
                }

                expr_outcome.free(cx)?;

                let expr_outcome = cx.with_state_checkpoint(|cx| {
                    let expr_outcome = branch.body.compile(cx, Some(output))?;

                    if matches!(pat_outcome, PatOutcome::Refutable) && end != label {
                        cx.push(Inst::Jump { label: end });
                    }

                    Ok(expr_outcome)
                })?;

                if label != end {
                    cx.label(label)?;
                }

                Ok((pat_outcome, expr_outcome))
            })?;

            if let PatOutcome::Irrefutable = pat_outcome {
                // NB: End label *might* be used, so before we mark as
                // unreachable we need to provide it here since the last end
                // label might otherwise be supressed.
                cx.label(end)?;
                cx.state = CtxtState::Unreachable { reported: false };

                // If the *expression* was unreachable and this is the first
                // (and only reachable) expression mark this control flow as
                // unreachable as well.
                if first && matches!(expr_outcome, ExprOutcome::Unreachable) {
                    checkpoint = None;
                }
            }
        }

        cx.label(end)?;

        if let Some(state) = checkpoint {
            cx.state = state;
        }

        match_expr_outcome.free(cx)?;
        // At this point, the match address is guaranteed to no longer be used
        // since we've assembled all branches which reference it.
        cx.scopes.free(cx.span, address)?;
        Ok(())
    })?;

    Ok(ExprOutcome::Output(address))
}

/// Assembling of a binary expression.
#[instrument]
fn compile_expr_binary<'hir>(
    cx: &mut Ctxt<'_, 'hir>,
    lhs: &'hir Expr<'hir>,
    op: ast::BinOp,
    rhs: &'hir Expr<'hir>,
    output: Option<AssemblyAddress>,
) -> Result<ExprOutcome> {
    if op.is_assign() {
        return compile_assign_binop(cx, lhs, rhs, op, output);
    }

    if op.is_conditional() {
        return compile_conditional_binop(cx, lhs, rhs, op, output);
    }

    let op = match op {
        ast::BinOp::Eq(..) => InstOp::Eq,
        ast::BinOp::Neq(..) => InstOp::Neq,
        ast::BinOp::Lt(..) => InstOp::Lt,
        ast::BinOp::Gt(..) => InstOp::Gt,
        ast::BinOp::Lte(..) => InstOp::Lte,
        ast::BinOp::Gte(..) => InstOp::Gte,
        ast::BinOp::Is(..) => InstOp::Is,
        ast::BinOp::IsNot(..) => InstOp::IsNot,
        ast::BinOp::And(..) => InstOp::And,
        ast::BinOp::Or(..) => InstOp::Or,
        ast::BinOp::Add(..) => InstOp::Add,
        ast::BinOp::Sub(..) => InstOp::Sub,
        ast::BinOp::Div(..) => InstOp::Div,
        ast::BinOp::Mul(..) => InstOp::Mul,
        ast::BinOp::Rem(..) => InstOp::Rem,
        ast::BinOp::BitAnd(..) => InstOp::BitAnd,
        ast::BinOp::BitXor(..) => InstOp::BitXor,
        ast::BinOp::BitOr(..) => InstOp::BitOr,
        ast::BinOp::Shl(..) => InstOp::Shl,
        ast::BinOp::Shr(..) => InstOp::Shr,
        op => {
            return Err(cx.error(CompileErrorKind::UnsupportedBinaryOp { op }));
        }
    };

    let address = cx.alloc_or(output, |cx, output| {
        cx.addresses([*lhs, *rhs], [output], |cx, [a, b]| {
            cx.push(Inst::Op { op, a, b, output });
            Ok(())
        })
    })?;

    return Ok(ExprOutcome::Output(address));

    fn compile_conditional_binop<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        lhs: &'hir Expr<'hir>,
        rhs: &'hir Expr<'hir>,
        op: ast::BinOp,
        output: Option<AssemblyAddress>,
    ) -> Result<ExprOutcome> {
        let address = cx.alloc_or(output, |cx, output| {
            let end_label = cx.new_label("conditional_end");
            lhs.compile(cx, Some(output))?.free(cx)?;

            match op {
                ast::BinOp::And(..) => {
                    cx.push(Inst::JumpIfNot {
                        address: output,
                        label: end_label,
                    });
                }
                ast::BinOp::Or(..) => {
                    cx.push(Inst::JumpIf {
                        address: output,
                        label: end_label,
                    });
                }
                op => {
                    return Err(cx.error(CompileErrorKind::UnsupportedBinaryOp { op }));
                }
            }

            rhs.compile(cx, Some(output))?.free(cx)?;
            cx.label(end_label)?;
            Ok(())
        })?;

        Ok(ExprOutcome::Output(address))
    }

    fn compile_assign_binop<'hir>(
        cx: &mut Ctxt<'_, 'hir>,
        lhs: &'hir Expr<'hir>,
        rhs: &'hir Expr<'hir>,
        op: ast::BinOp,
        output: Option<AssemblyAddress>,
    ) -> Result<ExprOutcome> {
        let address = cx.alloc_or(output, |cx, output| {
            let (lhs, rhs, target) = match lhs.kind {
                ExprKind::Address { address, .. } => {
                    rhs.compile(cx, Some(output))?.free(cx)?;
                    (address, output, InstTarget::Offset)
                }
                _ => {
                    return Err(cx.error(CompileErrorKind::UnsupportedBinaryExpr));
                }
            };

            let op = match op {
                ast::BinOp::AddAssign(..) => InstAssignOp::Add,
                ast::BinOp::SubAssign(..) => InstAssignOp::Sub,
                ast::BinOp::MulAssign(..) => InstAssignOp::Mul,
                ast::BinOp::DivAssign(..) => InstAssignOp::Div,
                ast::BinOp::RemAssign(..) => InstAssignOp::Rem,
                ast::BinOp::BitAndAssign(..) => InstAssignOp::BitAnd,
                ast::BinOp::BitXorAssign(..) => InstAssignOp::BitXor,
                ast::BinOp::BitOrAssign(..) => InstAssignOp::BitOr,
                ast::BinOp::ShlAssign(..) => InstAssignOp::Shl,
                ast::BinOp::ShrAssign(..) => InstAssignOp::Shr,
                _ => {
                    return Err(cx.error(CompileErrorKind::UnsupportedBinaryExpr));
                }
            };

            let output = cx.scopes.temporary();

            cx.push(Inst::Assign {
                lhs,
                rhs,
                target,
                op,
                // NB: while an assign operation doesn't output anything, this
                // might result in a call to an external function which expects
                // to always have a writable address slot available, regardless
                // of what it outputs. So we hand it a temporary slot, which is
                // just an address on the stack which will probably immediately
                // be used for something else.
                //
                // The virtual machine ensures that this address is cleared
                // immediately after it's been called.
                output,
            });

            Ok(())
        })?;

        Ok(ExprOutcome::Output(address))
    }
}

/// Compile a break expression.
#[instrument]
fn compile_expr_break<'hir>(
    cx: &mut Ctxt<'_, 'hir>,
    value: &'hir ExprBreakValue<'hir>,
) -> Result<ExprOutcome> {
    match value {
        ExprBreakValue::None => {
            if let Some(ControlFlow::Loop(control_flow)) = cx
                .scopes
                .find_ancestor(cx.scope, |flow| matches!(flow, ControlFlow::Loop { .. }))
            {
                cx.push(Inst::Jump {
                    label: control_flow.end,
                });
            } else {
                return Err(cx.error(CompileErrorKind::BreakOutsideOfLoop));
            }
        }
        ExprBreakValue::Expr(expr) => {
            if let Some(ControlFlow::Loop(control_flow)) = cx
                .scopes
                .find_ancestor(cx.scope, |flow| matches!(flow, ControlFlow::Loop { .. }))
            {
                expr.compile(cx, control_flow.output)?.free(cx)?;
                cx.push(Inst::Jump {
                    label: control_flow.end,
                });
            } else {
                return Err(cx.error(CompileErrorKind::BreakOutsideOfLoop));
            }
        }
        ExprBreakValue::Label(expected) => {
            if let Some(ControlFlow::Loop(control_flow)) = cx.scopes.find_ancestor(
                cx.scope,
                |flow| matches!(flow, ControlFlow::Loop(l) if l.label.as_ref() == Some(expected)),
            ) {
                cx.push(Inst::Jump {
                    label: control_flow.end,
                });
            } else {
                let name = cx.scopes.name(cx.span, *expected)?;
                return Err(cx.error(CompileErrorKind::MissingLabel { label: name.into() }));
            }
        }
    }

    Ok(ExprOutcome::Empty)
}

/// Compile a continue expression.
#[instrument]
fn compile_expr_continue<'hir>(
    cx: &mut Ctxt<'_, 'hir>,
    label: Option<BindingName>,
) -> Result<ExprOutcome> {
    let control_flow = match label {
        Some(expected) => cx.scopes.find_ancestor(
            cx.scope,
            |flow| matches!(flow, ControlFlow::Loop(l) if l.label == Some(expected)),
        ),
        None => cx
            .scopes
            .find_ancestor(cx.scope, |flow| matches!(flow, ControlFlow::Loop { .. })),
    };

    if let Some(ControlFlow::Loop(control_flow)) = control_flow {
        cx.push(Inst::Jump {
            label: control_flow.start,
        });

        return Ok(ExprOutcome::Unreachable);
    }

    if let Some(label) = label {
        let label = cx.scopes.name(cx.span, label)?;
        Err(cx.error(CompileErrorKind::MissingLabel {
            label: label.into(),
        }))
    } else {
        Err(cx.error(CompileErrorKind::ContinueOutsideOfLoop))
    }
}

/// Compile a loop expression.
#[instrument]
fn compile_expr_loop<'hir>(
    cx: &mut Ctxt<'_, 'hir>,
    hir: &'hir hir::ExprLoop<'hir>,
    output: Option<AssemblyAddress>,
) -> Result<ExprOutcome> {
    let label = match hir.label {
        Some(label) => Some(
            cx.scopes
                .binding_name(label.resolve(resolve_context!(cx.q))?),
        ),
        None => None,
    };

    let start = cx.new_label("loop_start");
    let end = cx.new_label("loop_end");
    let scope = cx
        .scopes
        .push_loop(cx.span, Some(cx.scope), label, start, end, output)?;

    let (cleanup, checkpoint) = match hir.condition {
        hir::LoopCondition::Forever => {
            cx.label(start)?;
            // NB: these loops are guaranteed to be entered at least once, so no
            // need to utilize a state checkpoint here.
            (None, None)
        }
        hir::LoopCondition::Condition { condition } => {
            cx.label(start)?;

            let (expr, pat) = match *condition {
                hir::Condition::Expr(hir) => {
                    let lit = alloc!(cx; cx.expr(ExprKind::Store { value: InstValue::Bool(true) }));
                    (assemble_expr_value(cx, hir)?, cx.pat(PatKind::Lit { lit }))
                }
                hir::Condition::ExprLet(hir) => {
                    let expr = assemble_expr_value(cx, hir.expr)?;
                    let pat = cx.with_scope(scope, |cx| assemble_pat(cx, hir.pat))?;
                    (expr, pat)
                }
            };

            pat.bind(cx, expr)?.compile(cx, end)?;
            (None, Some(cx.state))
        }
        hir::LoopCondition::Iterator { binding, iter } => {
            let iter_var = assemble_expr_value(cx, iter)?
                .compile(cx, None)?
                .ensure_address(cx)?;

            cx.push(Inst::CallInstance {
                hash: *Protocol::INTO_ITER,
                address: *iter_var,
                count: 0,
                output: *iter_var,
            });

            cx.label(start)?;

            let value_var = cx.scopes.alloc();

            cx.push(Inst::IterNext {
                address: *iter_var,
                label: end,
                output: value_var,
            });

            let expr = cx.expr(ExprKind::Address {
                address: value_var,
                binding: None,
            });

            cx.with_scope(scope, |cx| {
                assemble_pat(cx, binding)?.bind(cx, expr)?.compile(cx, end)
            })?;
            (Some((iter_var, value_var)), Some(cx.state))
        }
    };

    let outcome = cx
        .with_scope(scope, |cx| assemble_block(cx, hir.body))?
        .compile(cx, output)?;
    cx.push(Inst::Jump { label: start });

    if let Some(state) = checkpoint {
        cx.state = state;
    }

    cx.label(end)?;

    if let Some((a, b)) = cleanup {
        a.free(cx)?;
        cx.scopes.free(cx.span, b)?;
    }

    cx.scopes.pop(cx.span, scope)?;
    Ok(outcome)
}

/// Compile an item.
#[instrument]
fn compile_expr_meta<'hir>(
    cx: &mut Ctxt<'_, 'hir>,
    meta: &PrivMeta,
    needs: Needs,
    named: &'hir Named<'hir>,
    output: Option<AssemblyAddress>,
) -> Result<ExprOutcome> {
    let outcome = if let Needs::Value = needs {
        match &meta.kind {
            PrivMetaKind::Struct {
                type_hash,
                variant: PrivVariantMeta::Unit,
                ..
            }
            | PrivMetaKind::Variant {
                type_hash,
                variant: PrivVariantMeta::Unit,
                ..
            } => {
                named.assert_not_generic()?;

                let address = cx.alloc_or(output, |cx, output| {
                    cx.push_with_comment(
                        Inst::Call {
                            hash: *type_hash,
                            address: output,
                            count: 0,
                            output,
                        },
                        meta.info(cx.q.pool),
                    );
                    Ok(())
                })?;

                ExprOutcome::Output(address)
            }
            PrivMetaKind::Variant {
                variant: PrivVariantMeta::Tuple(tuple),
                ..
            }
            | PrivMetaKind::Struct {
                variant: PrivVariantMeta::Tuple(tuple),
                ..
            } if tuple.args == 0 => {
                named.assert_not_generic()?;

                let address = cx.alloc_or(output, |cx, output| {
                    cx.push_with_comment(
                        Inst::Call {
                            hash: tuple.hash,
                            address: output,
                            count: 0,
                            output,
                        },
                        meta.info(cx.q.pool),
                    );
                    Ok(())
                })?;

                ExprOutcome::Output(address)
            }
            PrivMetaKind::Struct {
                variant: PrivVariantMeta::Tuple(tuple),
                ..
            } => {
                named.assert_not_generic()?;

                let address = cx.alloc_or(output, |cx, output| {
                    cx.push_with_comment(
                        Inst::LoadFn {
                            hash: tuple.hash,
                            output,
                        },
                        meta.info(cx.q.pool),
                    );
                    Ok(())
                })?;

                ExprOutcome::Output(address)
            }
            PrivMetaKind::Variant {
                variant: PrivVariantMeta::Tuple(tuple),
                ..
            } => {
                named.assert_not_generic()?;

                let address = cx.alloc_or(output, |cx, output| {
                    cx.push_with_comment(
                        Inst::LoadFn {
                            hash: tuple.hash,
                            output,
                        },
                        meta.info(cx.q.pool),
                    );
                    Ok(())
                })?;

                ExprOutcome::Output(address)
            }
            PrivMetaKind::Function { type_hash, .. } => {
                let hash = if let Some((span, generics)) = named.generics {
                    let parameters = cx.with_span(span, |cx| generics_parameters(cx, generics))?;
                    type_hash.with_parameters(parameters)
                } else {
                    *type_hash
                };

                let output = cx.alloc_or(output, |cx, output| {
                    cx.push_with_comment(Inst::LoadFn { hash, output }, meta.info(cx.q.pool));
                    Ok(())
                })?;

                ExprOutcome::Output(output)
            }
            PrivMetaKind::Const { const_value, .. } => {
                named.assert_not_generic()?;
                assemble_expr_const(cx, const_value)?.compile(cx, output)?
            }
            _ => {
                return Err(cx.error(CompileErrorKind::ExpectedMeta {
                    meta: meta.info(cx.q.pool),
                    expected: "something that can be used as a value",
                }));
            }
        }
    } else {
        named.assert_not_generic()?;

        let type_hash = match meta.type_hash_of() {
            Some(type_hash) => type_hash,
            None => {
                return Err(cx.error(CompileErrorKind::ExpectedMeta {
                    meta: meta.info(cx.q.pool),
                    expected: "something that has a type",
                }));
            }
        };

        let output = cx.alloc_or(output, |cx, output| {
            cx.push(Inst::Store {
                value: InstValue::Type(type_hash),
                output,
            });
            Ok(())
        })?;

        ExprOutcome::Output(output)
    };

    Ok(outcome)
}

#[instrument]
fn compile_expr_call_hash<'hir>(
    cx: &mut Ctxt<'_, 'hir>,
    hash: Hash,
    args: &[Expr<'hir>],
    output: Option<AssemblyAddress>,
) -> Result<ExprOutcome> {
    let output = cx.alloc_or(output, |cx, output| {
        cx.array(args, |cx, address, count| {
            cx.push(Inst::Call {
                hash,
                address,
                count,
                output,
            });
            Ok(())
        })
    })?;

    Ok(ExprOutcome::Output(output))
}

#[instrument]
fn compile_expr_call_instance<'hir>(
    cx: &mut Ctxt<'_, 'hir>,
    lhs: &Expr<'hir>,
    hash: Hash,
    args: &[Expr<'hir>],
    output: Option<AssemblyAddress>,
) -> Result<ExprOutcome> {
    let address = cx.scopes.array_index();

    {
        let output = cx.scopes.array_index();
        lhs.compile(cx, Some(output))?.free(cx)?;
        cx.scopes.alloc_array_item();
    }

    for hir in args {
        let output = cx.scopes.array_index();
        hir.compile(cx, Some(output))?.free(cx)?;
        cx.scopes.alloc_array_item();
    }

    let address = cx.alloc_or(output, |cx, output| {
        cx.push(Inst::CallInstance {
            hash,
            address,
            count: args.len(),
            output,
        });
        Ok(())
    })?;

    cx.scopes.free_array(cx.span, args.len())?;
    Ok(ExprOutcome::Output(address))
}
