use crate::arena::AllocIter;
use crate::ast::Span;
use crate::compile::assemble::{
    arena_error, arena_slice_write_error, assemble_expr_value, assemble_pat, Ctxt, Expr, ExprKind,
    Pat, PatKind, Result,
};
use crate::hir;
use crate::runtime::InstValue;

#[derive(Debug, Clone, Copy)]
pub(crate) struct ConditionBranch<'hir> {
    pub(crate) span: Span,
    pub(crate) pat: Pat<'hir>,
    pub(crate) condition: Expr<'hir>,
    pub(crate) body: hir::Block<'hir>,
}

pub(crate) struct Conditions<'hir> {
    branches: AllocIter<'hir, ConditionBranch<'hir>>,
}

impl<'hir> Conditions<'hir> {
    /// Construct a new set of matches.
    pub(crate) fn new(cx: &mut Ctxt<'_, 'hir>, len: usize) -> Result<Self> {
        Ok(Self {
            branches: cx.arena.alloc_iter(len).map_err(arena_error(cx.span))?,
        })
    }

    /// Assemble conditions into an expression.
    pub(crate) fn assemble(self, cx: &mut Ctxt<'_, 'hir>) -> Result<Expr<'hir>> {
        Ok(cx.expr(ExprKind::Conditions {
            branches: self.branches.finish(),
        }))
    }

    /// Add a branch.
    pub(crate) fn add_branch(
        &mut self,
        cx: &mut Ctxt<'_, 'hir>,
        condition: Option<&'hir hir::Condition<'hir>>,
        body: &'hir hir::Block<'hir>,
    ) -> Result<()> {
        let (condition, pat) = match condition {
            Some(hir::Condition::Expr(hir)) => (assemble_expr_value(cx, hir)?, None),
            Some(hir::Condition::ExprLet(hir)) => (
                assemble_expr_value(cx, hir.expr)?,
                Some(assemble_pat(cx, hir.pat)?),
            ),
            None => {
                let condition = cx.expr(ExprKind::Store {
                    value: InstValue::Bool(true),
                });

                (condition, None)
            }
        };

        let pat = match pat {
            Some(hir) => hir,
            None => {
                let lit = alloc!(cx; cx.expr(ExprKind::Store { value: InstValue::Bool(true) }));
                cx.pat(PatKind::Lit { lit })
            }
        };

        let branch = ConditionBranch {
            span: body.span,
            pat,
            condition,
            body: *body,
        };

        self.branches
            .write(branch)
            .map_err(arena_slice_write_error(cx.span))?;

        Ok(())
    }
}
