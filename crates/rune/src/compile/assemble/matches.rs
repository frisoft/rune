use crate::arena::AllocIter;
use crate::ast::Span;
use crate::compile::assemble::{
    arena_error, arena_slice_write_error, assemble_expr_value, assemble_pat, BoundPat, Ctxt, Expr,
    ExprKind, PatKind, Result,
};
use crate::compile::AssemblyAddress;
use crate::hir;

#[derive(Debug, Clone, Copy)]
pub(crate) struct MatchBranch<'hir> {
    pub(crate) span: Span,
    pub(crate) pat: BoundPat<'hir>,
    pub(crate) condition: Expr<'hir>,
    pub(crate) body: Expr<'hir>,
}

pub(crate) struct Matches<'hir> {
    expr: Expr<'hir>,
    address: AssemblyAddress,
    branches: AllocIter<'hir, MatchBranch<'hir>>,
}

impl<'hir> Matches<'hir> {
    /// Construct a new set of matches.
    pub(crate) fn new(
        cx: &mut Ctxt<'_, 'hir>,
        len: usize,
        expr: &'hir hir::Expr<'hir>,
    ) -> Result<Self> {
        Ok(Self {
            expr: assemble_expr_value(cx, expr)?,
            address: cx.scopes.alloc(),
            branches: cx.arena.alloc_iter(len).map_err(arena_error(cx.span))?,
        })
    }

    /// Assemble matches into an expression.
    pub(crate) fn assemble(self, cx: &mut Ctxt<'_, 'hir>) -> Result<Expr<'hir>> {
        Ok(cx.expr(ExprKind::Matches {
            expr: alloc!(cx; self.expr),
            address: self.address,
            branches: self.branches.finish(),
        }))
    }

    /// Add a branch.
    pub(crate) fn add_branch(
        &mut self,
        cx: &mut Ctxt<'_, 'hir>,
        pat: Option<&'hir hir::Pat<'hir>>,
        condition: Option<&'hir hir::Expr<'hir>>,
        body: &'hir hir::Expr<'hir>,
    ) -> Result<()> {
        let pat = match pat {
            Some(hir) => assemble_pat(cx, hir)?,
            None => cx.pat(PatKind::Ignore),
        };

        let condition = match condition {
            Some(hir) => hir,
            None => {
                alloc!(cx; hir::Expr { span: cx.span, kind: hir::ExprKind::Empty })
            }
        };

        let scope = cx.scopes.push_branch(cx.span, Some(cx.scope))?;

        let (pat, condition, body) = cx.with_scope(scope, |cx| {
            let branch_expr = cx.expr(ExprKind::Address {
                binding: None,
                address: self.address,
            });

            let pat = pat.bind(cx, branch_expr)?;
            let condition = assemble_expr_value(cx, condition)?;
            let body = assemble_expr_value(cx, body)?.free_scope();
            Ok((pat, condition, body))
        })?;

        let branch = MatchBranch {
            span: body.span,
            pat,
            condition,
            body,
        };

        self.branches
            .write(branch)
            .map_err(arena_slice_write_error(cx.span))?;

        Ok(())
    }
}
