use hir::FnArg;

use crate::ast;
use crate::hir;
use crate::hir::HirError;

pub struct Ctx<'hir> {
    /// Arena used for allocations.
    arena: &'hir hir::arena::Arena,
}

impl<'hir> Ctx<'hir> {
    /// Construct a new context.
    pub fn new(arena: &'hir hir::arena::Arena) -> Self {
        Self { arena }
    }
}

/// Lower an identifier.
fn ident<'hir>(ctx: &Ctx<'hir>, ast: &ast::Ident) -> Result<hir::Ident, HirError> {
    Ok(hir::Ident { source: ast.source })
}

/// Lower a function item.
pub fn item_fn<'hir>(ctx: &Ctx<'hir>, ast: &ast::ItemFn) -> Result<hir::ItemFn<'hir>, HirError> {
    Ok(hir::ItemFn {
        id: ast.id,
        attributes: ctx
            .arena
            .alloc_iter(ast.attributes.iter().map(|ast| attribute(ctx, ast)))?,
        visibility: ctx.arena.alloc(match &ast.visibility {
            ast::Visibility::Inherited => hir::Visibility::Inherited,
            ast::Visibility::Public(_) => hir::Visibility::Public,
            ast::Visibility::Crate(_) => hir::Visibility::Crate,
            ast::Visibility::Super(_) => hir::Visibility::Super,
            ast::Visibility::SelfValue(_) => hir::Visibility::SelfValue,
            ast::Visibility::In(ast) => {
                hir::Visibility::In(ctx.arena.alloc(path(ctx, &ast.restriction.path)?)?)
            }
        })?,
        name: ctx.arena.alloc(ident(ctx, &ast.name)?)?,
        args: ctx
            .arena
            .alloc_iter(ast.args.iter().map(|(ast, _)| fn_arg(ctx, ast)))?,
        body: ctx.arena.alloc(block(ctx, &ast.body)?)?,
    })
}

/// Lower a function argument.
fn fn_arg<'hir>(ctx: &Ctx<'hir>, ast: &ast::FnArg) -> Result<FnArg<'hir>, HirError> {
    Ok(match ast {
        ast::FnArg::SelfValue(_) => hir::FnArg::SelfValue,
        ast::FnArg::Pat(ast) => hir::FnArg::Pat(ctx.arena.alloc(pat(ctx, ast)?)?),
    })
}

/// Lower an expression.
fn expr<'hir>(ctx: &Ctx<'hir>, ast: &ast::Expr) -> Result<hir::Expr<'hir>, HirError> {
    match ast {
        ast::Expr::Path(ast) => Ok(hir::Expr::Path(ctx.arena.alloc(path(ctx, ast)?)?)),
        ast::Expr::Assign(_) => todo!(),
        ast::Expr::While(_) => todo!(),
        ast::Expr::Loop(_) => todo!(),
        ast::Expr::For(_) => todo!(),
        ast::Expr::Let(_) => todo!(),
        ast::Expr::If(_) => todo!(),
        ast::Expr::Match(_) => todo!(),
        ast::Expr::Call(_) => todo!(),
        ast::Expr::FieldAccess(_) => todo!(),
        ast::Expr::Group(_) => todo!(),
        ast::Expr::Empty(_) => todo!(),
        ast::Expr::Binary(_) => todo!(),
        ast::Expr::Unary(_) => todo!(),
        ast::Expr::Index(_) => todo!(),
        ast::Expr::Break(_) => todo!(),
        ast::Expr::Continue(_) => todo!(),
        ast::Expr::Yield(_) => todo!(),
        ast::Expr::Block(_) => todo!(),
        ast::Expr::Return(_) => todo!(),
        ast::Expr::Await(_) => todo!(),
        ast::Expr::Try(_) => todo!(),
        ast::Expr::Select(_) => todo!(),
        ast::Expr::Closure(_) => todo!(),
        ast::Expr::Lit(_) => todo!(),
        ast::Expr::ForceSemi(_) => todo!(),
        ast::Expr::MacroCall(_) => todo!(),
        ast::Expr::Object(_) => todo!(),
        ast::Expr::Tuple(_) => todo!(),
        ast::Expr::Vec(_) => todo!(),
        ast::Expr::Range(_) => todo!(),
    }
}

fn pat<'hir>(ctx: &Ctx<'hir>, ast: &ast::Pat) -> Result<hir::Pat<'hir>, HirError> {
    Ok(match ast {
        ast::Pat::PatIgnore(_) => hir::Pat::PatIgnore,
        ast::Pat::PatPath(ast) => hir::Pat::PatPath(todo!()),
        ast::Pat::PatLit(_) => todo!(),
        ast::Pat::PatVec(_) => todo!(),
        ast::Pat::PatTuple(_) => todo!(),
        ast::Pat::PatObject(_) => todo!(),
        ast::Pat::PatBinding(_) => todo!(),
        ast::Pat::PatRest(_) => todo!(),
    })
}

fn path<'hir>(ctx: &Ctx<'hir>, ast: &ast::Path) -> Result<hir::Path<'hir>, HirError> {
    let iter = ast.rest.iter().map(|(_, s)| path_segment(ctx, s));

    Ok(hir::Path {
        id: ast.id,
        first: path_segment(ctx, &ast.first)?,
        rest: ctx.arena.alloc_iter(iter)?,
    })
}

fn path_segment<'hir>(
    ctx: &Ctx<'hir>,
    ast: &ast::PathSegment,
) -> Result<hir::PathSegment<'hir>, HirError> {
    Ok(match ast {
        ast::PathSegment::SelfType(..) => hir::PathSegment::SelfType,
        ast::PathSegment::SelfValue(..) => hir::PathSegment::SelfValue,
        ast::PathSegment::Ident(ast) => hir::PathSegment::Ident(ctx.arena.alloc(ident(ctx, ast)?)?),
        ast::PathSegment::Crate(..) => hir::PathSegment::Crate,
        ast::PathSegment::Super(..) => hir::PathSegment::Super,
        ast::PathSegment::Generics(ast) => {
            let iter = ast.iter().map(|(e, _)| expr(ctx, &e.expr));
            hir::PathSegment::Generics(ctx.arena.alloc_iter(iter)?)
        }
    })
}

fn block<'hir>(ctx: &Ctx<'hir>, ast: &ast::Block) -> Result<hir::Block<'hir>, HirError> {
    Ok(hir::Block {
        id: ast.id,
        statements: todo!(),
    })
}

fn attribute<'hir>(ctx: &Ctx<'hir>, ast: &ast::Attribute) -> Result<hir::Attribute, HirError> {
    Ok(hir::Attribute {})
}
