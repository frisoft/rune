use crate::arena::Arena;
use crate::ast::{self, Spanned};
use crate::compile::{CompileError, CompileErrorKind};
use crate::hir;
use crate::query::{self, Query};
use crate::SourceId;

/// Allocate a single object in the arena.
macro_rules! alloc {
    ($span:expr => $cx:expr; $value:expr) => {
        $cx.arena.alloc($value).map_err(|e| {
            CompileError::new(
                $span,
                CompileErrorKind::ArenaAllocError {
                    requested: e.requested,
                },
            )
        })?
    };
}

/// Unpacks an optional value and allocates it in the arena.
macro_rules! option {
    ($span:expr => $cx:expr; $value:expr, |$pat:pat_param| $closure:expr) => {
        match $value {
            Some($pat) => {
                Some(&*alloc!($span => $cx; $closure))
            }
            None => {
                None
            }
        }
    };
}

/// Unpacks an iterator value and allocates it in the arena as a slice.
macro_rules! iter {
    ($span:expr => $cx:expr; $iter:expr, |$pat:pat_param| $closure:expr) => {{
        let mut it = IntoIterator::into_iter($iter);
        let span = Spanned::span($span);

        let mut writer = match $cx.arena.alloc_iter(ExactSizeIterator::len(&it)) {
            Ok(writer) => writer,
            Err(e) => {
                return Err(CompileError::new(
                    span,
                    CompileErrorKind::ArenaAllocError {
                        requested: e.requested,
                    },
                ));
            }
        };

        while let Some($pat) = it.next() {
            if let Err(e) = writer.write($closure) {
                return Err(CompileError::new(
                    span,
                    CompileErrorKind::ArenaWriteSliceOutOfBounds { index: e.index },
                ));
            }
        }

        writer.finish()
    }};
}

type Result<T> = ::std::result::Result<T, CompileError>;

pub struct Ctxt<'a, 'hir> {
    /// Source being processed.
    source_id: SourceId,
    /// Query system.
    q: Query<'a>,
    /// Arena used for allocations.
    arena: &'hir Arena,
}

impl<'a, 'hir> Ctxt<'a, 'hir> {
    /// Construct a new contctx.
    pub(crate) fn new(source_id: SourceId, q: Query<'a>, arena: &'hir Arena) -> Self {
        Self {
            source_id,
            q,
            arena,
        }
    }
}

/// Lower a function item.
pub fn item_fn<'hir>(cx: &mut Ctxt<'_, 'hir>, ast: &ast::ItemFn) -> Result<hir::ItemFn<'hir>> {
    Ok(hir::ItemFn {
        id: ast.id,
        span: ast.span(),
        visibility: alloc!(ast => cx; visibility(cx, &ast.visibility)?),
        name: alloc!(ast => cx; ast.name),
        args: iter!(ast => cx; &ast.args, |(ast, _)| fn_arg(cx, ast)?),
        body: alloc!(ast => cx; block(cx, &ast.body)?),
    })
}

/// Lower a closure expression.
pub fn expr_closure<'hir>(
    cx: &mut Ctxt<'_, 'hir>,
    ast: &ast::ExprClosure,
) -> Result<hir::ExprClosure<'hir>> {
    Ok(hir::ExprClosure {
        id: ast.id,
        args: match &ast.args {
            ast::ExprClosureArgs::Empty { .. } => &[],
            ast::ExprClosureArgs::List { args, .. } => {
                iter!(ast => cx; args, |(ast, _)| fn_arg(cx, ast)?)
            }
        },
        body: alloc!(ast => cx; expr(cx, &ast.body)?),
    })
}

#[derive(Clone, Copy, Spanned)]
enum BlockStatement<'a> {
    Local(&'a ast::Local),
    Expr(&'a ast::Expr),
}

/// Lower the specified block.
pub fn block<'hir>(cx: &mut Ctxt<'_, 'hir>, ast: &ast::Block) -> Result<hir::Block<'hir>> {
    let mut tail_expr = None;
    let mut must_be_last = None;
    let mut statements = Vec::with_capacity(ast.statements.len());

    for stmt in &ast.statements {
        if let Some(span) = must_be_last {
            return Err(CompileError::new(
                span,
                CompileErrorKind::ExpectedBlockSemiColon {
                    followed_span: stmt.span(),
                },
            ));
        }

        let ast = match stmt {
            ast::Stmt::Local(ast) => {
                statements.push(BlockStatement::Local(ast));
                continue;
            }
            ast::Stmt::Expr(ast) => {
                if ast.needs_semi() {
                    must_be_last = Some(ast.span());
                }

                ast
            }
            ast::Stmt::Semi(ast) => {
                if !ast.needs_semi() {
                    cx.q.diagnostics
                        .uneccessary_semi_colon(cx.source_id, ast.semi_token.span());
                }

                &ast.expr
            }
            ast::Stmt::Item(item, semi) => {
                if let Some(semi) = semi {
                    if !item.needs_semi_colon() {
                        cx.q.diagnostics
                            .uneccessary_semi_colon(cx.source_id, semi.span());
                    }
                }

                continue;
            }
        };

        statements.extend(tail_expr.replace(ast).map(BlockStatement::Expr));
    }

    let tail = if let Some(ast) = tail_expr {
        Some(&*alloc!(ast => cx; expr(cx, ast)?))
    } else {
        None
    };

    Ok(hir::Block {
        id: ast.id,
        span: ast.span(),
        statements: iter!(ast => cx; statements, |ast| block_stmt(cx, ast)?),
        tail,
    })
}

/// Lower a statement
fn block_stmt<'hir>(cx: &mut Ctxt<'_, 'hir>, ast: BlockStatement<'_>) -> Result<hir::Stmt<'hir>> {
    Ok(match ast {
        BlockStatement::Local(ast) => hir::Stmt::Local(alloc!(ast => cx; local(cx, ast)?)),
        BlockStatement::Expr(ast) => hir::Stmt::Expr(alloc!(ast => cx; expr(cx, ast)?)),
    })
}

/// Lower an expression.
pub fn expr<'hir>(cx: &mut Ctxt<'_, 'hir>, ast: &ast::Expr) -> Result<hir::Expr<'hir>> {
    let kind = match ast {
        ast::Expr::Path(ast) => hir::ExprKind::Path(alloc!(ast => cx; path(cx, ast)?)),
        ast::Expr::Assign(ast) => hir::ExprKind::Assign(alloc!(ast => cx; hir::ExprAssign {
            lhs: alloc!(ast => cx; expr(cx, &ast.lhs)?),
            rhs: alloc!(ast => cx; expr(cx, &ast.rhs)?),
        })),
        ast::Expr::While(ast) => hir::ExprKind::Loop(alloc!(ast => cx; hir::ExprLoop {
            label: option!(ast => cx; &ast.label, |(ast, _)| label(cx, ast)?),
            condition: hir::LoopCondition::Condition {
                condition: alloc!(ast => cx; condition(cx, &ast.condition)?),
            },
            body: alloc!(ast => cx; block(cx, &ast.body)?),
        })),
        ast::Expr::Loop(ast) => hir::ExprKind::Loop(alloc!(ast => cx; hir::ExprLoop {
            label: option!(ast => cx; &ast.label, |(ast, _)| label(cx, ast)?),
            condition: hir::LoopCondition::Forever,
            body: alloc!(ast => cx; block(cx, &ast.body)?),
        })),
        ast::Expr::For(ast) => hir::ExprKind::Loop(alloc!(ast => cx; hir::ExprLoop {
            label: option!(ast => cx; &ast.label, |(ast, _)| label(cx, ast)?),
            condition: hir::LoopCondition::Iterator {
                binding: alloc!(ast => cx; pat(cx, &ast.binding)?),
                iter: alloc!(ast => cx; expr(cx, &ast.iter)?)
            },
            body: alloc!(ast => cx; block(cx, &ast.body)?),
        })),
        ast::Expr::Let(ast) => hir::ExprKind::Let(alloc!(ast => cx; hir::ExprLet {
            pat: alloc!(ast => cx; pat(cx, &ast.pat)?),
            expr: alloc!(ast => cx; expr(cx, &ast.expr)?),
        })),
        ast::Expr::If(ast) => hir::ExprKind::If(alloc!(ast => cx; hir::ExprIf {
            condition: alloc!(ast => cx; condition(cx, &ast.condition)?),
            block: alloc!(ast => cx; block(cx, &ast.block)?),
            expr_else_ifs: iter!(ast => cx; &ast.expr_else_ifs, |ast| hir::ExprElseIf {
                span: ast.span(),
                condition: alloc!(ast => cx; condition(cx, &ast.condition)?),
                block: alloc!(ast => cx; block(cx, &ast.block)?),
            }),
            expr_else: option!(ast => cx; &ast.expr_else, |ast| hir::ExprElse {
                span: ast.span(),
                block: alloc!(ast => cx; block(cx, &ast.block)?)
            }),
        })),
        ast::Expr::Match(ast) => hir::ExprKind::Match(alloc!(ast => cx; hir::ExprMatch {
            expr: alloc!(ast => cx; expr(cx, &ast.expr)?),
            branches: iter!(ast => cx; &ast.branches, |(ast, _)| hir::ExprMatchBranch {
                span: ast.span(),
                pat: alloc!(ast => cx; pat(cx, &ast.pat)?),
                condition: option!(ast => cx; &ast.condition, |(_, ast)| expr(cx, ast)?),
                body: alloc!(ast => cx; expr(cx, &ast.body)?),
            }),
        })),
        ast::Expr::Call(ast) => hir::ExprKind::Call(alloc!(ast => cx; hir::ExprCall {
            id: ast.id,
            expr: alloc!(ast => cx; expr(cx, &ast.expr)?),
            args: iter!(ast => cx; &ast.args, |(ast, _)| expr(cx, ast)?),
        })),
        ast::Expr::FieldAccess(ast) => {
            hir::ExprKind::FieldAccess(alloc!(ast => cx; hir::ExprFieldAccess {
                expr: alloc!(ast => cx; expr(cx, &ast.expr)?),
                expr_field: alloc!(ast => cx; match &ast.expr_field {
                    ast::ExprField::Path(ast) => hir::ExprField::Path(alloc!(ast => cx; path(cx, ast)?)),
                    ast::ExprField::LitNumber(ast) => hir::ExprField::LitNumber(alloc!(ast => cx; *ast)),
                }),
            }))
        }
        ast::Expr::Empty(ast) => hir::ExprKind::Group(alloc!(ast => cx; expr(cx, &ast.expr)?)),
        ast::Expr::Binary(ast) => hir::ExprKind::Binary(alloc!(ast => cx; hir::ExprBinary {
            lhs: alloc!(ast => cx; expr(cx, &ast.lhs)?),
            op: ast.op,
            rhs: alloc!(ast => cx; expr(cx, &ast.rhs)?),
        })),
        ast::Expr::Unary(ast) => hir::ExprKind::Unary(alloc!(ast => cx; hir::ExprUnary {
            op: ast.op,
            expr: alloc!(ast => cx; expr(cx, &ast.expr)?),
        })),
        ast::Expr::Index(ast) => hir::ExprKind::Index(alloc!(ast => cx; hir::ExprIndex {
            target: alloc!(ast => cx; expr(cx, &ast.target)?),
            index: alloc!(ast => cx; expr(cx, &ast.index)?),
        })),
        ast::Expr::Block(ast) => hir::ExprKind::Block(alloc!(ast => cx; expr_block(cx, ast)?)),
        ast::Expr::Break(ast) => {
            hir::ExprKind::Break(alloc!(ast => cx; match ast.expr.as_deref() {
                None => hir::ExprBreakValue::None,
                Some(ast::ExprBreakValue::Expr(ast)) => hir::ExprBreakValue::Expr(alloc!(ast => cx; expr(cx, ast)?)),
                Some(ast::ExprBreakValue::Label(ast)) => hir::ExprBreakValue::Label(alloc!(ast => cx; label(cx, ast)?)),
            }))
        }
        ast::Expr::Continue(ast) => {
            hir::ExprKind::Continue(option!(ast => cx; &ast.label, |ast| label(cx, ast)?))
        }
        ast::Expr::Yield(ast) => {
            hir::ExprKind::Yield(option!(ast => cx; &ast.expr, |ast| expr(cx, ast)?))
        }
        ast::Expr::Return(ast) => {
            hir::ExprKind::Return(option!(ast => cx; &ast.expr, |ast| expr(cx, ast)?))
        }
        ast::Expr::Await(ast) => hir::ExprKind::Await(alloc!(ast => cx; expr(cx, &ast.expr)?)),
        ast::Expr::Try(ast) => hir::ExprKind::Try(alloc!(ast => cx; expr(cx, &ast.expr)?)),
        ast::Expr::Select(ast) => hir::ExprKind::Select(alloc!(ast => cx; hir::ExprSelect {
            branches: iter!(ast => cx; &ast.branches, |(ast, _)| {
                match ast {
                    ast::ExprSelectBranch::Pat(ast) => hir::ExprSelectBranch::Pat(alloc!(ast => cx; hir::ExprSelectPatBranch {
                        pat: alloc!(&ast.pat => cx; pat(cx, &ast.pat)?),
                        expr: alloc!(&ast.expr => cx; expr(cx, &ast.expr)?),
                        body: alloc!(&ast.body => cx; expr(cx, &ast.body)?),
                    })),
                    ast::ExprSelectBranch::Default(ast) => hir::ExprSelectBranch::Default(alloc!(&ast.body => cx; expr(cx, &ast.body)?)),
                }
            })
        })),
        ast::Expr::Closure(ast) => {
            hir::ExprKind::Closure(alloc!(ast => cx; expr_closure(cx, ast)?))
        }
        ast::Expr::Lit(ast) => hir::ExprKind::Lit(alloc!(&ast.lit => cx; ast.lit)),
        ast::Expr::Object(ast) => hir::ExprKind::Object(alloc!(ast => cx; hir::ExprObject {
            path: object_ident(cx, &ast.ident)?,
            assignments: iter!(ast => cx; &ast.assignments, |(ast, _)| hir::FieldAssign {
                span: ast.span(),
                key: alloc!(ast => cx; object_key(cx, &ast.key)?),
                assign: option!(ast => cx; &ast.assign, |(_, ast)| expr(cx, ast)?),
            })
        })),
        ast::Expr::Tuple(ast) => hir::ExprKind::Tuple(alloc!(ast => cx; hir::ExprSeq {
            items: iter!(ast => cx; &ast.items, |(ast, _)| expr(cx, ast)?),
        })),
        ast::Expr::Vec(ast) => hir::ExprKind::Vec(alloc!(ast => cx; hir::ExprSeq {
            items: iter!(ast => cx; &ast.items, |(ast, _)| expr(cx, ast)?),
        })),
        ast::Expr::Range(ast) => hir::ExprKind::Range(alloc!(ast => cx; hir::ExprRange {
            from: option!(ast => cx; &ast.from, |ast| expr(cx, ast)?),
            limits: match ast.limits {
                ast::ExprRangeLimits::HalfOpen(_) => hir::ExprRangeLimits::HalfOpen,
                ast::ExprRangeLimits::Closed(_) => hir::ExprRangeLimits::Closed,
            },
            to: option!(ast => cx; &ast.to, |ast| expr(cx, ast)?),
        })),
        ast::Expr::Group(ast) => hir::ExprKind::Group(alloc!(ast => cx; expr(cx, &ast.expr)?)),
        ast::Expr::MacroCall(ast) => hir::ExprKind::MacroCall(
            alloc!(ast => cx; match cx.q.builtin_macro_for(ast)?.clone() {
                query::BuiltInMacro::Template(ast) => hir::MacroCall::Template(alloc!(ast => cx; hir::BuiltInTemplate {
                    span: ast.span,
                    from_literal: ast.from_literal,
                    exprs: iter!(&ast => cx; &ast.exprs, |ast| expr(cx, ast)?),
                })),
                query::BuiltInMacro::Format(ast) => hir::MacroCall::Format(alloc!(ast => cx; hir::BuiltInFormat {
                    span: ast.span,
                    fill: ast.fill,
                    align: ast.align,
                    width: ast.width,
                    precision: ast.precision,
                    flags: ast.flags,
                    format_type: ast.format_type,
                    value: alloc!(&ast.value => cx; expr(cx, &ast.value)?),
                })),
                query::BuiltInMacro::File(ast) => hir::MacroCall::File(alloc!(ast => cx; hir::BuiltInFile {
                    span: ast.span,
                    value: ast.value,
                })),
                query::BuiltInMacro::Line(ast) => hir::MacroCall::Line(alloc!(ast => cx; hir::BuiltInLine {
                    span: ast.span,
                    value: ast.value,
                })),
            }
            ),
        ),
    };

    Ok(hir::Expr {
        span: ast.span(),
        kind,
    })
}

/// Lower a block expression.
pub fn expr_block<'hir>(
    cx: &mut Ctxt<'_, 'hir>,
    ast: &ast::ExprBlock,
) -> Result<hir::ExprBlock<'hir>> {
    Ok(hir::ExprBlock {
        kind: match (&ast.async_token, &ast.const_token) {
            (Some(..), None) => hir::ExprBlockKind::Async,
            (None, Some(..)) => hir::ExprBlockKind::Const,
            _ => hir::ExprBlockKind::Default,
        },
        block_move: ast.move_token.is_some(),
        block: alloc!(ast => cx; block(cx, &ast.block)?),
    })
}

/// Visibility covnersion.
fn visibility<'hir>(
    cx: &mut Ctxt<'_, 'hir>,
    ast: &ast::Visibility,
) -> Result<hir::Visibility<'hir>> {
    Ok(match ast {
        ast::Visibility::Inherited => hir::Visibility::Inherited,
        ast::Visibility::Public(_) => hir::Visibility::Public,
        ast::Visibility::Crate(_) => hir::Visibility::Crate,
        ast::Visibility::Super(_) => hir::Visibility::Super,
        ast::Visibility::SelfValue(_) => hir::Visibility::SelfValue,
        ast::Visibility::In(ast) => {
            hir::Visibility::In(alloc!(ast => cx; path(cx, &ast.restriction.path)?))
        }
    })
}

/// Lower a function argument.
fn fn_arg<'hir>(cx: &mut Ctxt<'_, 'hir>, ast: &ast::FnArg) -> Result<hir::FnArg<'hir>> {
    Ok(match ast {
        ast::FnArg::SelfValue(ast) => hir::FnArg::SelfValue(ast.span()),
        ast::FnArg::Pat(ast) => hir::FnArg::Pat(alloc!(ast => cx; pat(cx, ast)?)),
    })
}

/// Lower an assignment.
fn local<'hir>(cx: &mut Ctxt<'_, 'hir>, ast: &ast::Local) -> Result<hir::Local<'hir>> {
    Ok(hir::Local {
        span: ast.span(),
        pat: alloc!(ast => cx; pat(cx, &ast.pat)?),
        expr: alloc!(ast => cx; expr(cx, &ast.expr)?),
    })
}

fn pat<'hir>(cx: &mut Ctxt<'_, 'hir>, ast: &ast::Pat) -> Result<hir::Pat<'hir>> {
    Ok(hir::Pat {
        span: ast.span(),
        kind: match ast {
            ast::Pat::PatIgnore(..) => hir::PatKind::PatIgnore,
            ast::Pat::PatPath(ast) => {
                hir::PatKind::PatPath(alloc!(ast => cx; path(cx, &ast.path)?))
            }
            ast::Pat::PatLit(ast) => hir::PatKind::PatLit(alloc!(ast => cx; expr(cx, &ast.expr)?)),
            ast::Pat::PatVec(ast) => {
                let items = iter!(ast => cx; &ast.items, |(ast, _)| pat(cx, ast)?);

                hir::PatKind::PatVec(alloc!(ast => cx; hir::PatItems {
                    path: None,
                    items,
                    is_open: ast.rest.is_some(),
                }))
            }
            ast::Pat::PatTuple(ast) => {
                let items = iter!(ast => cx; &ast.items, |(ast, _)| pat(cx, ast)?);

                hir::PatKind::PatTuple(alloc!(ast => cx; hir::PatItems {
                    path: option!(ast => cx; &ast.path, |ast| path(cx, ast)?),
                    items,
                    is_open: ast.rest.is_some(),
                }))
            }
            ast::Pat::PatObject(ast) => hir::PatKind::PatObject(alloc!(ast => cx; hir::PatObject {
                path: object_ident(cx, &ast.ident)?,
                bindings: iter!(ast => cx; &ast.items, |(ast, _)| hir::PatBinding {
                    span: ast.span(),
                    key: alloc!(ast => cx; object_key(cx, &ast.key)?),
                    pat: alloc!(ast => cx; pat(cx, &ast.pat)?),
                }),
                is_open: ast.rest.is_some(),
            })),
        },
    })
}

fn object_key<'hir>(cx: &mut Ctxt<'_, 'hir>, ast: &ast::ObjectKey) -> Result<hir::ObjectKey<'hir>> {
    Ok(match ast {
        ast::ObjectKey::LitStr(ast) => hir::ObjectKey::LitStr(alloc!(ast => cx; *ast)),
        ast::ObjectKey::Path(ast) => hir::ObjectKey::Path(alloc!(ast => cx; path(cx, ast)?)),
    })
}

/// Lower an object identifier to an optional path.
fn object_ident<'hir>(
    cx: &mut Ctxt<'_, 'hir>,
    ast: &ast::ObjectIdent,
) -> Result<Option<&'hir hir::Path<'hir>>> {
    Ok(match ast {
        ast::ObjectIdent::Anonymous(_) => None,
        ast::ObjectIdent::Named(ast) => Some(alloc!(ast => cx; path(cx, ast)?)),
    })
}

/// Lower the given path.
pub fn path<'hir>(cx: &mut Ctxt<'_, 'hir>, ast: &ast::Path) -> Result<hir::Path<'hir>> {
    Ok(hir::Path {
        id: ast.id,
        span: ast.span(),
        global: ast.global.as_ref().map(Spanned::span),
        trailing: ast.trailing.as_ref().map(Spanned::span),
        first: alloc!(ast => cx; path_segment(cx, &ast.first)?),
        rest: iter!(ast => cx; &ast.rest, |(_, s)| path_segment(cx, s)?),
    })
}

fn path_segment<'hir>(
    cx: &mut Ctxt<'_, 'hir>,
    ast: &ast::PathSegment,
) -> Result<hir::PathSegment<'hir>> {
    let kind = match ast {
        ast::PathSegment::SelfType(..) => hir::PathSegmentKind::SelfType,
        ast::PathSegment::SelfValue(..) => hir::PathSegmentKind::SelfValue,
        ast::PathSegment::Ident(ast) => hir::PathSegmentKind::Ident(alloc!(ast => cx; *ast)),
        ast::PathSegment::Crate(..) => hir::PathSegmentKind::Crate,
        ast::PathSegment::Super(..) => hir::PathSegmentKind::Super,
        ast::PathSegment::Generics(ast) => {
            hir::PathSegmentKind::Generics(iter!(ast => cx; ast, |(e, _)| expr(cx, &e.expr)?))
        }
    };

    Ok(hir::PathSegment {
        span: ast.span(),
        kind,
    })
}

fn label<'hir>(_: &Ctxt<'_, 'hir>, ast: &ast::Label) -> Result<ast::Label> {
    Ok(ast::Label {
        span: ast.span,
        source: ast.source,
    })
}

fn condition<'hir>(cx: &mut Ctxt<'_, 'hir>, ast: &ast::Condition) -> Result<hir::Condition<'hir>> {
    Ok(match ast {
        ast::Condition::Expr(ast) => hir::Condition::Expr(alloc!(ast => cx; expr(cx, ast)?)),
        ast::Condition::ExprLet(ast) => hir::Condition::ExprLet(alloc!(ast => cx; hir::ExprLet {
            pat: alloc!(ast => cx; pat(cx, &ast.pat)?),
            expr: alloc!(ast => cx; expr(cx, &ast.expr)?),
        })),
    })
}
