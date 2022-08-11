use crate::ast::prelude::*;

/// A pattern match.
#[derive(Debug, Clone, PartialEq, Eq, ToTokens, Spanned)]
#[non_exhaustive]
pub enum Pat {
    /// An ignored binding `_`.
    PatIgnore(PatIgnore),
    /// A variable binding `n`.
    PatPath(PatPath),
    /// A literal pattern. This is represented as an expression.
    PatLit(PatLit),
    /// A vector pattern.
    PatVec(PatVec),
    /// A tuple pattern.
    PatTuple(PatTuple),
    /// An object pattern.
    PatObject(PatObject),
}

/// Parsing a block expression.
///
/// # Examples
///
/// ```
/// use rune::{ast, testing};
///
/// testing::roundtrip::<ast::Pat>("()");
/// testing::roundtrip::<ast::Pat>("42");
/// testing::roundtrip::<ast::Pat>("-42");
/// testing::roundtrip::<ast::Pat>("3.1415");
/// testing::roundtrip::<ast::Pat>("-3.1415");
/// testing::roundtrip::<ast::Pat>("b'a'");
/// testing::roundtrip::<ast::Pat>("'a'");
/// testing::roundtrip::<ast::Pat>("b\"hello world\"");
/// testing::roundtrip::<ast::Pat>("\"hello world\"");
/// testing::roundtrip::<ast::Pat>("var");
/// testing::roundtrip::<ast::Pat>("_");
/// testing::roundtrip::<ast::Pat>("Foo(n)");
/// ```
impl Parse for Pat {
    fn parse(p: &mut Parser<'_>) -> Result<Self, ParseError> {
        let attributes = p.parse::<Vec<ast::Attribute>>()?;

        match p.nth(0)? {
            K![byte] => {
                return Ok(Self::PatLit(PatLit {
                    attributes,
                    expr: Box::new(ast::Expr::from_lit(ast::Lit::Byte(p.parse()?))),
                }));
            }
            K![char] => {
                return Ok(Self::PatLit(PatLit {
                    attributes,
                    expr: Box::new(ast::Expr::from_lit(ast::Lit::Char(p.parse()?))),
                }));
            }
            K![bytestr] => {
                return Ok(Self::PatLit(PatLit {
                    attributes,
                    expr: Box::new(ast::Expr::from_lit(ast::Lit::ByteStr(p.parse()?))),
                }));
            }
            K![true] | K![false] => {
                return Ok(Self::PatLit(PatLit {
                    attributes,
                    expr: Box::new(ast::Expr::from_lit(ast::Lit::Bool(p.parse()?))),
                }));
            }
            K![str] => {
                return Ok(Self::PatLit(PatLit {
                    attributes,
                    expr: Box::new(ast::Expr::from_lit(ast::Lit::Str(p.parse()?))),
                }));
            }
            K![number] => {
                return Ok(Self::PatLit(PatLit {
                    attributes,
                    expr: Box::new(ast::Expr::from_lit(ast::Lit::Number(p.parse()?))),
                }));
            }
            K!['('] => {
                return Ok({
                    Self::PatTuple(PatTuple {
                        attributes,
                        path: None,
                        open: p.parse()?,
                        items: parse_items(p)?,
                        rest: p.parse()?,
                        close: p.parse()?,
                    })
                });
            }
            K!['['] => {
                return Ok(Self::PatVec(PatVec {
                    attributes,
                    open: p.parse()?,
                    items: parse_items(p)?,
                    rest: p.parse()?,
                    close: p.parse()?,
                }))
            }
            K![#] => {
                return Ok(Self::PatObject(PatObject {
                    attributes,
                    ident: p.parse()?,
                    open: p.parse()?,
                    items: parse_items(p)?,
                    rest: p.parse()?,
                    close: p.parse()?,
                }))
            }
            K![-] => {
                let expr: ast::Expr = p.parse()?;

                if expr.is_lit() {
                    return Ok(Self::PatLit(PatLit {
                        attributes,
                        expr: Box::new(expr),
                    }));
                }
            }
            K![_] => {
                return Ok(Self::PatIgnore(PatIgnore {
                    attributes,
                    underscore: p.parse()?,
                }))
            }
            _ if ast::Path::peek(p.peeker()) => {
                let path = p.parse::<ast::Path>()?;

                return Ok(match p.nth(0)? {
                    K!['('] => Self::PatTuple(PatTuple {
                        attributes,
                        path: Some(path),
                        open: p.parse()?,
                        items: parse_items(p)?,
                        rest: p.parse()?,
                        close: p.parse()?,
                    }),
                    K!['{'] => Self::PatObject(PatObject {
                        attributes,
                        ident: ast::ObjectIdent::Named(path),
                        open: p.parse()?,
                        items: parse_items(p)?,
                        rest: p.parse()?,
                        close: p.parse()?,
                    }),
                    _ => Self::PatPath(PatPath { attributes, path }),
                });
            }
            _ => (),
        }

        Err(ParseError::expected(p.tok_at(0)?, "pattern"))
    }
}

impl Peek for Pat {
    fn peek(p: &mut Peeker<'_>) -> bool {
        match p.nth(0) {
            K!['('] => true,
            K!['['] => true,
            K![#] => matches!(p.nth(1), K!['{']),
            K![_] => true,
            K![byte] | K![char] | K![number] | K![str] => true,
            K![true] | K![false] => true,
            K![-] => matches!(p.nth(1), K![number]),
            _ => ast::Path::peek(p),
        }
    }
}

/// Helper function to parse items.
fn parse_items<T, S>(p: &mut Parser<'_>) -> Result<Vec<(T, Option<S>)>, ParseError>
where
    T: Peek + Parse,
    S: Peek + Parse,
{
    let mut items = Vec::new();

    while p.peek::<T>()? {
        let expr = p.parse()?;
        let sep = p.parse::<Option<S>>()?;
        let is_end = sep.is_none();
        items.push((expr, sep));

        if is_end {
            break;
        }
    }

    Ok(items)
}

/// A literal pattern.
#[derive(Debug, Clone, PartialEq, Eq, ToTokens, Spanned)]
#[non_exhaustive]
pub struct PatLit {
    /// Attributes associated with the pattern.
    #[rune(iter)]
    pub attributes: Vec<ast::Attribute>,
    /// The literal expression.
    pub expr: Box<ast::Expr>,
}

/// An array pattern.
#[derive(Debug, Clone, PartialEq, Eq, ToTokens, Spanned)]
#[non_exhaustive]
pub struct PatVec {
    /// Attributes associated with the vector pattern.
    #[rune(iter)]
    pub attributes: Vec<ast::Attribute>,
    /// The open bracket.
    pub open: T!['['],
    /// Values in the type.
    #[rune(iter)]
    pub items: Vec<(Pat, Option<T![,]>)>,
    /// The rest pattern.
    #[rune(iter)]
    pub rest: Option<T![..]>,
    /// The close bracket.
    pub close: T![']'],
}

/// A tuple pattern.
#[derive(Debug, Clone, PartialEq, Eq, ToTokens, Spanned)]
#[non_exhaustive]
pub struct PatTuple {
    /// Attributes associated with the object pattern.
    #[rune(iter)]
    pub attributes: Vec<ast::Attribute>,
    /// The path, if the tuple is typed.
    #[rune(iter)]
    pub path: Option<ast::Path>,
    /// The open parenthesis.
    pub open: T!['('],
    /// The items in the tuple.
    pub items: Vec<(ast::Pat, Option<T![,]>)>,
    /// The rest pattern.
    #[rune(iter)]
    pub rest: Option<T![..]>,
    /// The closing parenthesis.
    pub close: T![')'],
}

/// An object pattern.
#[derive(Debug, Clone, PartialEq, Eq, ToTokens, Spanned)]
#[non_exhaustive]
pub struct PatObject {
    /// Attributes associated with the object pattern.
    #[rune(iter)]
    pub attributes: Vec<ast::Attribute>,
    /// The identifier of the object pattern.
    pub ident: ast::ObjectIdent,
    /// The open brace.
    pub open: T!['{'],
    /// The fields matched against.
    pub items: Vec<(PatBinding, Option<T![,]>)>,
    /// The rest pattern.
    #[rune(iter)]
    pub rest: Option<T![..]>,
    /// The closing brace.
    pub close: T!['}'],
}

/// An object item.
#[derive(Debug, Clone, PartialEq, Eq, ToTokens, Spanned, Parse)]
#[non_exhaustive]
pub struct PatBinding {
    /// The key of an object.
    pub key: ast::ObjectKey,
    /// The colon separator for the binding.
    pub colon: T![:],
    /// What the binding is to.
    pub pat: Box<ast::Pat>,
}

impl Peek for PatBinding {
    fn peek(p: &mut Peeker<'_>) -> bool {
        ast::ObjectKey::peek(p)
    }
}

/// A path pattern.
#[derive(Debug, Clone, PartialEq, Eq, ToTokens, Spanned)]
#[non_exhaustive]
pub struct PatPath {
    /// Attributes associate with the path.
    #[rune(iter)]
    pub attributes: Vec<ast::Attribute>,
    /// The path of the pattern.
    pub path: ast::Path,
}

/// An ignore pattern.
#[derive(Debug, Clone, PartialEq, Eq, ToTokens, Spanned)]
#[non_exhaustive]
pub struct PatIgnore {
    /// Attributes associate with the pattern.
    #[rune(iter)]
    pub attributes: Vec<ast::Attribute>,
    /// The ignore token`_`.
    pub underscore: T![_],
}
