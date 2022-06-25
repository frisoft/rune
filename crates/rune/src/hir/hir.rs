use crate::ast;
use crate::parse::{Id, Opaque};

/// An identifier, like `foo` or `Hello`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ident {
    /// The source of the identifier.
    pub source: ast::LitSource,
}

/// Visibility level restricted to some path: pub(self) or pub(super) or pub or pub(in some::module).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Visibility<'hir> {
    /// An inherited visibility level, this usually means private.
    Inherited,
    /// An unrestricted public visibility level: `pub`.
    Public,
    /// Crate visibility `pub`.
    Crate,
    /// Super visibility `pub(super)`.
    Super,
    /// Self visibility `pub(self)`.
    SelfValue,
    /// In visibility `pub(in path)`.
    In(&'hir Path<'hir>),
}

/// A pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Pat<'hir> {
    /// An ignored binding.
    PatIgnore,
    /// A path pattern.
    PatPath(&'hir Path<'hir>),
}

/// An hir expression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Expr<'hir> {
    Path(&'hir Path<'hir>),
}

/// A path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Path<'hir> {
    /// Opaque id associated with path.
    pub id: Id,
    /// The first component in the path.
    pub first: PathSegment<'hir>,
    /// The rest of the components in the path.
    pub rest: &'hir [PathSegment<'hir>],
}

/// A single segment in a path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PathSegment<'hir> {
    /// A path segment that contains `Self`.
    SelfType,
    /// A path segment that contains `self`.
    SelfValue,
    /// A path segment that is an identifier.
    Ident(&'hir Ident),
    /// The `crate` keyword used as a path segment.
    Crate,
    /// The `super` keyword use as a path segment.
    Super,
    /// A path segment that is a generic argument.
    Generics(&'hir [Expr<'hir>]),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Opaque)]
#[non_exhaustive]
pub struct ItemFn<'hir> {
    /// Opaque identifier for fn item.
    #[rune(id)]
    pub id: Id,
    /// The attributes for the fn
    pub attributes: &'hir [Attribute],
    /// The visibility of the `fn` item
    pub visibility: &'hir Visibility<'hir>,
    /// The name of the function.
    pub name: &'hir Ident,
    /// The arguments of the function.
    pub args: &'hir [FnArg<'hir>],
    /// The body of the function.
    pub body: &'hir Block<'hir>,
}

/// A single argument to a function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FnArg<'hir> {
    /// The `self` parameter.
    SelfValue,
    /// Function argument is a pattern binding.
    Pat(&'hir Pat<'hir>),
}

/// A block of statements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Opaque)]
#[non_exhaustive]
pub struct Block<'hir> {
    /// The unique identifier for the block expression.
    #[rune(id)]
    pub id: Id,
    /// Statements in the block.
    pub statements: &'hir [Stmt<'hir>],
}

/// A statement within a block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stmt<'hir> {
    /// A local declaration.
    Local(&'hir Local<'hir>),
    /// An expression.
    Expr(&'hir Expr<'hir>),
    /// An expression with a trailing semi-colon.
    Semi(&'hir Expr<'hir>),
}

/// A local variable declaration `let <pattern> = <expr>;`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct Local<'hir> {
    /// The attributes for the local expression
    pub attributes: &'hir [Attribute],
    /// The name of the binding.
    pub pat: &'hir Pat<'hir>,
    /// The expression the binding is assigned to.
    pub expr: &'hir Expr<'hir>,
}

/// Attribute like `#[derive(Debug)]`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct Attribute {}
