use crate::ast::{self, Span, Spanned};
use crate::parse::{Id, Opaque};

/// An identifier, like `foo` or `Hello`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Spanned)]
pub struct Ident {
    /// The span of the identifier.
    pub span: Span,
    /// The source of the identifier.
    #[rune(skip)]
    pub source: ast::LitSource,
}

/// A label, like `'foo`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Spanned)]
#[non_exhaustive]
pub struct Label {
    /// The span of the label.
    pub span: Span,
    /// The source of the label.
    #[rune(skip)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Spanned)]
#[non_exhaustive]
pub enum Pat<'hir> {
    /// An ignored binding.
    PatIgnore(Span),
    /// A path pattern.
    PatPath(&'hir Path<'hir>),
}

/// An hir expression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Spanned)]
pub enum Expr<'hir> {
    Path(&'hir Path<'hir>),
    Assign(&'hir ExprAssign<'hir>),
    While(&'hir ExprWhile<'hir>),
    Loop(&'hir ExprLoop<'hir>),
    For(&'hir ExprFor<'hir>),
    Let(&'hir ExprLet<'hir>),
    /*If(&'hir ExprIf<'hir>),
    Match(&'hir ExprMatch<'hir>),
    Call(&'hir ExprCall<'hir>),
    FieldAccess(&'hir ExprFieldAccess<'hir>),
    Group(&'hir ExprGroup<'hir>),
    Empty(&'hir ExprEmpty<'hir>),
    Binary(&'hir ExprBinary<'hir>),
    Unary(&'hir ExprUnary<'hir>),
    Index(&'hir ExprIndex<'hir>),
    Break(&'hir ExprBreak<'hir>),
    Continue(&'hir ExprContinue<'hir>),
    Yield(&'hir ExprYield<'hir>),
    Block(&'hir ExprBlock<'hir>),
    Return(&'hir ExprReturn<'hir>),
    Await(&'hir ExprAwait<'hir>),
    Try(&'hir ExprTry<'hir>),
    Select(&'hir ExprSelect<'hir>),
    Closure(&'hir ExprClosure<'hir>),
    Lit(&'hir ExprLit<'hir>),
    ForceSemi(&'hir ForceSemi<'hir>),
    MacroCall(&'hir MacroCall<'hir>),
    Object(&'hir ExprObject<'hir>),
    Tuple(&'hir ExprTuple<'hir>),
    Vec(&'hir ExprVec<'hir>),
    Range(&'hir ExprRange<'hir>),*/
}

/// An assign expression `a = b`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Spanned)]
#[non_exhaustive]
pub struct ExprAssign<'hir> {
    /// Attributes associated with the assign expression.
    pub attributes: &'hir [Attribute],
    /// The expression being awaited.
    pub lhs: &'hir Expr<'hir>,
    /// The value.
    pub rhs: &'hir Expr<'hir>,
}

/// A `while` loop: `while [expr] { ... }`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Spanned)]
#[non_exhaustive]
pub struct ExprWhile<'hir> {
    /// The attributes for the `while` loop
    #[rune(iter)]
    pub attributes: &'hir [Attribute],
    /// A label for the while loop.
    #[rune(iter)]
    pub label: Option<&'hir Label>,
    /// The name of the binding.
    pub condition: &'hir Condition<'hir>,
    /// The body of the while loop.
    pub body: &'hir Block<'hir>,
}

/// A `loop` expression: `loop { ... }`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Spanned)]
#[non_exhaustive]
pub struct ExprLoop<'hir> {
    /// The attributes for the `loop`
    #[rune(iter, meta)]
    pub attributes: &'hir [Attribute],
    /// A label.
    #[rune(iter, meta)]
    pub label: Option<&'hir Label>,
    /// The body of the loop.
    pub body: &'hir Block<'hir>,
}

/// A `for` loop over an iterator: `for i in [1, 2, 3] {}`.
#[derive(Debug, Clone, Copy, PartialEq, Spanned)]
#[non_exhaustive]
pub struct ExprFor<'hir> {
    /// The attributes of the `for` loop
    #[rune(iter)]
    pub attributes: &'hir [Attribute],
    /// The label of the loop.
    #[rune(iter)]
    pub label: Option<&'hir Label>,
    /// The pattern binding to use.
    /// Non-trivial pattern bindings will panic if the value doesn't match.
    pub binding: &'hir Pat<'hir>,
    /// Expression producing the iterator.
    pub iter: &'hir Expr<'hir>,
    /// The body of the loop.
    pub body: &'hir Block<'hir>,
}

/// A let expression `let <name> = <expr>`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Spanned)]
#[non_exhaustive]
pub struct ExprLet<'hir> {
    /// The attributes for the let expression
    #[rune(iter)]
    pub attributes: &'hir [Attribute],
    /// The name of the binding.
    pub pat: &'hir Pat<'hir>,
    /// The expression the binding is assigned to.
    pub expr: &'hir Expr<'hir>,
}

/// The condition in an if statement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Spanned)]
#[non_exhaustive]
pub enum Condition<'hir> {
    /// A regular expression.
    Expr(&'hir Expr<'hir>),
    /// A pattern match.
    ExprLet(&'hir ExprLet<'hir>),
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Opaque, Spanned)]
#[non_exhaustive]
pub struct Block<'hir> {
    /// The unique identifier for the block expression.
    #[rune(id)]
    pub id: Id,
    /// The span of the block.
    pub span: Span,
    /// Statements in the block.
    #[rune(iter)]
    pub statements: &'hir [Stmt<'hir>],
}

/// A statement within a block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Spanned)]
pub enum Stmt<'hir> {
    /// A local declaration.
    Local(&'hir Local<'hir>),
    /// An expression.
    Expr(&'hir Expr<'hir>),
    /// An expression with a trailing semi-colon.
    Semi(&'hir Expr<'hir>),
}

/// A local variable declaration `let <pattern> = <expr>;`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Spanned)]
#[non_exhaustive]
pub struct Local<'hir> {
    /// The attributes for the local expression
    #[rune(iter)]
    pub attributes: &'hir [Attribute],
    /// The name of the binding.
    pub pat: &'hir Pat<'hir>,
    /// The expression the binding is assigned to.
    pub expr: &'hir Expr<'hir>,
}

/// Attribute like `#[derive(Debug)]`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Spanned)]
#[non_exhaustive]
pub struct Attribute {
    /// The span of the attribute.
    pub span: Span,
}
