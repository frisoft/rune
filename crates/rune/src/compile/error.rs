use core::fmt;

use crate::no_std::io;
use crate::no_std::path::PathBuf;
use crate::no_std::prelude::*;

use crate::ast;
use crate::ast::unescape;
use crate::ast::{Span, Spanned};
use crate::compile::ir;
use crate::compile::{HasSpan, ItemBuf, Location, MetaInfo, Visibility};
use crate::indexing::items::{GuardMismatch, MissingLastId};
use crate::macros::{SyntheticId, SyntheticKind};
use crate::parse::{Expectation, IntoExpectation, LexerMode};
use crate::query::MissingId;
use crate::runtime::debug::DebugSignature;
use crate::runtime::unit::EncodeError;
use crate::runtime::{AccessError, TypeInfo, TypeOf};
use crate::{Hash, SourceId};

/// An error raised by the compiler.
#[derive(Debug)]
pub struct Error {
    span: Span,
    kind: Box<ErrorKind>,
}

impl Error {
    /// Construct a new compile error.
    pub(crate) fn new<S, K>(span: S, kind: K) -> Self
    where
        S: Spanned,
        ErrorKind: From<K>,
    {
        Self {
            span: span.span(),
            kind: Box::new(ErrorKind::from(kind)),
        }
    }

    /// Construct an error which is made of a single message.
    pub fn msg<S, M>(span: S, message: M) -> Self
    where
        S: Spanned,
        M: fmt::Display,
    {
        Self {
            span: span.span(),
            kind: Box::new(ErrorKind::Custom {
                message: message.to_string().into(),
            }),
        }
    }

    /// Get the kind of the error.
    #[cfg(feature = "emit")]
    pub(crate) fn kind(&self) -> &ErrorKind {
        &self.kind
    }

    /// Convert into the kind of the error.
    #[cfg(test)]
    pub(crate) fn into_kind(self) -> ErrorKind {
        *self.kind
    }
}

impl Spanned for Error {
    #[inline]
    fn span(&self) -> Span {
        self.span
    }
}

impl crate::no_std::error::Error for Error {
    fn source(&self) -> Option<&(dyn crate::no_std::error::Error + 'static)> {
        self.kind.source()
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
        ::core::fmt::Display::fmt(&self.kind, f)
    }
}

impl<E> From<HasSpan<E>> for Error
where
    ErrorKind: From<E>,
{
    fn from(spanned: HasSpan<E>) -> Self {
        Error {
            span: spanned.span,
            kind: Box::new(ErrorKind::from(spanned.error)),
        }
    }
}

impl From<ir::scopes::MissingLocal> for ErrorKind {
    #[inline]
    fn from(error: ir::scopes::MissingLocal) -> Self {
        ErrorKind::MissingLocal {
            name: error.0.to_string(),
        }
    }
}

impl From<&'static str> for ErrorKind {
    #[inline]
    fn from(value: &'static str) -> Self {
        ErrorKind::Custom {
            message: Box::from(value),
        }
    }
}

// NB: Sometimes errors are boxed because they're so big.
impl<T> From<Box<T>> for ErrorKind
where
    ErrorKind: From<T>,
{
    #[inline]
    fn from(kind: Box<T>) -> Self {
        ErrorKind::from(*kind)
    }
}

impl Error {
    /// Error when we got mismatched meta.
    pub fn expected_meta<S>(spanned: S, meta: MetaInfo, expected: &'static str) -> Self
    where
        S: Spanned,
    {
        Self::new(spanned, ErrorKind::ExpectedMeta { meta, expected })
    }

    /// Construct an resolve expected error.
    pub(crate) fn expected<A, E>(actual: A, expected: E) -> Self
    where
        A: IntoExpectation + Spanned,
        E: IntoExpectation,
    {
        Self::new(
            actual.span(),
            ErrorKind::Expected {
                actual: actual.into_expectation(),
                expected: expected.into_expectation(),
            },
        )
    }

    /// Construct an unsupported error.
    pub(crate) fn unsupported<T, E>(actual: T, what: E) -> Self
    where
        T: Spanned,
        E: IntoExpectation,
    {
        Self::new(
            actual.span(),
            ErrorKind::Unsupported {
                what: what.into_expectation(),
            },
        )
    }

    /// An error raised when we expect a certain constant value but get another.
    pub(crate) fn expected_type<S, E>(spanned: S, actual: &ir::Value) -> Self
    where
        S: Spanned,
        E: TypeOf,
    {
        Self::new(
            spanned,
            IrErrorKind::Expected {
                expected: E::type_info(),
                actual: actual.type_info(),
            },
        )
    }
}

/// Compiler error.
#[derive(Debug)]
#[non_exhaustive]
pub(crate) enum ErrorKind {
    Custom {
        message: Box<str>,
    },
    Expected {
        actual: Expectation,
        expected: Expectation,
    },
    Unsupported {
        what: Expectation,
    },
    IrError(IrErrorKind),
    MetaConflict(MetaConflict),
    AccessError(AccessError),
    EncodeError(EncodeError),
    MissingLastId(MissingLastId),
    GuardMismatch(GuardMismatch),
    MissingScope(MissingScope),
    PopError(PopError),
    MissingId(MissingId),
    UnescapeError(unescape::ErrorKind),
    FileError {
        path: PathBuf,
        error: io::Error,
    },
    ModNotFound {
        path: PathBuf,
    },
    ModAlreadyLoaded {
        item: ItemBuf,
        existing: (SourceId, Span),
    },
    MissingMacro {
        item: ItemBuf,
    },
    MissingSelf,
    MissingLocal {
        name: String,
    },
    MissingItem {
        item: ItemBuf,
    },
    MissingItemHash {
        hash: Hash,
    },
    MissingItemParameters {
        item: ItemBuf,
        parameters: Box<[Option<Hash>]>,
    },
    UnsupportedGlobal,
    UnsupportedModuleSource,
    UnsupportedModuleRoot {
        root: PathBuf,
    },
    UnsupportedModuleItem {
        item: ItemBuf,
    },
    UnsupportedSelf,
    UnsupportedUnaryOp {
        op: ast::UnOp,
    },
    UnsupportedBinaryOp {
        op: ast::BinOp,
    },
    UnsupportedLitObject {
        meta: MetaInfo,
    },
    LitObjectMissingField {
        field: Box<str>,
        item: ItemBuf,
    },
    LitObjectNotField {
        field: Box<str>,
        item: ItemBuf,
    },
    UnsupportedAssignExpr,
    UnsupportedBinaryExpr,
    UnsupportedRef,
    UnsupportedSelectPattern,
    UnsupportedArgumentCount {
        expected: usize,
        actual: usize,
    },
    UnsupportedPatternExpr,
    UnsupportedBinding,
    DuplicateObjectKey {
        existing: Span,
        object: Span,
    },
    InstanceFunctionOutsideImpl,
    UnsupportedTupleIndex {
        number: ast::Number,
    },
    BreakOutsideOfLoop,
    ContinueOutsideOfLoop,
    SelectMultipleDefaults,
    ExpectedBlockSemiColon {
        followed_span: Span,
    },
    FnConstAsyncConflict,
    BlockConstAsyncConflict,
    ClosureKind,
    UnsupportedSelfType,
    UnsupportedSuper,
    UnsupportedSuperInSelfType,
    UnsupportedAfterGeneric,
    IllegalUseSegment,
    UseAliasNotSupported,
    FunctionConflict {
        existing: DebugSignature,
    },
    FunctionReExportConflict {
        hash: Hash,
    },
    ConstantConflict {
        hash: Hash,
    },
    StaticStringMissing {
        hash: Hash,
        slot: usize,
    },
    StaticBytesMissing {
        hash: Hash,
        slot: usize,
    },
    StaticStringHashConflict {
        hash: Hash,
        current: String,
        existing: String,
    },
    StaticBytesHashConflict {
        hash: Hash,
        current: Vec<u8>,
        existing: Vec<u8>,
    },
    StaticObjectKeysMissing {
        hash: Hash,
        slot: usize,
    },
    StaticObjectKeysHashConflict {
        hash: Hash,
        current: Box<[String]>,
        existing: Box<[String]>,
    },
    MissingLoopLabel {
        label: Box<str>,
    },
    ExpectedLeadingPathSegment,
    UnsupportedVisibility,
    ExpectedMeta {
        expected: &'static str,
        meta: MetaInfo,
    },
    NoSuchBuiltInMacro {
        name: Box<str>,
    },
    VariableMoved {
        moved_at: Span,
    },
    UnsupportedGenerics,
    NestedTest {
        nested_span: Span,
    },
    NestedBench {
        nested_span: Span,
    },
    MissingFunctionHash {
        hash: Hash,
    },
    FunctionConflictHash {
        hash: Hash,
    },
    PatternMissingFields {
        item: ItemBuf,
        fields: Box<[Box<str>]>,
    },
    MissingLabelLocation {
        name: &'static str,
        index: usize,
    },
    MaxMacroRecursion {
        depth: usize,
        max: usize,
    },
    YieldInConst,
    AwaitInConst,
    AwaitOutsideAsync,
    ExpectedEof {
        actual: ast::Kind,
    },
    UnexpectedEof,
    BadLexerMode {
        actual: LexerMode,
        expected: LexerMode,
    },
    ExpectedEscape,
    UnterminatedStrLit,
    UnterminatedByteStrLit,
    UnterminatedCharLit,
    UnterminatedByteLit,
    ExpectedCharClose,
    ExpectedCharOrLabel,
    ExpectedByteClose,
    UnexpectedChar {
        c: char,
    },
    PrecedenceGroupRequired,
    BadNumberOutOfBounds,
    BadFieldAccess,
    ExpectedMacroCloseDelimiter {
        expected: ast::Kind,
        actual: ast::Kind,
    },
    MultipleMatchingAttributes {
        name: &'static str,
    },
    MissingSourceId {
        source_id: SourceId,
    },
    ExpectedMultilineCommentTerm,
    BadSlice,
    BadSyntheticId {
        kind: SyntheticKind,
        id: SyntheticId,
    },
    BadCharLiteral,
    BadByteLiteral,
    BadNumberLiteral,
    AmbiguousItem {
        item: ItemBuf,
        locations: Vec<(Location, ItemBuf)>,
    },
    AmbiguousContextItem {
        item: ItemBuf,
        infos: Box<[MetaInfo]>,
    },
    NotVisible {
        chain: Vec<Location>,
        location: Location,
        visibility: Visibility,
        item: ItemBuf,
        from: ItemBuf,
    },
    NotVisibleMod {
        chain: Vec<Location>,
        location: Location,
        visibility: Visibility,
        item: ItemBuf,
        from: ItemBuf,
    },
    MissingMod {
        item: ItemBuf,
    },
    ImportCycle {
        path: Vec<ImportStep>,
    },
    ImportRecursionLimit {
        count: usize,
        #[allow(unused)]
        path: Vec<ImportStep>,
    },
    LastUseComponent,
    VariantRttiConflict {
        hash: Hash,
    },
    TypeRttiConflict {
        hash: Hash,
    },
    ArenaWriteSliceOutOfBounds {
        index: usize,
    },
    ArenaAllocError {
        requested: usize,
    },
    UnsupportedPatternRest,
    UnsupportedMut,
    UnsupportedSuffix,
}

impl crate::no_std::error::Error for ErrorKind {
    fn source(&self) -> Option<&(dyn crate::no_std::error::Error + 'static)> {
        match self {
            ErrorKind::IrError(source) => Some(source),
            ErrorKind::MetaConflict(source) => Some(source),
            ErrorKind::AccessError(source) => Some(source),
            ErrorKind::EncodeError(source) => Some(source),
            ErrorKind::MissingLastId(source) => Some(source),
            ErrorKind::GuardMismatch(source) => Some(source),
            ErrorKind::MissingScope(source) => Some(source),
            ErrorKind::PopError(source) => Some(source),
            ErrorKind::MissingId(source) => Some(source),
            ErrorKind::UnescapeError(source) => Some(source),
            ErrorKind::FileError { error, .. } => Some(error),
            _ => None,
        }
    }
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ErrorKind::Custom { message } => {
                write!(f, "{message}", message = message)?;
            }
            ErrorKind::Expected { actual, expected } => {
                write!(
                    f,
                    "Expected `{expected}`, but got `{actual}`",
                    expected = expected,
                    actual = actual
                )?;
            }
            ErrorKind::Unsupported { what } => {
                write!(f, "Unsupported `{what}`", what = what)?;
            }
            ErrorKind::IrError(error) => {
                error.fmt(f)?;
            }
            ErrorKind::MetaConflict(error) => {
                error.fmt(f)?;
            }
            ErrorKind::AccessError(error) => {
                error.fmt(f)?;
            }
            ErrorKind::EncodeError(error) => {
                error.fmt(f)?;
            }
            ErrorKind::MissingLastId(error) => {
                error.fmt(f)?;
            }
            ErrorKind::GuardMismatch(error) => {
                error.fmt(f)?;
            }
            ErrorKind::MissingScope(error) => {
                error.fmt(f)?;
            }
            ErrorKind::PopError(error) => {
                error.fmt(f)?;
            }
            ErrorKind::MissingId(error) => {
                error.fmt(f)?;
            }
            ErrorKind::UnescapeError(error) => {
                error.fmt(f)?;
            }
            ErrorKind::FileError { path, error } => {
                write!(
                    f,
                    "Failed to load `{path}`: {error}",
                    path = path.display(),
                    error = error
                )?;
            }
            ErrorKind::ModNotFound { path } => {
                write!(
                    f,
                    "File not found, expected a module file like `{path}.rn`",
                    path = path.display()
                )?;
            }
            ErrorKind::ModAlreadyLoaded { item, .. } => {
                write!(f, "Module `{item}` has already been loaded")?;
            }
            ErrorKind::MissingMacro { item } => {
                write!(f, "Missing macro `{item}`")?;
            }
            ErrorKind::MissingSelf => write!(f, "No `self` in current context")?,
            ErrorKind::MissingLocal { name } => {
                write!(f, "No local variable `{name}`")?;
            }
            ErrorKind::MissingItem { item } => {
                write!(f, "Missing item `{item}`")?;
            }
            ErrorKind::MissingItemHash { hash } => {
                write!(
                    f,
                    "Tried to insert meta with hash `{hash}` which does not have an item",
                )?;
            }
            ErrorKind::MissingItemParameters { item, parameters } => {
                write!(f, "Missing item `{item} {parameters:?}`",)?;
            }
            ErrorKind::UnsupportedGlobal => {
                write!(f, "Unsupported crate prefix `::`")?;
            }
            ErrorKind::UnsupportedModuleSource => {
                write!(
                    f,
                    "Cannot load modules using a source without an associated URL"
                )?;
            }
            ErrorKind::UnsupportedModuleRoot { root } => {
                write!(
                    f,
                    "Cannot load modules relative to `{root}`",
                    root = root.display()
                )?;
            }
            ErrorKind::UnsupportedModuleItem { item } => {
                write!(f, "Cannot load module for `{item}`")?;
            }
            ErrorKind::UnsupportedSelf => {
                write!(f, "Keyword `self` not supported here")?;
            }
            ErrorKind::UnsupportedUnaryOp { op } => {
                write!(f, "Unsupported unary operator `{op}`")?;
            }
            ErrorKind::UnsupportedBinaryOp { op } => {
                write!(f, "Unsupported binary operator `{op}`")?;
            }
            ErrorKind::UnsupportedLitObject { meta } => {
                write!(f, "Item `{meta}` is not an object", meta = meta)?;
            }
            ErrorKind::LitObjectMissingField { field, item } => {
                write!(f, "Missing field `{field}` in declaration of `{item}`",)?;
            }
            ErrorKind::LitObjectNotField { field, item } => {
                write!(f, "Field `{field}` is not a field in `{item}`",)?;
            }
            ErrorKind::UnsupportedAssignExpr => {
                write!(f, "Cannot assign to expression")?;
            }
            ErrorKind::UnsupportedBinaryExpr => {
                write!(f, "Unsupported binary expression")?;
            }
            ErrorKind::UnsupportedRef => {
                write!(f, "Cannot take reference of expression")?;
            }
            ErrorKind::UnsupportedSelectPattern => {
                write!(f, "Unsupported select pattern")?;
            }
            ErrorKind::UnsupportedArgumentCount { expected, actual } => {
                write!(
                    f,
                    "Wrong number of arguments, expected `{expected}` but got `{actual}`",
                )?;
            }
            ErrorKind::UnsupportedPatternExpr => {
                write!(f, "This kind of expression is not supported as a pattern")?;
            }
            ErrorKind::UnsupportedBinding => {
                write!(f, "Not a valid binding")?;
            }
            ErrorKind::DuplicateObjectKey { .. } => {
                write!(f, "Duplicate key in literal object")?;
            }
            ErrorKind::InstanceFunctionOutsideImpl => {
                write!(f, "Instance function declared outside of `impl` block")?;
            }
            ErrorKind::UnsupportedTupleIndex { number } => {
                write!(f, "Unsupported tuple index `{number}`")?;
            }
            ErrorKind::BreakOutsideOfLoop => {
                write!(f, "Break outside of loop")?;
            }
            ErrorKind::ContinueOutsideOfLoop => {
                write!(f, "Continue outside of loop")?;
            }
            ErrorKind::SelectMultipleDefaults => {
                write!(f, "Multiple `default` branches in select")?;
            }
            ErrorKind::ExpectedBlockSemiColon { .. } => {
                write!(f, "Expected expression to be terminated by a semicolon `;`")?;
            }
            ErrorKind::FnConstAsyncConflict => {
                write!(
                    f,
                    "An `fn` can't both be `async` and `const` at the same time"
                )?;
            }
            ErrorKind::BlockConstAsyncConflict => {
                write!(
                    f,
                    "A block can't both be `async` and `const` at the same time"
                )?;
            }
            ErrorKind::ClosureKind => {
                write!(f, "Unsupported closure kind")?;
            }
            ErrorKind::UnsupportedSelfType => {
                write!(
                    f,
                    "Keyword `Self` is only supported inside of `impl` blocks"
                )?;
            }
            ErrorKind::UnsupportedSuper => {
                write!(
                    f,
                    "Keyword `super` is not supported at the root module level"
                )?;
            }
            ErrorKind::UnsupportedSuperInSelfType => {
                write!(
                    f,
                    "Keyword `super` can't be used in paths starting with `Self`"
                )?;
            }
            ErrorKind::UnsupportedAfterGeneric => {
                write!(
                    f,
                    "This kind of path component cannot follow a generic argument"
                )?;
            }
            ErrorKind::IllegalUseSegment => {
                write!(
                    f,
                    "Another segment can't follow wildcard `*` or group imports"
                )?;
            }
            ErrorKind::UseAliasNotSupported => {
                write!(
                    f,
                    "Use aliasing is not supported for wildcard `*` or group imports"
                )?;
            }
            ErrorKind::FunctionConflict { existing } => {
                write!(
                    f,
                    "Conflicting function signature already exists `{existing}`",
                )?;
            }
            ErrorKind::FunctionReExportConflict { hash } => {
                write!(f, "Conflicting function hash already exists `{hash}`",)?;
            }
            ErrorKind::ConstantConflict { hash } => {
                write!(f, "Conflicting constant for hash `{hash}`", hash = hash)?;
            }
            ErrorKind::StaticStringMissing { hash, slot } => {
                write!(
                    f,
                    "Missing static string for hash `{hash}` and slot `{slot}`",
                )?;
            }
            ErrorKind::StaticBytesMissing { hash, slot } => {
                write!(
                    f,
                    "Missing static byte string for hash `{hash}` and slot `{slot}`",
                )?;
            }
            ErrorKind::StaticStringHashConflict {
                hash,
                current,
                existing,
            } => {
                write!(f,"Conflicting static string for hash `{hash}`\n        between `{existing:?}` and `{current:?}`")?;
            }
            ErrorKind::StaticBytesHashConflict {
                hash,
                current,
                existing,
            } => {
                write!(f,"Conflicting static string for hash `{hash}`\n        between `{existing:?}` and `{current:?}`")?;
            }
            ErrorKind::StaticObjectKeysMissing { hash, slot } => {
                write!(
                    f,
                    "Missing static object keys for hash `{hash}` and slot `{slot}`",
                    hash = hash,
                    slot = slot
                )?;
            }
            ErrorKind::StaticObjectKeysHashConflict {
                hash,
                current,
                existing,
            } => {
                write!(f,"Conflicting static object keys for hash `{hash}`\n        between `{existing:?}` and `{current:?}`")?;
            }
            ErrorKind::MissingLoopLabel { label } => {
                write!(f, "Missing loop label `{label}`", label = label)?;
            }
            ErrorKind::ExpectedLeadingPathSegment => {
                write!(f, "Segment is only supported in the first position")?;
            }
            ErrorKind::UnsupportedVisibility => {
                write!(f, "Visibility modifier not supported")?;
            }
            ErrorKind::ExpectedMeta { expected, meta } => {
                write!(f, "Expected {expected} but got `{meta}`",)?;
            }
            ErrorKind::NoSuchBuiltInMacro { name } => {
                write!(f, "No such built-in macro `{name}`")?;
            }
            ErrorKind::VariableMoved { .. } => {
                write!(f, "Variable moved")?;
            }
            ErrorKind::UnsupportedGenerics => {
                write!(f, "Unsupported generic argument")?;
            }
            ErrorKind::NestedTest { .. } => {
                write!(f, "Attribute `#[test]` is not supported on nested items")?;
            }
            ErrorKind::NestedBench { .. } => {
                write!(f, "Attribute `#[bench]` is not supported on nested items")?;
            }
            ErrorKind::MissingFunctionHash { hash } => {
                write!(f, "Missing function with hash `{hash}`", hash = hash)?;
            }
            ErrorKind::FunctionConflictHash { hash } => {
                write!(
                    f,
                    "Conflicting function already exists `{hash}`",
                    hash = hash
                )?;
            }
            ErrorKind::PatternMissingFields { item, .. } => {
                write!(f, "Non-exhaustive pattern for `{item}`")?;
            }
            ErrorKind::MissingLabelLocation { name, index } => {
                write!(
                    f,
                    "Use of label `{name}_{index}` which has no code location",
                )?;
            }
            ErrorKind::MaxMacroRecursion { depth, max } => {
                write!(
                    f,
                    "Reached macro recursion limit at {depth}, limit is {max}",
                )?;
            }
            ErrorKind::YieldInConst => {
                write!(f, "Expression `yield` inside of constant function")?;
            }
            ErrorKind::AwaitInConst => {
                write!(f, "Expression `.await` inside of constant context")?;
            }
            ErrorKind::AwaitOutsideAsync => {
                write!(f, "Expression `.await` outside of async function or block")?;
            }
            ErrorKind::ExpectedEof { actual } => {
                write!(f, "Expected end of file, but got `{actual}`",)?;
            }
            ErrorKind::UnexpectedEof => {
                write!(f, "Unexpected end of file")?;
            }
            ErrorKind::BadLexerMode { actual, expected } => {
                write!(f, "Bad lexer mode `{actual}`, expected `{expected}`",)?;
            }
            ErrorKind::ExpectedEscape => {
                write!(f, "Expected escape sequence")?;
            }
            ErrorKind::UnterminatedStrLit => {
                write!(f, "Unterminated string literal")?;
            }
            ErrorKind::UnterminatedByteStrLit => {
                write!(f, "Unterminated byte string literal")?;
            }
            ErrorKind::UnterminatedCharLit => {
                write!(f, "Unterminated character literal")?;
            }
            ErrorKind::UnterminatedByteLit => {
                write!(f, "Unterminated byte literal")?;
            }
            ErrorKind::ExpectedCharClose => {
                write!(f, "Expected character literal to be closed")?;
            }
            ErrorKind::ExpectedCharOrLabel => {
                write!(f, "Expected label or character")?;
            }
            ErrorKind::ExpectedByteClose => {
                write!(f, "Expected byte literal to be closed")?;
            }
            ErrorKind::UnexpectedChar { c } => {
                write!(f, "Unexpected character `{c}`", c = c)?;
            }
            ErrorKind::PrecedenceGroupRequired => {
                write!(f, "Group required in expression to determine precedence")?;
            }
            ErrorKind::BadNumberOutOfBounds => {
                write!(
                    f,
                    "Number literal out of bounds `-9223372036854775808` to `9223372036854775807`"
                )?;
            }
            ErrorKind::BadFieldAccess => {
                write!(f, "Unsupported field access")?;
            }
            ErrorKind::ExpectedMacroCloseDelimiter { expected, actual } => {
                write!(
                    f,
                    "Expected close delimiter `{expected}`, but got `{actual}`",
                )?;
            }
            ErrorKind::MultipleMatchingAttributes { name } => {
                write!(f, "Can only specify one attribute named `{name}`",)?;
            }
            ErrorKind::MissingSourceId { source_id } => {
                write!(f, "Missing source id `{source_id}`", source_id = source_id)?;
            }
            ErrorKind::ExpectedMultilineCommentTerm => {
                write!(f, "Expected multiline comment to be terminated with a `*/`")?;
            }
            ErrorKind::BadSlice => {
                write!(f, "Tried to read bad slice from source")?;
            }
            ErrorKind::BadSyntheticId { kind, id } => {
                write!(
                    f,
                    "Tried to get bad synthetic identifier `{id}` for `{kind}`",
                )?;
            }
            ErrorKind::BadCharLiteral => {
                write!(f, "Bad character literal")?;
            }
            ErrorKind::BadByteLiteral => {
                write!(f, "Bad byte literal")?;
            }
            ErrorKind::BadNumberLiteral => {
                write!(f, "Number literal not valid")?;
            }
            ErrorKind::AmbiguousItem { item, .. } => {
                write!(f, "Item `{item}` can refer to multiple things")?;
            }
            ErrorKind::AmbiguousContextItem { item, .. } => {
                write!(
                    f,
                    "Item `{item}` can refer to multiple things from the context"
                )?;
            }
            ErrorKind::NotVisible {
                visibility,
                item,
                from,
                ..
            } => {
                write!(f,"Item `{item}` with visibility `{visibility}`, is not accessible from module `{from}`")?;
            }
            ErrorKind::NotVisibleMod {
                visibility,
                item,
                from,
                ..
            } => {
                write!(f,"Module `{item}` with {visibility} visibility, is not accessible from module `{from}`")?;
            }
            ErrorKind::MissingMod { item } => {
                write!(f, "Missing query meta for module {item}")?;
            }
            ErrorKind::ImportCycle { .. } => {
                write!(f, "Cycle in import")?;
            }
            ErrorKind::ImportRecursionLimit { count, .. } => {
                write!(f, "Import recursion limit reached ({count})", count = count)?;
            }
            ErrorKind::LastUseComponent => {
                write!(f, "Missing last use component")?;
            }
            ErrorKind::VariantRttiConflict { hash } => {
                write!(f,"Tried to insert variant runtime type information, but conflicted with hash `{hash}`")?;
            }
            ErrorKind::TypeRttiConflict { hash } => {
                write!(
                    f,
                    "Tried to insert runtime type information, but conflicted with hash `{hash}`",
                    hash = hash
                )?;
            }
            ErrorKind::ArenaWriteSliceOutOfBounds { index } => {
                write!(
                    f,
                    "Writing arena slice out of bounds for index {index}",
                    index = index
                )?;
            }
            ErrorKind::ArenaAllocError { requested } => {
                write!(
                    f,
                    "Allocation error for {requested} bytes",
                    requested = requested
                )?;
            }
            ErrorKind::UnsupportedPatternRest => {
                write!(f, "Pattern `..` is not supported in this location")?;
            }
            ErrorKind::UnsupportedMut => {
                write!(
                    f,
                    "The `mut` modifier is not supported in Rune, everything is mutable by default"
                )?;
            }
            ErrorKind::UnsupportedSuffix => {
                write!(
                    f,
                    "Unsupported suffix, expected one of `u8`, `i64`, or `f64`"
                )?;
            }
        }

        Ok(())
    }
}

impl From<IrErrorKind> for ErrorKind {
    #[inline]
    fn from(source: IrErrorKind) -> Self {
        ErrorKind::IrError(source)
    }
}

impl From<MetaConflict> for ErrorKind {
    #[inline]
    fn from(source: MetaConflict) -> Self {
        ErrorKind::MetaConflict(source)
    }
}

impl From<AccessError> for ErrorKind {
    #[inline]
    fn from(source: AccessError) -> Self {
        ErrorKind::AccessError(source)
    }
}

impl From<EncodeError> for ErrorKind {
    #[inline]
    fn from(source: EncodeError) -> Self {
        ErrorKind::EncodeError(source)
    }
}

impl From<MissingLastId> for ErrorKind {
    #[inline]
    fn from(source: MissingLastId) -> Self {
        ErrorKind::MissingLastId(source)
    }
}

impl From<GuardMismatch> for ErrorKind {
    #[inline]
    fn from(source: GuardMismatch) -> Self {
        ErrorKind::GuardMismatch(source)
    }
}

impl From<MissingScope> for ErrorKind {
    #[inline]
    fn from(source: MissingScope) -> Self {
        ErrorKind::MissingScope(source)
    }
}

impl From<PopError> for ErrorKind {
    #[inline]
    fn from(source: PopError) -> Self {
        ErrorKind::PopError(source)
    }
}

impl From<MissingId> for ErrorKind {
    #[inline]
    fn from(source: MissingId) -> Self {
        ErrorKind::MissingId(source)
    }
}

impl From<unescape::ErrorKind> for ErrorKind {
    #[inline]
    fn from(source: unescape::ErrorKind) -> Self {
        ErrorKind::UnescapeError(source)
    }
}

/// Error when encoding AST.
#[derive(Debug)]
#[non_exhaustive]
pub(crate) enum IrErrorKind {
    /// Encountered an expression that is not supported as a constant
    /// expression.
    NotConst,
    /// Trying to process a cycle of constants.
    ConstCycle,
    /// Encountered a compile meta used in an inappropriate position.
    UnsupportedMeta {
        /// Unsupported compile meta.
        meta: MetaInfo,
    },
    /// A constant evaluation errored.
    Expected {
        /// The expected value.
        expected: TypeInfo,
        /// The value we got instead.
        actual: TypeInfo,
    },
    /// Exceeded evaluation budget.
    BudgetExceeded,
    /// Missing a tuple index.
    MissingIndex {
        /// The index that was missing.
        index: usize,
    },
    /// Missing an object field.
    MissingField {
        /// The field that was missing.
        field: Box<str>,
    },
    /// Missing const or local with the given name.
    MissingConst {
        /// Name of the missing thing.
        name: Box<str>,
    },
    /// Error raised when trying to use a break outside of a loop.
    BreakOutsideOfLoop,
    ArgumentCountMismatch {
        actual: usize,
        expected: usize,
    },
}

impl crate::no_std::error::Error for IrErrorKind {}

impl fmt::Display for IrErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            IrErrorKind::NotConst => {
                write!(f, "Expected a constant expression")?;
            }
            IrErrorKind::ConstCycle => {
                write!(f, "Constant cycle detected")?;
            }
            IrErrorKind::UnsupportedMeta { meta } => {
                write!(f, "Item `{meta}` is not supported here",)?
            }
            IrErrorKind::Expected { expected, actual } => {
                write!(f, "Expected a value of type {expected} but got {actual}",)?
            }
            IrErrorKind::BudgetExceeded => {
                write!(f, "Evaluation budget exceeded")?;
            }
            IrErrorKind::MissingIndex { index } => {
                write!(f, "Missing index {index}",)?;
            }
            IrErrorKind::MissingField { field } => {
                write!(f, "Missing field `{field}`",)?;
            }
            IrErrorKind::MissingConst { name } => {
                write!(f, "No constant or local matching `{name}`",)?;
            }
            IrErrorKind::BreakOutsideOfLoop => {
                write!(f, "Break outside of supported loop")?;
            }
            IrErrorKind::ArgumentCountMismatch { actual, expected } => {
                write!(
                    f,
                    "Argument count mismatch, got {actual} but expected {expected}",
                )?;
            }
        }

        Ok(())
    }
}

/// A single step in an import.
///
/// This is used to indicate a step in an import chain in an error message.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ImportStep {
    /// The location of the import.
    pub location: Location,
    /// The item being imported.
    pub item: ItemBuf,
}

#[derive(Debug)]
/// Tried to add an item that already exists.
pub(crate) struct MetaConflict {
    /// The meta we tried to insert.
    pub(crate) current: MetaInfo,
    /// The existing item.
    pub(crate) existing: MetaInfo,
    /// Parameters hash.
    pub(crate) parameters: Hash,
}

impl fmt::Display for MetaConflict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let MetaConflict {
            current,
            existing,
            parameters,
        } = self;
        write!(f, "Can't insert item `{current}` ({parameters}) because conflicting meta `{existing}` already exists")
    }
}

impl crate::no_std::error::Error for MetaConflict {}

#[derive(Debug)]
pub(crate) struct MissingScope(pub(crate) usize);

impl fmt::Display for MissingScope {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Missing scope with id {}", self.0)
    }
}

impl crate::no_std::error::Error for MissingScope {}

#[derive(Debug)]
pub(crate) enum PopError {
    MissingScope(usize),
    MissingParentScope(usize),
}

impl fmt::Display for PopError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PopError::MissingScope(id) => write!(f, "Missing scope with id {id}"),
            PopError::MissingParentScope(id) => write!(f, "Missing parent scope with id {id}"),
        }
    }
}

impl crate::no_std::error::Error for PopError {}
