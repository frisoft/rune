//! Helpers for building assembly.

use std::collections::BTreeMap;
use std::fmt;

use crate::ast::Span;
use crate::collections::HashMap;
use crate::compile::{CompileError, CompileErrorKind, Label, Location};
use crate::runtime::AssemblyInst;
use crate::{Hash, SourceId};

type Result<T> = ::std::result::Result<T, CompileError>;

/// The address as declared by a scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum AssemblyAddress {
    /// A slot address.
    Slot(usize),
    /// An array address.
    Array(usize),
}

/// Helper structure to build instructions and maintain certain invariants.
#[derive(Debug, Default)]
pub(crate) struct Assembly {
    /// The location that caused the assembly.
    pub(crate) location: Location,
    /// Label to offset.
    pub(crate) labels: HashMap<Label, usize>,
    /// Registered label by offset.
    pub(crate) labels_rev: BTreeMap<usize, Label>,
    /// Instructions with spans.
    pub(crate) instructions: Vec<(Span, AssemblyInst)>,
    /// Comments associated with instructions.
    pub(crate) comments: HashMap<usize, Vec<Box<str>>>,
    /// The number of labels.
    pub(crate) label_count: usize,
    /// The collection of functions required by this assembly.
    pub(crate) required_functions: HashMap<Hash, Vec<(Span, SourceId)>>,
}

impl Assembly {
    /// Construct a new assembly.
    pub(crate) fn new(location: Location, label_count: usize) -> Self {
        Self {
            location,
            labels: Default::default(),
            labels_rev: Default::default(),
            instructions: Default::default(),
            comments: Default::default(),
            label_count,
            required_functions: Default::default(),
        }
    }

    /// Construct and return a new label.
    pub(crate) fn new_label(&mut self, name: &'static str) -> Label {
        let label = Label::new(name, self.label_count);
        self.label_count += 1;
        label
    }

    /// Apply the label at the current instruction offset.
    pub(crate) fn label(&mut self, span: Span, label: Label) -> Result<()> {
        let offset = self.instructions.len();

        if self.labels.insert(label, offset).is_some() {
            return Err(CompileError::new(
                span,
                CompileErrorKind::DuplicateLabel {
                    label: label.to_string().into(),
                },
            ));
        }

        self.labels_rev.insert(offset, label);
        Ok(())
    }

    /// Push an instruction.
    pub(crate) fn push(&mut self, span: Span, inst: AssemblyInst) {
        if let AssemblyInst::Call { hash, .. } = inst {
            self.required_functions
                .entry(hash)
                .or_default()
                .push((span, self.location.source_id));
        }

        self.instructions.push((span, inst));
    }

    /// Push an instruction with a custom comment.
    pub(crate) fn push_with_comment<C>(&mut self, span: Span, inst: AssemblyInst, comment: C)
    where
        C: fmt::Display,
    {
        let pos = self.instructions.len();

        self.comments
            .entry(pos)
            .or_default()
            .push(comment.to_string().into());

        self.push(span, inst);
    }
}
