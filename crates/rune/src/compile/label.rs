//! A simple label used to jump to a code location.

use std::fmt;

use crate::runtime::DebugLabel;

/// A label that can be jumped to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Label {
    name: &'static str,
    id: usize,
}

impl Label {
    /// Construct a new label.
    pub(crate) const fn new(name: &'static str, id: usize) -> Self {
        Self { name, id }
    }

    /// Convert into owned label.
    pub(crate) fn into_owned(self) -> DebugLabel {
        DebugLabel::new(self.name.into(), self.id)
    }
}

impl fmt::Display for Label {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}_{}", self.name, self.id)
    }
}
