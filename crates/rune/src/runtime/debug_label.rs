use std::borrow::Cow;
use std::fmt;

use serde::{Deserialize, Serialize};

/// A label that can be jumped to.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DebugLabel {
    /// The name of the label.
    name: Cow<'static, str>,
    /// The id of the label.
    id: usize,
}

impl DebugLabel {
    pub(crate) const fn new(name: Cow<'static, str>, id: usize) -> Self {
        Self { name, id }
    }
}

impl fmt::Display for DebugLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}_{}", self.name, self.id)
    }
}
