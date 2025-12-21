//! Statusline types and rendering.
//!
//! Provides data structures for building structured statusline content with
//! powerline-style sections and components.

/// Color specification for statusline components.
/// Uses the terminal's indexed color palette (0-255), RGB, or default fg.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatuslineColor {
    /// Use the default foreground color.
    Default,
    /// Use an indexed color from the 256-color palette (0-15 for ANSI colors).
    Indexed(u8),
    /// Use an RGB color.
    Rgb(u8, u8, u8),
}

impl Default for StatuslineColor {
    fn default() -> Self {
        StatuslineColor::Default
    }
}

/// A single component/segment of the statusline.
/// Components are rendered left-to-right with optional separators.
#[derive(Debug, Clone)]
pub struct StatuslineComponent {
    /// The text content of this component.
    pub text: String,
    /// Foreground color for this component.
    pub fg: StatuslineColor,
    /// Whether this text should be bold.
    pub bold: bool,
}

impl StatuslineComponent {
    /// Create a new statusline component with default styling.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            fg: StatuslineColor::Default,
            bold: false,
        }
    }

    /// Set the foreground color using an indexed palette color.
    pub fn fg(mut self, color_index: u8) -> Self {
        self.fg = StatuslineColor::Indexed(color_index);
        self
    }

    /// Set the foreground color using RGB values.
    pub fn rgb_fg(mut self, r: u8, g: u8, b: u8) -> Self {
        self.fg = StatuslineColor::Rgb(r, g, b);
        self
    }

    /// Set bold styling.
    pub fn bold(mut self) -> Self {
        self.bold = true;
        self
    }

    /// Create a separator component (e.g., "/", " > ", etc.).
    pub fn separator(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            fg: StatuslineColor::Indexed(8), // Dim gray by default
            bold: false,
        }
    }
}

/// A section of the statusline with its own background color.
/// Sections are rendered left-to-right and end with a powerline transition arrow.
#[derive(Debug, Clone)]
pub struct StatuslineSection {
    /// The components within this section.
    pub components: Vec<StatuslineComponent>,
    /// Background color for this section.
    pub bg: StatuslineColor,
}

impl StatuslineSection {
    /// Create a new section with the given indexed background color.
    pub fn new(bg_color: u8) -> Self {
        Self {
            components: Vec::new(),
            bg: StatuslineColor::Indexed(bg_color),
        }
    }

    /// Create a new section with an RGB background color.
    pub fn with_rgb_bg(r: u8, g: u8, b: u8) -> Self {
        Self {
            components: Vec::new(),
            bg: StatuslineColor::Rgb(r, g, b),
        }
    }

    /// Create a new section with the default (transparent) background.
    pub fn transparent() -> Self {
        Self {
            components: Vec::new(),
            bg: StatuslineColor::Default,
        }
    }

    /// Add a component to this section.
    pub fn push(mut self, component: StatuslineComponent) -> Self {
        self.components.push(component);
        self
    }

    /// Add multiple components to this section.
    pub fn with_components(mut self, components: Vec<StatuslineComponent>) -> Self {
        self.components = components;
        self
    }
}

/// Content to display in the statusline.
/// Either structured sections (for ZTerm's default CWD/git display) or raw ANSI
/// content (from neovim or other programs that provide their own statusline).
#[derive(Debug, Clone)]
pub enum StatuslineContent {
    /// Structured sections with powerline-style transitions.
    Sections(Vec<StatuslineSection>),
    /// Raw ANSI-formatted string (rendered as-is without section styling).
    Raw(String),
}

impl Default for StatuslineContent {
    fn default() -> Self {
        StatuslineContent::Sections(Vec::new())
    }
}
