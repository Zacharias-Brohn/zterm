//! Configuration management for ZTerm.
//!
//! Loads configuration from `~/.config/zterm/config.json`.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

/// Position of the tab bar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum TabBarPosition {
    /// Tab bar at the top of the window.
    #[default]
    Top,
    /// Tab bar at the bottom of the window.
    Bottom,
    /// Tab bar is hidden.
    Hidden,
}

/// A keybinding specification.
/// Format: "modifier+modifier+key" where modifiers are: ctrl, alt, shift, super
/// Examples: "ctrl+shift+t", "ctrl+w", "alt+1"
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(transparent)]
pub struct Keybind(pub String);

impl Keybind {
    /// Parses the keybind into modifiers and key.
    /// Returns (ctrl, alt, shift, super_key, key_char_or_name)
    ///
    /// Supports special syntax for symbol keys:
    /// - "ctrl+alt+plus" or "ctrl+alt++" for the + key
    /// - "ctrl+minus" or "ctrl+-" for the - key
    /// - Symbol names: plus, minus, equal, bracket_left, bracket_right, etc.
    pub fn parse(&self) -> Option<(bool, bool, bool, bool, String)> {
        let lowercase = self.0.to_lowercase();
        
        // Handle the special case where the key is "+" at the end
        // e.g., "ctrl+alt++" should parse as ctrl+alt with key "+"
        let (modifier_part, key) = if lowercase.ends_with("++") {
            // Last char is the key "+", everything before the final "++" is modifiers
            let prefix = &lowercase[..lowercase.len() - 2];
            (prefix, "+".to_string())
        } else if lowercase == "+" {
            // Just the plus key alone
            ("", "+".to_string())
        } else if let Some(last_plus) = lowercase.rfind('+') {
            // Normal case: split at last +
            let key_part = &lowercase[last_plus + 1..];
            let mod_part = &lowercase[..last_plus];
            // Normalize symbol names to actual characters
            let key = Self::normalize_key_name(key_part)
                .map(|s| s.to_string())
                .unwrap_or_else(|| key_part.to_string());
            (mod_part, key)
        } else {
            // No modifiers, just a key
            let key = Self::normalize_key_name(&lowercase)
                .map(|s| s.to_string())
                .unwrap_or_else(|| lowercase.clone());
            ("", key)
        };
        
        if key.is_empty() {
            return None;
        }
        
        let mut ctrl = false;
        let mut alt = false;
        let mut shift = false;
        let mut super_key = false;
        
        // Parse modifiers from the modifier part
        for part in modifier_part.split('+') {
            match part {
                "ctrl" | "control" => ctrl = true,
                "alt" => alt = true,
                "shift" => shift = true,
                "super" | "meta" | "cmd" => super_key = true,
                "" => {} // Empty parts from splitting
                _ => {} // Unknown modifiers ignored
            }
        }
        
        Some((ctrl, alt, shift, super_key, key))
    }
    
    /// Normalizes key names to their canonical form.
    /// Supports both symbol names ("plus", "minus") and literal symbols ("+", "-").
    /// Returns a static str for known keys, None for unknown (caller uses input).
    fn normalize_key_name(name: &str) -> Option<&'static str> {
        Some(match name {
            // Arrow keys
            "left" | "arrowleft" | "arrow_left" => "left",
            "right" | "arrowright" | "arrow_right" => "right",
            "up" | "arrowup" | "arrow_up" => "up",
            "down" | "arrowdown" | "arrow_down" => "down",
            
            // Other special keys
            "enter" | "return" => "enter",
            "tab" => "tab",
            "escape" | "esc" => "escape",
            "backspace" | "back" => "backspace",
            "delete" | "del" => "delete",
            "insert" | "ins" => "insert",
            "home" => "home",
            "end" => "end",
            "pageup" | "page_up" | "pgup" => "pageup",
            "pagedown" | "page_down" | "pgdn" => "pagedown",
            
            // Function keys
            "f1" => "f1",
            "f2" => "f2",
            "f3" => "f3",
            "f4" => "f4",
            "f5" => "f5",
            "f6" => "f6",
            "f7" => "f7",
            "f8" => "f8",
            "f9" => "f9",
            "f10" => "f10",
            "f11" => "f11",
            "f12" => "f12",
            
            // Symbol name aliases
            "plus" => "+",
            "minus" => "-",
            "equal" | "equals" => "=",
            "bracket_left" | "bracketleft" | "lbracket" => "[",
            "bracket_right" | "bracketright" | "rbracket" => "]",
            "brace_left" | "braceleft" | "lbrace" => "{",
            "brace_right" | "braceright" | "rbrace" => "}",
            "semicolon" => ";",
            "colon" => ":",
            "apostrophe" | "quote" => "'",
            "quotedbl" | "doublequote" => "\"",
            "comma" => ",",
            "period" | "dot" => ".",
            "slash" => "/",
            "backslash" => "\\",
            "grave" | "backtick" => "`",
            "tilde" => "~",
            "at" => "@",
            "hash" | "pound" => "#",
            "dollar" => "$",
            "percent" => "%",
            "caret" => "^",
            "ampersand" => "&",
            "asterisk" | "star" => "*",
            "paren_left" | "parenleft" | "lparen" => "(",
            "paren_right" | "parenright" | "rparen" => ")",
            "underscore" => "_",
            "pipe" | "bar" => "|",
            "question" => "?",
            "exclam" | "exclamation" | "bang" => "!",
            "less" | "lessthan" => "<",
            "greater" | "greaterthan" => ">",
            "space" => " ",
            // Unknown - caller handles passthrough
            _ => return None,
        })
    }
}

/// Terminal actions that can be bound to keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Action {
    /// Create a new tab.
    NewTab,
    /// Switch to the next tab.
    NextTab,
    /// Switch to the previous tab.
    PrevTab,
    /// Switch to tab by index (1-9).
    Tab1,
    Tab2,
    Tab3,
    Tab4,
    Tab5,
    Tab6,
    Tab7,
    Tab8,
    Tab9,
    /// Split pane horizontally (new pane below).
    SplitHorizontal,
    /// Split pane vertically (new pane to the right).
    SplitVertical,
    /// Close the current pane (closes tab if last pane).
    ClosePane,
    /// Focus the pane above the current one.
    FocusPaneUp,
    /// Focus the pane below the current one.
    FocusPaneDown,
    /// Focus the pane to the left of the current one.
    FocusPaneLeft,
    /// Focus the pane to the right of the current one.
    FocusPaneRight,
    /// Copy selection to clipboard.
    Copy,
    /// Paste from clipboard.
    Paste,
}

/// Keybinding configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Keybindings {
    /// Create new tab.
    pub new_tab: Keybind,
    /// Switch to next tab.
    pub next_tab: Keybind,
    /// Switch to previous tab.
    pub prev_tab: Keybind,
    /// Switch to tab 1.
    pub tab_1: Keybind,
    /// Switch to tab 2.
    pub tab_2: Keybind,
    /// Switch to tab 3.
    pub tab_3: Keybind,
    /// Switch to tab 4.
    pub tab_4: Keybind,
    /// Switch to tab 5.
    pub tab_5: Keybind,
    /// Switch to tab 6.
    pub tab_6: Keybind,
    /// Switch to tab 7.
    pub tab_7: Keybind,
    /// Switch to tab 8.
    pub tab_8: Keybind,
    /// Switch to tab 9.
    pub tab_9: Keybind,
    /// Split pane horizontally (new pane below).
    pub split_horizontal: Keybind,
    /// Split pane vertically (new pane to the right).
    pub split_vertical: Keybind,
    /// Close current pane (closes tab if last pane).
    pub close_pane: Keybind,
    /// Focus pane above.
    pub focus_pane_up: Keybind,
    /// Focus pane below.
    pub focus_pane_down: Keybind,
    /// Focus pane to the left.
    pub focus_pane_left: Keybind,
    /// Focus pane to the right.
    pub focus_pane_right: Keybind,
    /// Copy selection to clipboard.
    pub copy: Keybind,
    /// Paste from clipboard.
    pub paste: Keybind,
}

impl Default for Keybindings {
    fn default() -> Self {
        Self {
            new_tab: Keybind("ctrl+shift+t".to_string()),
            next_tab: Keybind("ctrl+tab".to_string()),
            prev_tab: Keybind("ctrl+shift+tab".to_string()),
            tab_1: Keybind("alt+1".to_string()),
            tab_2: Keybind("alt+2".to_string()),
            tab_3: Keybind("alt+3".to_string()),
            tab_4: Keybind("alt+4".to_string()),
            tab_5: Keybind("alt+5".to_string()),
            tab_6: Keybind("alt+6".to_string()),
            tab_7: Keybind("alt+7".to_string()),
            tab_8: Keybind("alt+8".to_string()),
            tab_9: Keybind("alt+9".to_string()),
            split_horizontal: Keybind("ctrl+shift+h".to_string()),
            split_vertical: Keybind("ctrl+shift+e".to_string()),
            close_pane: Keybind("ctrl+shift+w".to_string()),
            focus_pane_up: Keybind("ctrl+shift+up".to_string()),
            focus_pane_down: Keybind("ctrl+shift+down".to_string()),
            focus_pane_left: Keybind("ctrl+shift+left".to_string()),
            focus_pane_right: Keybind("ctrl+shift+right".to_string()),
            copy: Keybind("ctrl+shift+c".to_string()),
            paste: Keybind("ctrl+shift+v".to_string()),
        }
    }
}

impl Keybindings {
    /// Builds a lookup map from parsed keybinds to actions.
    pub fn build_action_map(&self) -> HashMap<(bool, bool, bool, bool, String), Action> {
        let mut map = HashMap::new();
        
        let bindings: &[(&Keybind, Action)] = &[
            (&self.new_tab, Action::NewTab),
            (&self.next_tab, Action::NextTab),
            (&self.prev_tab, Action::PrevTab),
            (&self.tab_1, Action::Tab1),
            (&self.tab_2, Action::Tab2),
            (&self.tab_3, Action::Tab3),
            (&self.tab_4, Action::Tab4),
            (&self.tab_5, Action::Tab5),
            (&self.tab_6, Action::Tab6),
            (&self.tab_7, Action::Tab7),
            (&self.tab_8, Action::Tab8),
            (&self.tab_9, Action::Tab9),
            (&self.split_horizontal, Action::SplitHorizontal),
            (&self.split_vertical, Action::SplitVertical),
            (&self.close_pane, Action::ClosePane),
            (&self.focus_pane_up, Action::FocusPaneUp),
            (&self.focus_pane_down, Action::FocusPaneDown),
            (&self.focus_pane_left, Action::FocusPaneLeft),
            (&self.focus_pane_right, Action::FocusPaneRight),
            (&self.copy, Action::Copy),
            (&self.paste, Action::Paste),
        ];
        
        for (keybind, action) in bindings {
            if let Some(parsed) = keybind.parse() {
                map.insert(parsed, *action);
            }
        }
        
        map
    }
}

/// Main configuration struct for ZTerm.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    /// Font family name to use. The terminal will look for Regular, Bold, Italic, and BoldItalic
    /// variants of this font. If not specified, falls back to system monospace fonts.
    /// Example: "JetBrainsMono Nerd Font" or "0xProto Nerd Font"
    pub font_family: Option<String>,
    /// Font size in points.
    pub font_size: f32,
    /// Position of the tab bar: "top", "bottom", or "hidden".
    pub tab_bar_position: TabBarPosition,
    /// Background opacity (0.0 = fully transparent, 1.0 = fully opaque).
    /// Requires compositor support for transparency.
    pub background_opacity: f32,
    /// Number of lines to keep in scrollback buffer.
    pub scrollback_lines: usize,
    /// Duration in milliseconds for the inactive pane fade animation.
    /// Set to 0 for instant transitions.
    pub inactive_pane_fade_ms: u64,
    /// Dim factor for inactive panes (0.0 = fully dimmed/black, 1.0 = no dimming).
    pub inactive_pane_dim: f32,
    /// Intensity of the edge glow effect when pane navigation fails (0.0 = disabled, 1.0 = full intensity).
    /// The edge glow provides visual feedback when you try to navigate to a pane that doesn't exist.
    pub edge_glow_intensity: f32,
    /// Process names that should receive pane navigation keys instead of zterm handling them.
    /// When the foreground process matches one of these names, Alt+Arrow keys are passed
    /// to the application (e.g., for Neovim buffer navigation) instead of switching panes.
    /// Example: ["nvim", "vim", "helix"]
    pub pass_keys_to_programs: Vec<String>,
    /// Keybindings.
    pub keybindings: Keybindings,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            font_family: None,
            font_size: 16.0,
            tab_bar_position: TabBarPosition::Top,
            background_opacity: 1.0,
            scrollback_lines: 50_000,
            inactive_pane_fade_ms: 150,
            inactive_pane_dim: 0.6,
            edge_glow_intensity: 1.0,
            pass_keys_to_programs: vec!["nvim".to_string(), "vim".to_string()],
            keybindings: Keybindings::default(),
        }
    }
}

impl Config {
    /// Returns the path to the config file.
    pub fn config_path() -> Option<PathBuf> {
        dirs::config_dir().map(|p| p.join("zterm").join("config.json"))
    }

    /// Loads configuration from the default config file.
    /// If the file doesn't exist, writes the default config to that location.
    /// Returns default config if file can't be parsed.
    pub fn load() -> Self {
        let Some(config_path) = Self::config_path() else {
            log::warn!("Could not determine config directory, using defaults");
            return Self::default();
        };

        if !config_path.exists() {
            log::info!("No config file found at {:?}, creating with defaults", config_path);
            let default_config = Self::default();
            if let Err(e) = default_config.save() {
                log::warn!("Failed to write default config: {}", e);
            }
            return default_config;
        }

        match fs::read_to_string(&config_path) {
            Ok(contents) => match serde_json::from_str(&contents) {
                Ok(config) => {
                    log::info!("Loaded config from {:?}", config_path);
                    config
                }
                Err(e) => {
                    log::error!("Failed to parse config file: {}", e);
                    Self::default()
                }
            },
            Err(e) => {
                log::error!("Failed to read config file: {}", e);
                Self::default()
            }
        }
    }

    /// Saves the current configuration to the default config file.
    pub fn save(&self) -> Result<(), std::io::Error> {
        let Some(config_path) = Self::config_path() else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "Could not determine config directory",
            ));
        };

        // Create parent directories if they don't exist
        if let Some(parent) = config_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        fs::write(&config_path, json)?;
        log::info!("Saved config to {:?}", config_path);
        Ok(())
    }
}
