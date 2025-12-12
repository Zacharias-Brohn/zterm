//! ZTerm Client - GPU-accelerated terminal emulator that connects to the daemon.

use zterm::client::DaemonClient;
use zterm::config::{Action, Config};
use zterm::keyboard::{FunctionalKey, KeyEncoder, KeyEventType, KeyboardState, Modifiers};
use zterm::protocol::{ClientMessage, DaemonMessage, Direction, PaneId, PaneInfo, PaneSnapshot, WindowState};
use zterm::renderer::Renderer;

use polling::{Event, Events, Poller};
use std::collections::HashMap;
use std::os::fd::{AsRawFd, BorrowedFd};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, KeyEvent, Modifiers as WinitModifiers, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::keyboard::{Key, NamedKey};
use winit::platform::wayland::EventLoopBuilderExtWayland;
use winit::window::{Window, WindowId};

/// Main application state.
struct App {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    daemon_client: Option<DaemonClient>,
    /// Current window state (tabs info) from daemon.
    window_state: Option<WindowState>,
    /// All pane snapshots from daemon.
    panes: Vec<PaneSnapshot>,
    /// Whether we need to redraw.
    dirty: bool,
    /// Current modifier state.
    modifiers: WinitModifiers,
    /// Keyboard state for encoding (tracks protocol mode from daemon).
    keyboard_state: KeyboardState,
    /// Application configuration.
    config: Config,
    /// Keybinding action map.
    action_map: HashMap<(bool, bool, bool, bool, String), Action>,
    /// Event loop proxy for waking from daemon poll thread.
    event_loop_proxy: Option<EventLoopProxy<()>>,
    /// Shutdown signal for daemon poll thread.
    shutdown: Arc<AtomicBool>,
}

const DAEMON_SOCKET_KEY: usize = 1;

impl App {
    fn new() -> Self {
        let config = Config::load();
        log::info!("Config: font_size={}", config.font_size);
        
        // Build action map from keybindings
        let action_map = config.keybindings.build_action_map();
        
        Self {
            window: None,
            renderer: None,
            daemon_client: None,
            window_state: None,
            panes: Vec::new(),
            dirty: true,
            modifiers: WinitModifiers::default(),
            keyboard_state: KeyboardState::new(),
            config,
            action_map,
            event_loop_proxy: None,
            shutdown: Arc::new(AtomicBool::new(false)),
        }
    }
    
    fn set_event_loop_proxy(&mut self, proxy: EventLoopProxy<()>) {
        self.event_loop_proxy = Some(proxy);
    }

    fn initialize(&mut self, event_loop: &ActiveEventLoop) {
        let init_start = std::time::Instant::now();
        
        // Create window first so it appears immediately
        let mut window_attributes = Window::default_attributes()
            .with_title("ZTerm")
            .with_inner_size(PhysicalSize::new(800, 600));

        // Enable transparency if background opacity is less than 1.0
        if self.config.background_opacity < 1.0 {
            window_attributes = window_attributes.with_transparent(true);
        }

        let window = Arc::new(
            event_loop
                .create_window(window_attributes)
                .expect("Failed to create window"),
        );
        log::debug!("Window created in {:?}", init_start.elapsed());

        // Start daemon connection in parallel with renderer initialization
        let daemon_start = std::time::Instant::now();
        let mut daemon_client = match DaemonClient::connect() {
            Ok(client) => client,
            Err(e) => {
                log::error!("Failed to connect to daemon: {}", e);
                event_loop.exit();
                return;
            }
        };
        log::debug!("Daemon connected in {:?}", daemon_start.elapsed());

        // Create renderer (this is the slow part - GPU initialization)
        let renderer_start = std::time::Instant::now();
        let renderer = pollster::block_on(Renderer::new(window.clone(), &self.config));
        log::debug!("Renderer created in {:?}", renderer_start.elapsed());

        // Calculate terminal size based on window size
        let (cols, rows) = renderer.terminal_size();

        // Send hello with our size
        if let Err(e) = daemon_client.hello(cols, rows) {
            log::error!("Failed to send hello: {}", e);
            event_loop.exit();
            return;
        }

        // Wait for initial state
        match daemon_client.recv() {
            Ok(DaemonMessage::FullState { window: win_state, panes }) => {
                log::debug!("Received initial state with {} tabs, {} panes", 
                    win_state.tabs.len(), panes.len());
                self.window_state = Some(win_state);
                self.panes = panes;
            }
            Ok(msg) => {
                log::warn!("Unexpected initial message: {:?}", msg);
            }
            Err(e) => {
                log::error!("Failed to receive initial state: {}", e);
                event_loop.exit();
                return;
            }
        }
        
        // Switch to non-blocking mode for the event loop
        if let Err(e) = daemon_client.set_nonblocking() {
            log::error!("Failed to set non-blocking mode: {}", e);
            event_loop.exit();
            return;
        }

        // Set up polling for daemon socket in a background thread
        // This thread will wake the event loop when data is available
        if let Some(proxy) = self.event_loop_proxy.clone() {
            let daemon_fd = daemon_client.as_raw_fd();
            let shutdown = self.shutdown.clone();
            
            std::thread::spawn(move || {
                let poller = match Poller::new() {
                    Ok(p) => p,
                    Err(e) => {
                        log::error!("Failed to create poller: {}", e);
                        return;
                    }
                };
                
                // SAFETY: daemon_fd is valid for the lifetime of the daemon_client,
                // and we signal shutdown before dropping daemon_client
                unsafe {
                    if let Err(e) = poller.add(daemon_fd, Event::readable(DAEMON_SOCKET_KEY)) {
                        log::error!("Failed to add daemon socket to poller: {}", e);
                        return;
                    }
                }
                
                let mut events = Events::new();
                
                while !shutdown.load(Ordering::Relaxed) {
                    events.clear();
                    
                    // Wait for data with a timeout so we can check shutdown
                    match poller.wait(&mut events, Some(Duration::from_millis(100))) {
                        Ok(_) if !events.is_empty() => {
                            // Wake the event loop by sending an empty event
                            let _ = proxy.send_event(());
                            
                            // Re-register for more events
                            // SAFETY: daemon_fd is still valid
                            unsafe {
                                let _ = poller.modify(
                                    std::os::fd::BorrowedFd::borrow_raw(daemon_fd),
                                    Event::readable(DAEMON_SOCKET_KEY)
                                );
                            }
                        }
                        Ok(_) => {} // Timeout, no events
                        Err(e) => {
                            log::error!("Poller error: {}", e);
                            break;
                        }
                    }
                }
                
                log::debug!("Daemon poll thread exiting");
            });
        }

        self.window = Some(window);
        self.renderer = Some(renderer);
        self.daemon_client = Some(daemon_client);

        log::info!("Client initialized in {:?}: {}x{} cells", init_start.elapsed(), cols, rows);
    }

    /// Gets all pane snapshots with their layout info for the active tab.
    /// Returns (panes_with_info, active_pane_id) with cloned/owned data.
    fn active_tab_panes(&self) -> (Vec<(PaneSnapshot, PaneInfo)>, PaneId) {
        let Some(win) = self.window_state.as_ref() else {
            return (Vec::new(), 0);
        };
        let Some(tab) = win.tabs.get(win.active_tab) else {
            return (Vec::new(), 0);
        };
        
        let active_pane_id = tab.panes.get(tab.active_pane)
            .map(|p| p.id)
            .unwrap_or(0);
        
        let panes_with_info: Vec<(PaneSnapshot, PaneInfo)> = tab.panes.iter()
            .filter_map(|pane_info| {
                self.panes.iter()
                    .find(|snap| snap.pane_id == pane_info.id)
                    .map(|snap| (snap.clone(), pane_info.clone()))
            })
            .collect();
        
        (panes_with_info, active_pane_id)
    }

    /// Gets the active pane ID and its snapshot.
    fn get_active_pane(&self) -> Option<(PaneId, &PaneSnapshot)> {
        let win = self.window_state.as_ref()?;
        let tab = win.tabs.get(win.active_tab)?;
        let pane_info = tab.panes.get(tab.active_pane)?;
        let snapshot = self.panes.iter().find(|s| s.pane_id == pane_info.id)?;
        Some((pane_info.id, snapshot))
    }

    fn poll_daemon(&mut self) {
        let Some(client) = &mut self.daemon_client else { return };

        // Read all available messages (non-blocking)
        // The background thread wakes us when data is available
        let mut messages = Vec::new();
        loop {
            match client.try_recv() {
                Ok(Some(msg)) => {
                    messages.push(msg);
                }
                Ok(None) => break,
                Err(e) => {
                    log::error!("Daemon connection error: {}", e);
                    // Daemon disconnected - we'll handle this after the loop
                    messages.push(DaemonMessage::Shutdown);
                    break;
                }
            }
        }

        // Now process messages without holding the client borrow
        for msg in messages {
            self.handle_daemon_message(msg);
        }
    }

    fn handle_daemon_message(&mut self, msg: DaemonMessage) {
        match msg {
            DaemonMessage::FullState { window, panes } => {
                log::debug!("Received full state: {} tabs, {} panes", 
                    window.tabs.len(), panes.len());
                self.window_state = Some(window);
                self.panes = panes;
                self.dirty = true;
            }
            DaemonMessage::PaneUpdate { pane_id, cells, cursor } => {
                log::debug!("Received pane update for pane {}", pane_id);
                if let Some(pane) = self.panes.iter_mut().find(|p| p.pane_id == pane_id) {
                    pane.cells = cells;
                    pane.cursor = cursor;
                } else {
                    // New pane
                    self.panes.push(PaneSnapshot {
                        pane_id,
                        cells,
                        cursor,
                        scroll_offset: 0,
                        scrollback_len: 0,
                    });
                }
                self.dirty = true;
            }
            DaemonMessage::TabChanged { active_tab } => {
                log::debug!("Tab changed to {}", active_tab);
                if let Some(ref mut win) = self.window_state {
                    win.active_tab = active_tab;
                }
                self.dirty = true;
            }
            DaemonMessage::TabCreated { tab } => {
                log::debug!("Tab created: {:?}", tab);
                if let Some(ref mut win) = self.window_state {
                    win.tabs.push(tab);
                }
                self.dirty = true;
            }
            DaemonMessage::TabClosed { tab_id } => {
                log::debug!("Tab closed: {}", tab_id);
                if let Some(ref mut win) = self.window_state {
                    win.tabs.retain(|t| t.id != tab_id);
                    // Adjust active tab if needed
                    if win.active_tab >= win.tabs.len() && !win.tabs.is_empty() {
                        win.active_tab = win.tabs.len() - 1;
                    }
                }
                // Remove panes for closed tab
                // Note: daemon should send updated panes, but we clean up just in case
                self.dirty = true;
            }
            DaemonMessage::PaneCreated { tab_id, pane } => {
                log::debug!("Pane created in tab {}: {:?}", tab_id, pane);
                if let Some(ref mut win) = self.window_state {
                    if let Some(tab) = win.tabs.iter_mut().find(|t| t.id == tab_id) {
                        tab.panes.push(pane);
                    }
                }
                self.dirty = true;
            }
            DaemonMessage::PaneClosed { tab_id, pane_id } => {
                log::debug!("Pane {} closed in tab {}", pane_id, tab_id);
                if let Some(ref mut win) = self.window_state {
                    if let Some(tab) = win.tabs.iter_mut().find(|t| t.id == tab_id) {
                        tab.panes.retain(|p| p.id != pane_id);
                    }
                }
                // Also remove from pane snapshots
                self.panes.retain(|p| p.pane_id != pane_id);
                self.dirty = true;
            }
            DaemonMessage::PaneFocused { tab_id, active_pane } => {
                log::debug!("Pane focus changed in tab {}: pane {}", tab_id, active_pane);
                if let Some(ref mut win) = self.window_state {
                    if let Some(tab) = win.tabs.iter_mut().find(|t| t.id == tab_id) {
                        tab.active_pane = active_pane;
                    }
                }
                self.dirty = true;
            }
            DaemonMessage::Shutdown => {
                log::info!("Daemon shutting down");
                self.daemon_client = None;
            }
        }
    }

    /// Checks if the key event matches a keybinding and executes the action.
    /// Returns true if the key was consumed by a keybinding.
    fn check_keybinding(&mut self, event: &KeyEvent) -> bool {
        // Only process key presses, not releases or repeats
        if event.state != ElementState::Pressed || event.repeat {
            return false;
        }

        let mod_state = self.modifiers.state();
        let ctrl = mod_state.control_key();
        let alt = mod_state.alt_key();
        let shift = mod_state.shift_key();
        let super_key = mod_state.super_key();

        // Get the key name
        let key_name = match &event.logical_key {
            Key::Named(named) => {
                match named {
                    NamedKey::Tab => "tab".to_string(),
                    NamedKey::Enter => "enter".to_string(),
                    NamedKey::Escape => "escape".to_string(),
                    NamedKey::Backspace => "backspace".to_string(),
                    NamedKey::Delete => "delete".to_string(),
                    NamedKey::Insert => "insert".to_string(),
                    NamedKey::Home => "home".to_string(),
                    NamedKey::End => "end".to_string(),
                    NamedKey::PageUp => "pageup".to_string(),
                    NamedKey::PageDown => "pagedown".to_string(),
                    NamedKey::ArrowUp => "up".to_string(),
                    NamedKey::ArrowDown => "down".to_string(),
                    NamedKey::ArrowLeft => "left".to_string(),
                    NamedKey::ArrowRight => "right".to_string(),
                    NamedKey::Space => " ".to_string(),
                    NamedKey::F1 => "f1".to_string(),
                    NamedKey::F2 => "f2".to_string(),
                    NamedKey::F3 => "f3".to_string(),
                    NamedKey::F4 => "f4".to_string(),
                    NamedKey::F5 => "f5".to_string(),
                    NamedKey::F6 => "f6".to_string(),
                    NamedKey::F7 => "f7".to_string(),
                    NamedKey::F8 => "f8".to_string(),
                    NamedKey::F9 => "f9".to_string(),
                    NamedKey::F10 => "f10".to_string(),
                    NamedKey::F11 => "f11".to_string(),
                    NamedKey::F12 => "f12".to_string(),
                    _ => return false,
                }
            }
            Key::Character(c) => c.to_lowercase(),
            _ => return false,
        };

        // Look up the action
        let lookup = (ctrl, alt, shift, super_key, key_name);
        let Some(action) = self.action_map.get(&lookup).copied() else {
            return false;
        };

        // Execute the action
        self.execute_action(action);
        true
    }

    fn execute_action(&mut self, action: Action) {
        let Some(client) = &mut self.daemon_client else { return };

        match action {
            Action::NewTab => {
                log::debug!("Action: NewTab");
                let _ = client.create_tab();
            }
            Action::NextTab => {
                log::debug!("Action: NextTab");
                let _ = client.next_tab();
            }
            Action::PrevTab => {
                log::debug!("Action: PrevTab");
                let _ = client.prev_tab();
            }
            Action::Tab1 => { let _ = client.switch_tab_index(0); }
            Action::Tab2 => { let _ = client.switch_tab_index(1); }
            Action::Tab3 => { let _ = client.switch_tab_index(2); }
            Action::Tab4 => { let _ = client.switch_tab_index(3); }
            Action::Tab5 => { let _ = client.switch_tab_index(4); }
            Action::Tab6 => { let _ = client.switch_tab_index(5); }
            Action::Tab7 => { let _ = client.switch_tab_index(6); }
            Action::Tab8 => { let _ = client.switch_tab_index(7); }
            Action::Tab9 => { let _ = client.switch_tab_index(8); }
            Action::SplitHorizontal => {
                log::debug!("Action: SplitHorizontal");
                let _ = client.split_horizontal();
            }
            Action::SplitVertical => {
                log::debug!("Action: SplitVertical");
                let _ = client.split_vertical();
            }
            Action::ClosePane => {
                log::debug!("Action: ClosePane");
                let _ = client.close_pane();
            }
            Action::FocusPaneUp => {
                log::debug!("Action: FocusPaneUp");
                let _ = client.focus_pane(Direction::Up);
            }
            Action::FocusPaneDown => {
                log::debug!("Action: FocusPaneDown");
                let _ = client.focus_pane(Direction::Down);
            }
            Action::FocusPaneLeft => {
                log::debug!("Action: FocusPaneLeft");
                let _ = client.focus_pane(Direction::Left);
            }
            Action::FocusPaneRight => {
                log::debug!("Action: FocusPaneRight");
                let _ = client.focus_pane(Direction::Right);
            }
        }
    }

    fn handle_keyboard_input(&mut self, event: KeyEvent) {
        // First check if this is a keybinding
        if self.check_keybinding(&event) {
            return;
        }

        // Determine event type
        let event_type = match event.state {
            ElementState::Pressed => {
                if event.repeat {
                    KeyEventType::Repeat
                } else {
                    KeyEventType::Press
                }
            }
            ElementState::Released => KeyEventType::Release,
        };

        // In legacy mode, ignore release events
        if event_type == KeyEventType::Release && !self.keyboard_state.report_events() {
            return;
        }

        // Build modifiers from the tracked state
        let mod_state = self.modifiers.state();
        let modifiers = Modifiers {
            shift: mod_state.shift_key(),
            alt: mod_state.alt_key(),
            ctrl: mod_state.control_key(),
            super_key: mod_state.super_key(),
            hyper: false,
            meta: false,
            caps_lock: false,
            num_lock: false,
        };

        let encoder = KeyEncoder::new(&self.keyboard_state);

        let bytes: Option<Vec<u8>> = match &event.logical_key {
            Key::Named(named) => {
                let func_key = match named {
                    NamedKey::Enter => Some(FunctionalKey::Enter),
                    NamedKey::Backspace => Some(FunctionalKey::Backspace),
                    NamedKey::Tab => Some(FunctionalKey::Tab),
                    NamedKey::Escape => Some(FunctionalKey::Escape),
                    NamedKey::Space => None,
                    NamedKey::ArrowUp => Some(FunctionalKey::Up),
                    NamedKey::ArrowDown => Some(FunctionalKey::Down),
                    NamedKey::ArrowRight => Some(FunctionalKey::Right),
                    NamedKey::ArrowLeft => Some(FunctionalKey::Left),
                    NamedKey::Home => Some(FunctionalKey::Home),
                    NamedKey::End => Some(FunctionalKey::End),
                    NamedKey::PageUp => Some(FunctionalKey::PageUp),
                    NamedKey::PageDown => Some(FunctionalKey::PageDown),
                    NamedKey::Insert => Some(FunctionalKey::Insert),
                    NamedKey::Delete => Some(FunctionalKey::Delete),
                    NamedKey::F1 => Some(FunctionalKey::F1),
                    NamedKey::F2 => Some(FunctionalKey::F2),
                    NamedKey::F3 => Some(FunctionalKey::F3),
                    NamedKey::F4 => Some(FunctionalKey::F4),
                    NamedKey::F5 => Some(FunctionalKey::F5),
                    NamedKey::F6 => Some(FunctionalKey::F6),
                    NamedKey::F7 => Some(FunctionalKey::F7),
                    NamedKey::F8 => Some(FunctionalKey::F8),
                    NamedKey::F9 => Some(FunctionalKey::F9),
                    NamedKey::F10 => Some(FunctionalKey::F10),
                    NamedKey::F11 => Some(FunctionalKey::F11),
                    NamedKey::F12 => Some(FunctionalKey::F12),
                    NamedKey::CapsLock => Some(FunctionalKey::CapsLock),
                    NamedKey::ScrollLock => Some(FunctionalKey::ScrollLock),
                    NamedKey::NumLock => Some(FunctionalKey::NumLock),
                    NamedKey::PrintScreen => Some(FunctionalKey::PrintScreen),
                    NamedKey::Pause => Some(FunctionalKey::Pause),
                    NamedKey::ContextMenu => Some(FunctionalKey::Menu),
                    _ => None,
                };

                if let Some(key) = func_key {
                    Some(encoder.encode_functional(key, modifiers, event_type))
                } else if *named == NamedKey::Space {
                    Some(encoder.encode_char(' ', modifiers, event_type))
                } else {
                    None
                }
            }
            Key::Character(c) => {
                if let Some(ch) = c.chars().next() {
                    let key_char = ch.to_lowercase().next().unwrap_or(ch);
                    Some(encoder.encode_char(key_char, modifiers, event_type))
                } else {
                    None
                }
            }
            _ => None,
        };

        if let Some(bytes) = bytes {
            // Check scroll offset before borrowing client mutably
            let scroll_reset = self.get_active_pane()
                .filter(|(_, snapshot)| snapshot.scroll_offset > 0)
                .map(|(pane_id, snapshot)| (pane_id, snapshot.scroll_offset));
            
            // Now borrow client mutably
            if let Some(client) = &mut self.daemon_client {
                let _ = client.send_input(bytes);
                
                // Reset scroll position when typing (go back to live terminal)
                if let Some((active_pane_id, scroll_offset)) = scroll_reset {
                    let _ = client.send(&ClientMessage::Scroll { 
                        pane_id: active_pane_id, 
                        delta: -(scroll_offset as i32) 
                    });
                }
            }
        }
    }

    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }

        if let Some(renderer) = &mut self.renderer {
            renderer.resize(new_size.width, new_size.height);

            let (cols, rows) = renderer.terminal_size();

            if let Some(client) = &mut self.daemon_client {
                let _ = client.send_resize(cols, rows);
            }

            log::debug!("Resized to {}x{} cells", cols, rows);
            self.dirty = true;
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            self.initialize(event_loop);
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                log::info!("Window close requested");
                event_loop.exit();
            }

            WindowEvent::Resized(new_size) => {
                self.resize(new_size);
            }

            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                log::info!("Scale factor changed to {}", scale_factor);
                if let Some(renderer) = &mut self.renderer {
                    if renderer.set_scale_factor(scale_factor) {
                        let (cols, rows) = renderer.terminal_size();
                        
                        if let Some(client) = &mut self.daemon_client {
                            let _ = client.send_resize(cols, rows);
                        }
                        
                        log::info!("Terminal resized to {}x{} cells after scale change", cols, rows);
                    }
                    
                    if let Some(window) = &self.window {
                        window.request_redraw();
                    }
                }
            }

            WindowEvent::ModifiersChanged(new_modifiers) => {
                self.modifiers = new_modifiers;
            }

            WindowEvent::MouseWheel { delta, .. } => {
                // Handle mouse wheel for scrollback
                let lines = match delta {
                    MouseScrollDelta::LineDelta(_, y) => {
                        // y > 0 means scrolling up (into history), y < 0 means down
                        (y * 3.0) as i32  // 3 lines per scroll notch
                    }
                    MouseScrollDelta::PixelDelta(pos) => {
                        // Convert pixels to lines (rough approximation)
                        (pos.y / 20.0) as i32
                    }
                };
                
                if lines != 0 {
                    // Get the active pane ID to scroll
                    if let Some((active_pane_id, _)) = self.get_active_pane() {
                        if let Some(client) = &mut self.daemon_client {
                            let _ = client.send(&ClientMessage::Scroll { 
                                pane_id: active_pane_id, 
                                delta: lines 
                            });
                        }
                    }
                    if let Some(window) = &self.window {
                        window.request_redraw();
                    }
                }
            }

            WindowEvent::KeyboardInput { event, .. } => {
                self.handle_keyboard_input(event);
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }

            WindowEvent::RedrawRequested => {
                // Gather all panes for the active tab with their layout info (cloned to avoid borrow conflict)
                let (panes_with_info, active_pane_id) = self.active_tab_panes();
                let tabs = self.window_state.as_ref().map(|w| w.tabs.clone());
                let active_tab = self.window_state.as_ref().map(|w| w.active_tab).unwrap_or(0);
                
                if let Some(renderer) = &mut self.renderer {
                    if !panes_with_info.is_empty() {
                        let tabs = tabs.unwrap_or_default();
                        // Convert owned data to references for the renderer
                        let pane_refs: Vec<(&PaneSnapshot, &PaneInfo)> = panes_with_info.iter()
                            .map(|(snap, info)| (snap, info))
                            .collect();
                        match renderer.render_with_tabs(&pane_refs, active_pane_id, &tabs, active_tab) {
                            Ok(_) => {}
                            Err(wgpu::SurfaceError::Lost) => {
                                renderer.resize(renderer.width, renderer.height);
                            }
                            Err(wgpu::SurfaceError::OutOfMemory) => {
                                log::error!("Out of GPU memory!");
                                event_loop.exit();
                            }
                            Err(e) => {
                                log::error!("Render error: {:?}", e);
                            }
                        }
                        self.dirty = false;
                    }
                }
            }

            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        // Check if daemon is still connected
        if self.daemon_client.is_none() {
            log::info!("Lost connection to daemon, exiting");
            event_loop.exit();
            return;
        }

        // Poll daemon for updates
        self.poll_daemon();
        
        // Request redraw if we have new content
        if self.dirty {
            if let Some(window) = &self.window {
                window.request_redraw();
            }
        }
        
        // Use WaitUntil to wake up periodically and check for daemon messages
        // This is more compatible than relying on send_event across threads
        event_loop.set_control_flow(ControlFlow::WaitUntil(
            std::time::Instant::now() + Duration::from_millis(16)
        ));
    }
    
    fn user_event(&mut self, _event_loop: &ActiveEventLoop, _event: ()) {
        // Daemon poll thread woke us up - poll for messages
        self.poll_daemon();
        
        // Request redraw if we have new content
        if self.dirty {
            if let Some(window) = &self.window {
                window.request_redraw();
            }
        }
    }
}

impl Drop for App {
    fn drop(&mut self) {
        // Signal the daemon poll thread to exit
        self.shutdown.store(true, Ordering::Relaxed);
    }
}

fn main() {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    log::info!("Starting ZTerm client");

    // Create event loop with Wayland preference
    let event_loop = EventLoop::builder()
        .with_any_thread(true)
        .build()
        .expect("Failed to create event loop");

    // Use Wait instead of Poll to avoid busy-looping
    // The daemon poll thread will wake us when data is available
    event_loop.set_control_flow(ControlFlow::Wait);

    let mut app = App::new();
    
    // Give the app a proxy to wake the event loop from the daemon poll thread
    app.set_event_loop_proxy(event_loop.create_proxy());
    
    event_loop.run_app(&mut app).expect("Event loop error");
}
