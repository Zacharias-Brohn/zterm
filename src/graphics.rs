//! Kitty Graphics Protocol implementation.
//!
//! This module handles parsing and processing of Kitty graphics protocol commands
//! which allow terminals to display images inline.
//!
//! Protocol reference: https://sw.kovidgoyal.net/kitty/graphics-protocol/

use std::collections::HashMap;
use std::io::{Cursor, Read};
use std::time::Instant;

use flate2::read::ZlibDecoder;
use image::{codecs::gif::GifDecoder, AnimationDecoder, ImageFormat};

/// Action to perform with the graphics command.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum Action {
    /// Transmit image data (store but don't display).
    #[default]
    Transmit,
    /// Transmit and display image.
    TransmitAndDisplay,
    /// Display a previously transmitted image.
    Put,
    /// Delete images.
    Delete,
    /// Transmit animation frame.
    AnimationFrame,
    /// Control animation.
    AnimationControl,
    /// Query terminal for support.
    Query,
}

/// Image data format.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum Format {
    /// 24-bit RGB (3 bytes per pixel).
    Rgb,
    /// 32-bit RGBA (4 bytes per pixel).
    #[default]
    Rgba,
    /// PNG encoded data.
    Png,
    /// GIF encoded data (for animations).
    Gif,
}

/// How image data is transmitted.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum Transmission {
    /// Direct data in the escape sequence.
    #[default]
    Direct,
    /// Read from a file path.
    File,
    /// Read from a temporary file (deleted after read).
    TempFile,
    /// Read from shared memory.
    SharedMemory,
}

/// Compression method for the payload.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Compression {
    /// Zlib compression.
    Zlib,
}

/// Delete target specification.
#[derive(Clone, Debug, PartialEq, Default)]
pub enum DeleteTarget {
    /// Delete all images.
    #[default]
    All,
    /// Delete by image ID.
    ById(u32),
    /// Delete by image number (for virtual placements).
    ByNumber(u32),
    /// Delete at cursor position.
    AtCursor,
    /// Delete animation frames.
    AnimationFrames(u32),
    /// Delete by cell range.
    CellRange {
        x: Option<u32>,
        y: Option<u32>,
        z: Option<i32>,
    },
}

/// Parsed Kitty graphics command.
#[derive(Clone, Debug, Default)]
pub struct GraphicsCommand {
    /// Action to perform.
    pub action: Action,
    /// Image data format.
    pub format: Format,
    /// Transmission medium.
    pub transmission: Transmission,
    /// Image ID (for referencing stored images).
    pub image_id: Option<u32>,
    /// Placement ID (for multiple placements of same image).
    pub placement_id: Option<u32>,
    /// Image width in pixels (required for raw data).
    pub width: Option<u32>,
    /// Image height in pixels (required for raw data).
    pub height: Option<u32>,
    /// Source rectangle X offset.
    pub src_x: u32,
    /// Source rectangle Y offset.
    pub src_y: u32,
    /// Source rectangle width (0 = full width).
    pub src_width: u32,
    /// Source rectangle height (0 = full height).
    pub src_height: u32,
    /// Display columns (0 = auto).
    pub cols: u32,
    /// Display rows (0 = auto).
    pub rows: u32,
    /// X offset within cell (pixels).
    pub x_offset: u32,
    /// Y offset within cell (pixels).
    pub y_offset: u32,
    /// Z-index for layering.
    pub z_index: i32,
    /// Compression method.
    pub compression: Option<Compression>,
    /// More chunks coming (chunked transfer).
    pub more_chunks: bool,
    /// Quiet mode (0=normal, 1=suppress OK, 2=suppress all).
    pub quiet: u8,
    /// Cursor movement after display (0=move, 1=don't move).
    pub cursor_movement: u8,
    /// Delete target (for delete action).
    pub delete_target: DeleteTarget,
    /// Unicode placeholder (virtual placement).
    pub unicode_placeholder: bool,
    /// Parent image ID (for animation frames).
    pub parent_id: Option<u32>,
    /// Parent placement ID (for animation frames).
    pub parent_placement_id: Option<u32>,
    /// Frame number (for animation).
    pub frame_number: Option<u32>,
    /// Frame gap in milliseconds (z key for animation frames).
    pub frame_gap: Option<i32>,
    /// Base/source frame number for compositing (c key for animation, 1-indexed).
    pub base_frame: Option<u32>,
    /// Frame number being edited (r key for animation, 1-indexed).
    pub edit_frame: Option<u32>,
    /// Animation state (s key for animation control: 1=stop, 2=loading, 3=run).
    pub animation_state: Option<u8>,
    /// Loop count (v key for animation: 0=infinite, n=n loops).
    pub loop_count: Option<u32>,
    /// Composition mode (X key for animation: 0=alpha blend, 1=replace).
    pub composition_mode: u8,
    /// Background color for animation frames (Y key, 32-bit RGBA).
    pub background_color: Option<u32>,
    /// Raw payload data (base64 decoded).
    pub payload: Vec<u8>,
}

impl GraphicsCommand {
    /// Parse a graphics command from APC data.
    ///
    /// The data format is: `G<key>=<value>,<key>=<value>,...;<base64_payload>`
    pub fn parse(data: &[u8]) -> Option<Self> {
        // Must start with 'G'
        if data.first() != Some(&b'G') {
            return None;
        }

        let data = &data[1..]; // Skip 'G'

        // Find the semicolon separating control data from payload
        let (control_part, payload_part) =
            match data.iter().position(|&b| b == b';') {
                Some(pos) => (&data[..pos], &data[pos + 1..]),
                None => (data, &[][..]),
            };

        let mut cmd = GraphicsCommand::default();

        // Parse control key=value pairs
        // First pass: get action to know how to interpret overloaded keys
        let control_str = std::str::from_utf8(control_part).ok()?;
        log::debug!("Graphics control string: {}", control_str);

        // Collect key-value pairs
        let mut pairs: Vec<(&str, &str)> = Vec::new();
        for pair in control_str.split(',') {
            if pair.is_empty() {
                continue;
            }
            let mut parts = pair.splitn(2, '=');
            if let Some(key) = parts.next() {
                let value = parts.next().unwrap_or("");
                pairs.push((key, value));
                // Get action early so we know how to interpret other keys
                if key == "a" {
                    cmd.action = match value {
                        "t" => Action::Transmit,
                        "T" => Action::TransmitAndDisplay,
                        "p" => Action::Put,
                        "d" => Action::Delete,
                        "f" => Action::AnimationFrame,
                        "a" => Action::AnimationControl,
                        "q" => Action::Query,
                        _ => Action::Transmit,
                    };
                }
            }
        }

        let is_animation =
            matches!(cmd.action, Action::AnimationFrame | Action::AnimationControl);

        // Second pass: parse all keys with correct interpretation
        for (key, value) in pairs {
            match key {
                "a" => {} // Already parsed above
                "f" => {
                    cmd.format = match value {
                        "24" => Format::Rgb,
                        "32" => Format::Rgba,
                        "100" => Format::Png,
                        _ => Format::Rgba,
                    };
                }
                "t" => {
                    cmd.transmission = match value {
                        "d" => Transmission::Direct,
                        "f" => Transmission::File,
                        "t" => Transmission::TempFile,
                        "s" => Transmission::SharedMemory,
                        _ => Transmission::Direct,
                    };
                }
                "i" => cmd.image_id = value.parse().ok(),
                "I" => cmd.image_id = value.parse().ok(), // Alternate form
                "p" => cmd.placement_id = value.parse().ok(),
                "s" => {
                    // s = width for images, animation_state for animation control
                    if matches!(cmd.action, Action::AnimationControl) {
                        cmd.animation_state = value.parse().ok();
                    } else {
                        cmd.width = value.parse().ok();
                    }
                }
                "v" => {
                    // v = height for images, loop_count for animation control
                    if matches!(cmd.action, Action::AnimationControl) {
                        cmd.loop_count = value.parse().ok();
                    } else {
                        cmd.height = value.parse().ok();
                    }
                }
                "x" => cmd.src_x = value.parse().unwrap_or(0),
                "y" => cmd.src_y = value.parse().unwrap_or(0),
                "w" => cmd.src_width = value.parse().unwrap_or(0),
                "h" => cmd.src_height = value.parse().unwrap_or(0),
                "c" => {
                    // c = cols for images, base_frame for animation
                    if is_animation {
                        cmd.base_frame = value.parse().ok();
                    } else {
                        cmd.cols = value.parse().unwrap_or(0);
                    }
                }
                "r" => {
                    // r = rows for images, edit_frame for animation
                    if is_animation {
                        cmd.edit_frame = value.parse().ok();
                    } else {
                        cmd.rows = value.parse().unwrap_or(0);
                    }
                }
                "X" => {
                    // X = x_offset for images, composition_mode for animation
                    if is_animation {
                        cmd.composition_mode = value.parse().unwrap_or(0);
                    } else {
                        cmd.x_offset = value.parse().unwrap_or(0);
                        log::debug!(
                            "Parsed X={} as x_offset={}",
                            value,
                            cmd.x_offset
                        );
                    }
                }
                "Y" => {
                    // Y = y_offset for images, background_color for animation
                    if is_animation {
                        cmd.background_color = value.parse().ok();
                    } else {
                        cmd.y_offset = value.parse().unwrap_or(0);
                    }
                }
                "z" => {
                    // z = z_index for images, frame_gap for animation frames
                    if matches!(cmd.action, Action::AnimationFrame) {
                        cmd.frame_gap = value.parse().ok();
                    } else {
                        cmd.z_index = value.parse().unwrap_or(0);
                    }
                }
                "o" => {
                    if value == "z" {
                        cmd.compression = Some(Compression::Zlib);
                    }
                }
                "m" => cmd.more_chunks = value == "1",
                "q" => cmd.quiet = value.parse().unwrap_or(0),
                "C" => cmd.cursor_movement = value.parse().unwrap_or(0),
                "U" => cmd.unicode_placeholder = value == "1",
                "d" => {
                    // Delete target
                    cmd.delete_target = match value {
                        "a" | "A" => DeleteTarget::All,
                        "i" | "I" => DeleteTarget::ById(0), // ID set separately
                        "n" | "N" => DeleteTarget::ByNumber(0),
                        "c" | "C" => DeleteTarget::AtCursor,
                        "f" | "F" => DeleteTarget::AnimationFrames(0),
                        "p" | "P" | "q" | "Q" | "x" | "X" | "y" | "Y" | "z"
                        | "Z" => DeleteTarget::CellRange {
                            x: None,
                            y: None,
                            z: None,
                        },
                        _ => DeleteTarget::All,
                    };
                }
                _ => {} // Ignore unknown keys
            }
        }

        // Decode base64 payload
        if !payload_part.is_empty() {
            if let Ok(payload_str) = std::str::from_utf8(payload_part) {
                if let Ok(decoded) = base64_decode(payload_str) {
                    cmd.payload = decoded;
                }
            }
        }

        Some(cmd)
    }

    /// Decompress payload if compressed.
    pub fn decompress_payload(&mut self) -> Result<(), GraphicsError> {
        if let Some(Compression::Zlib) = self.compression {
            let mut decoder = ZlibDecoder::new(&self.payload[..]);
            let mut decompressed = Vec::new();
            decoder
                .read_to_end(&mut decompressed)
                .map_err(|_| GraphicsError::DecompressionFailed)?;
            self.payload = decompressed;
            self.compression = None;
        }
        Ok(())
    }

    /// Decode PNG payload to RGBA pixels.
    pub fn decode_png(&self) -> Result<(u32, u32, Vec<u8>), GraphicsError> {
        let img = image::load_from_memory_with_format(
            &self.payload,
            ImageFormat::Png,
        )
        .map_err(|_| GraphicsError::PngDecodeFailed)?;
        let rgba = img.to_rgba8();
        let (width, height) = rgba.dimensions();
        Ok((width, height, rgba.into_raw()))
    }

    /// Convert RGB payload to RGBA.
    pub fn rgb_to_rgba(&self) -> Vec<u8> {
        let mut rgba = Vec::with_capacity(self.payload.len() * 4 / 3);
        for chunk in self.payload.chunks(3) {
            if chunk.len() == 3 {
                rgba.push(chunk[0]);
                rgba.push(chunk[1]);
                rgba.push(chunk[2]);
                rgba.push(255);
            }
        }
        rgba
    }
}

/// Decode a GIF image, returning dimensions and animation data.
/// Returns (width, height, first_frame_data, animation_data).
pub fn decode_gif(
    data: &[u8],
) -> Result<(u32, u32, Vec<u8>, Option<AnimationData>), GraphicsError> {
    let cursor = Cursor::new(data);
    let decoder = GifDecoder::new(cursor).map_err(|e| {
        log::error!("GIF decode error: {}", e);
        GraphicsError::GifDecodeFailed
    })?;

    let frames_iter = decoder.into_frames();
    let mut frames = Vec::new();
    let mut width = 0u32;
    let mut height = 0u32;
    let mut total_duration_ms = 0u64;

    for frame_result in frames_iter {
        let frame = frame_result.map_err(|e| {
            log::error!("GIF frame decode error: {}", e);
            GraphicsError::GifDecodeFailed
        })?;

        let buffer = frame.buffer();
        let (w, h) = buffer.dimensions();
        width = w;
        height = h;

        // Get frame delay (in milliseconds)
        let delay = frame.delay();
        let (numer, denom) = delay.numer_denom_ms();
        let duration_ms = if denom > 0 { numer / denom } else { 100 };
        // GIF standard: delay of 0 means use default (100ms)
        let duration_ms = if duration_ms == 0 { 100 } else { duration_ms };

        total_duration_ms += duration_ms as u64;

        frames.push(AnimationFrame {
            data: buffer.as_raw().clone(),
            duration_ms,
        });
    }

    if frames.is_empty() {
        return Err(GraphicsError::GifDecodeFailed);
    }

    log::debug!("Decoded GIF: {}x{}, {} frames, {}ms total duration", 
        width, height, frames.len(), total_duration_ms);

    let first_frame = frames[0].data.clone();

    // If only one frame, treat as static image
    let animation = if frames.len() > 1 {
        Some(AnimationData {
            frames,
            current_frame: 0,
            frame_start: None,
            looping: true,
            total_duration_ms,
            state: AnimationState::Running,
            loops_remaining: None,
        })
    } else {
        None
    };

    Ok((width, height, first_frame, animation))
}

/// Decode a WebM video file, returning dimensions and animation data.
/// Only decodes video stream, audio is ignored.
#[cfg(feature = "webm")]
pub fn decode_webm(
    path: &str,
) -> Result<(u32, u32, Vec<u8>, Option<AnimationData>), GraphicsError> {
    use ffmpeg::format::{input, Pixel};
    use ffmpeg::media::Type;
    use ffmpeg::software::scaling::{
        context::Context as ScalingContext, flag::Flags,
    };
    use ffmpeg::util::frame::video::Video;
    use ffmpeg_next as ffmpeg;

    // Initialize FFmpeg (safe to call multiple times)
    ffmpeg::init().map_err(|e| {
        log::error!("FFmpeg init error: {}", e);
        GraphicsError::VideoDecodeFailed
    })?;

    // Open the file
    let mut input_ctx = input(&path).map_err(|e| {
        log::error!("Failed to open video file {}: {}", path, e);
        GraphicsError::FileReadFailed
    })?;

    // Find the video stream
    let video_stream =
        input_ctx.streams().best(Type::Video).ok_or_else(|| {
            log::error!("No video stream found in {}", path);
            GraphicsError::VideoDecodeFailed
        })?;

    let video_stream_index = video_stream.index();
    let time_base = video_stream.time_base();

    // Get decoder for this stream
    let context_decoder = ffmpeg::codec::context::Context::from_parameters(
        video_stream.parameters(),
    )
    .map_err(|e| {
        log::error!("Failed to create decoder context: {}", e);
        GraphicsError::VideoDecodeFailed
    })?;

    let mut decoder = context_decoder.decoder().video().map_err(|e| {
        log::error!("Failed to get video decoder: {}", e);
        GraphicsError::VideoDecodeFailed
    })?;

    let width = decoder.width();
    let height = decoder.height();

    // Create scaler to convert to RGBA
    let mut scaler = ScalingContext::get(
        decoder.format(),
        width,
        height,
        Pixel::RGBA,
        width,
        height,
        Flags::BILINEAR,
    )
    .map_err(|e| {
        log::error!("Failed to create scaler: {}", e);
        GraphicsError::VideoDecodeFailed
    })?;

    let mut frames = Vec::new();
    let mut total_duration_ms = 0u64;
    let mut last_pts: Option<i64> = None;

    // Process packets
    for (stream, packet) in input_ctx.packets() {
        if stream.index() != video_stream_index {
            continue;
        }

        decoder.send_packet(&packet).ok();

        let mut decoded = Video::empty();
        while decoder.receive_frame(&mut decoded).is_ok() {
            // Scale to RGBA
            let mut rgba_frame = Video::empty();
            if scaler.run(&decoded, &mut rgba_frame).is_err() {
                continue;
            }

            // Get RGBA pixel data
            let data = rgba_frame.data(0);
            let stride = rgba_frame.stride(0);

            // Copy data, handling stride if needed
            let mut rgba_data =
                Vec::with_capacity((width * height * 4) as usize);
            if stride == (width * 4) as usize {
                rgba_data
                    .extend_from_slice(&data[..(width * height * 4) as usize]);
            } else {
                // Handle stride padding
                for row in 0..height as usize {
                    let start = row * stride;
                    let end = start + (width * 4) as usize;
                    rgba_data.extend_from_slice(&data[start..end]);
                }
            }

            // Calculate frame duration from PTS
            let pts = decoded.pts().unwrap_or(0);
            let duration_ms = if let Some(last) = last_pts {
                let pts_diff = pts - last;
                ((pts_diff as f64) * f64::from(time_base) * 1000.0) as u32
            } else {
                // First frame - estimate from frame rate or use default
                33 // ~30fps default
            };
            last_pts = Some(pts);

            total_duration_ms += duration_ms as u64;

            frames.push(AnimationFrame {
                data: rgba_data,
                duration_ms: duration_ms.max(1), // Ensure at least 1ms
            });
        }
    }

    // Flush decoder
    decoder.send_eof().ok();
    let mut decoded = Video::empty();
    while decoder.receive_frame(&mut decoded).is_ok() {
        let mut rgba_frame = Video::empty();
        if scaler.run(&decoded, &mut rgba_frame).is_ok() {
            let data = rgba_frame.data(0);
            let stride = rgba_frame.stride(0);
            let mut rgba_data =
                Vec::with_capacity((width * height * 4) as usize);
            if stride == (width * 4) as usize {
                rgba_data
                    .extend_from_slice(&data[..(width * height * 4) as usize]);
            } else {
                for row in 0..height as usize {
                    let start = row * stride;
                    let end = start + (width * 4) as usize;
                    rgba_data.extend_from_slice(&data[start..end]);
                }
            }
            frames.push(AnimationFrame {
                data: rgba_data,
                duration_ms: 33,
            });
            total_duration_ms += 33;
        }
    }

    if frames.is_empty() {
        return Err(GraphicsError::VideoDecodeFailed);
    }

    let first_frame = frames[0].data.clone();

    // Videos always have animation data (even if just one frame)
    let animation = if frames.len() > 1 {
        Some(AnimationData {
            frames,
            current_frame: 0,
            frame_start: None,
            looping: true,
            total_duration_ms,
            state: AnimationState::Running,
            loops_remaining: None,
        })
    } else {
        None
    };

    log::debug!(
        "Decoded WebM: {}x{}, {} frames, {}ms total",
        width,
        height,
        animation.as_ref().map(|a| a.frames.len()).unwrap_or(1),
        total_duration_ms
    );

    Ok((width, height, first_frame, animation))
}

/// Errors that can occur during graphics processing.
#[derive(Clone, Debug, PartialEq)]
pub enum GraphicsError {
    /// Base64 decoding failed.
    Base64DecodeFailed,
    /// Zlib decompression failed.
    DecompressionFailed,
    /// PNG decoding failed.
    PngDecodeFailed,
    /// GIF decoding failed.
    GifDecodeFailed,
    /// WebM/video decoding failed.
    VideoDecodeFailed,
    /// Missing required dimensions.
    MissingDimensions,
    /// Invalid image data.
    InvalidData,
    /// Image not found.
    ImageNotFound,
    /// Missing image ID.
    MissingId,
    /// File read failed.
    FileReadFailed,
    /// Unsupported transmission or format.
    UnsupportedFormat,
}

/// Stored image data.
#[derive(Clone, Debug)]
pub struct ImageData {
    /// Unique image ID.
    pub id: u32,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// RGBA pixel data (current frame for animated images).
    pub data: Vec<u8>,
    /// Animation data if this is an animated image.
    pub animation: Option<AnimationData>,
}

/// Animation state for playback control.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub enum AnimationState {
    /// Animation is stopped (s=1).
    Stopped,
    /// Animation is loading frames (s=2).
    Loading,
    /// Animation is running normally (s=3).
    #[default]
    Running,
}

/// Animation data for GIF/WebM images.
#[derive(Clone, Debug)]
pub struct AnimationData {
    /// All frames of the animation.
    pub frames: Vec<AnimationFrame>,
    /// Current frame index.
    pub current_frame: usize,
    /// When the current frame started displaying.
    pub frame_start: Option<Instant>,
    /// Whether the animation should loop.
    pub looping: bool,
    /// Total duration of one loop in milliseconds.
    pub total_duration_ms: u64,
    /// Playback state.
    pub state: AnimationState,
    /// Number of loops remaining (None = infinite).
    pub loops_remaining: Option<u32>,
}

/// A single frame in an animation.
#[derive(Clone, Debug)]
pub struct AnimationFrame {
    /// RGBA pixel data for this frame.
    pub data: Vec<u8>,
    /// Duration to display this frame in milliseconds.
    pub duration_ms: u32,
}

/// Result of placing an image, used for cursor movement.
#[derive(Clone, Debug, Default)]
pub struct PlacementResult {
    /// Number of columns the image spans.
    pub cols: usize,
    /// Number of rows the image spans.
    pub rows: usize,
    /// Whether cursor movement should be suppressed (C=1).
    pub suppress_cursor_move: bool,
    /// Whether this is a virtual/Unicode placeholder placement (U=1).
    pub virtual_placement: bool,
}

/// A placement of an image on the terminal grid.
#[derive(Clone, Debug)]
pub struct ImagePlacement {
    /// Image ID this placement refers to.
    pub image_id: u32,
    /// Unique placement ID.
    pub placement_id: u32,
    /// Column position.
    pub col: usize,
    /// Row position (in scrollback-aware coordinates).
    pub row: usize,
    /// Display width in columns.
    pub cols: usize,
    /// Display height in rows.
    pub rows: usize,
    /// Z-index for layering.
    pub z_index: i32,
    /// Source rectangle X offset.
    pub src_x: u32,
    /// Source rectangle Y offset.
    pub src_y: u32,
    /// Source rectangle width.
    pub src_width: u32,
    /// Source rectangle height.
    pub src_height: u32,
    /// X offset within the first cell.
    pub x_offset: u32,
    /// Y offset within the first cell.
    pub y_offset: u32,
}

/// Storage for images and their placements.
#[derive(Default)]
pub struct ImageStorage {
    /// Stored images by ID.
    images: HashMap<u32, ImageData>,
    /// Active placements.
    placements: Vec<ImagePlacement>,
    /// Buffer for chunked transmissions (image_id -> accumulated data).
    chunk_buffer: HashMap<u32, ChunkBuffer>,
    /// Next auto-generated image ID.
    next_id: u32,
    /// Flag indicating images have changed and need re-upload to GPU.
    pub dirty: bool,
}

/// Buffer for accumulating chunked image data.
#[derive(Default)]
struct ChunkBuffer {
    command: Option<GraphicsCommand>,
    data: Vec<u8>,
}

impl ImageStorage {
    /// Create a new empty image storage.
    pub fn new() -> Self {
        Self {
            images: HashMap::new(),
            placements: Vec::new(),
            chunk_buffer: HashMap::new(),
            next_id: 1,
            dirty: false,
        }
    }

    /// Process a graphics command and return an optional response and placement result.
    /// The placement result contains dimensions for cursor movement after image display.
    pub fn process_command(
        &mut self,
        mut cmd: GraphicsCommand,
        cursor_col: usize,
        cursor_row: usize,
        cell_width: f32,
        cell_height: f32,
    ) -> (Option<String>, Option<PlacementResult>) {
        // Handle chunked transfer
        if cmd.more_chunks {
            let id = cmd.image_id.unwrap_or(0);
            let buffer = self.chunk_buffer.entry(id).or_default();
            buffer.data.extend_from_slice(&cmd.payload);
            if buffer.command.is_none() {
                buffer.command = Some(cmd);
            }
            return (None, None);
        }

        // Check if this completes a chunked transfer
        let id = cmd.image_id.unwrap_or(0);
        if let Some(mut buffer) = self.chunk_buffer.remove(&id) {
            buffer.data.extend_from_slice(&cmd.payload);
            if let Some(mut buffered_cmd) = buffer.command {
                buffered_cmd.payload = buffer.data;
                cmd = buffered_cmd;
            }
        }

        match cmd.action {
            Action::Query => (self.handle_query(&cmd), None),
            Action::Transmit => (self.handle_transmit(cmd), None),
            Action::TransmitAndDisplay => self.handle_transmit_and_display(
                cmd,
                cursor_col,
                cursor_row,
                cell_width,
                cell_height,
            ),
            Action::Put => self.handle_put(
                &cmd,
                cursor_col,
                cursor_row,
                cell_width,
                cell_height,
            ),
            Action::Delete => {
                self.handle_delete(&cmd);
                (None, None)
            }
            Action::AnimationFrame => {
                let response = self.handle_animation_frame(cmd);
                (response, None)
            }
            Action::AnimationControl => {
                let response = self.handle_animation_control(&cmd);
                (response, None)
            }
        }
    }

    /// Handle a query command.
    fn handle_query(&self, cmd: &GraphicsCommand) -> Option<String> {
        let id = cmd.image_id.unwrap_or(0);
        // Respond with OK to indicate we support the protocol
        Some(format!("\x1b_Gi={};OK\x1b\\", id))
    }

    /// Handle a transmit command (store image without displaying).
    fn handle_transmit(&mut self, mut cmd: GraphicsCommand) -> Option<String> {
        let result = self.store_image(&mut cmd);
        self.format_response(&cmd, result)
    }

    /// Handle transmit and display.
    fn handle_transmit_and_display(
        &mut self,
        mut cmd: GraphicsCommand,
        cursor_col: usize,
        cursor_row: usize,
        cell_width: f32,
        cell_height: f32,
    ) -> (Option<String>, Option<PlacementResult>) {
        log::debug!(
            "handle_transmit_and_display: transmission={:?}, payload_len={}",
            cmd.transmission,
            cmd.payload.len()
        );

        let suppress_cursor = cmd.cursor_movement == 1;
        let virtual_placement = cmd.unicode_placeholder;

        let result = self.store_image(&mut cmd);
        log::debug!("store_image result: {:?}", result);

        let placement_result = if let Ok(id) = result {
            // Update cmd.image_id with the assigned ID (needed if it was None)
            cmd.image_id = Some(id);
            let (cols, rows) = self.place_image(
                &cmd,
                cursor_col,
                cursor_row,
                cell_width,
                cell_height,
            );
            log::debug!("Placed image id={} at col={} row={}, cols={} rows={}, placements={}", 
                id, cursor_col, cursor_row, cols, rows, self.placements.len());
            Some(PlacementResult {
                cols,
                rows,
                suppress_cursor_move: suppress_cursor,
                virtual_placement,
            })
        } else {
            None
        };
        (self.format_response(&cmd, result), placement_result)
    }

    /// Handle a put command (display previously stored image).
    fn handle_put(
        &mut self,
        cmd: &GraphicsCommand,
        cursor_col: usize,
        cursor_row: usize,
        cell_width: f32,
        cell_height: f32,
    ) -> (Option<String>, Option<PlacementResult>) {
        let id = match cmd.image_id {
            Some(id) => id,
            None => return (None, None),
        };

        let suppress_cursor = cmd.cursor_movement == 1;
        let virtual_placement = cmd.unicode_placeholder;

        if self.images.contains_key(&id) {
            let (cols, rows) = self.place_image(
                cmd,
                cursor_col,
                cursor_row,
                cell_width,
                cell_height,
            );
            let placement_result = PlacementResult {
                cols,
                rows,
                suppress_cursor_move: suppress_cursor,
                virtual_placement,
            };
            (self.format_response(cmd, Ok(id)), Some(placement_result))
        } else {
            (
                self.format_response(cmd, Err(GraphicsError::ImageNotFound)),
                None,
            )
        }
    }

    /// Handle a delete command.
    fn handle_delete(&mut self, cmd: &GraphicsCommand) {
        match &cmd.delete_target {
            DeleteTarget::All => {
                self.images.clear();
                self.placements.clear();
                self.dirty = true;
            }
            DeleteTarget::ById(id) => {
                let id = cmd.image_id.unwrap_or(*id);
                self.images.remove(&id);
                self.placements.retain(|p| p.image_id != id);
                self.dirty = true;
            }
            DeleteTarget::AtCursor => {
                // Would need cursor position - simplified for now
                self.placements.clear();
                self.dirty = true;
            }
            _ => {
                // Other delete modes not yet implemented
            }
        }
    }

    /// Handle an animation frame command (a=f).
    /// This adds a frame to an existing image's animation.
    fn handle_animation_frame(&mut self, mut cmd: GraphicsCommand) -> Option<String> {
        let id = match cmd.image_id {
            Some(id) => id,
            None => {
                log::warn!("AnimationFrame without image_id");
                return self.format_response(&cmd, Err(GraphicsError::MissingId));
            }
        };

        log::debug!(
            "AnimationFrame: id={}, base_frame={:?}, edit_frame={:?}, frame_gap={:?}, size={}x{:?}, transmission={:?}, payload_len={}",
            id,
            cmd.base_frame,
            cmd.edit_frame,
            cmd.frame_gap,
            cmd.width.unwrap_or(0),
            cmd.height,
            cmd.transmission,
            cmd.payload.len()
        );

        // Handle file-based transmission (load actual data from file/shm)
        match cmd.transmission {
            Transmission::File | Transmission::TempFile => {
                let path = match std::str::from_utf8(&cmd.payload) {
                    Ok(p) => p.trim().to_string(),
                    Err(_) => {
                        log::warn!("Invalid file path in animation frame");
                        return self.format_response(&cmd, Err(GraphicsError::FileReadFailed));
                    }
                };
                log::debug!("Reading animation frame from file: {}", path);
                match std::fs::read(&path) {
                    Ok(data) => cmd.payload = data,
                    Err(e) => {
                        log::warn!("Failed to read animation frame file {}: {}", path, e);
                        return self.format_response(&cmd, Err(GraphicsError::FileReadFailed));
                    }
                }
                // Delete temp file after reading
                if cmd.transmission == Transmission::TempFile {
                    let _ = std::fs::remove_file(&path);
                }
            }
            Transmission::SharedMemory => {
                let shm_name = match std::str::from_utf8(&cmd.payload) {
                    Ok(p) => p.trim().to_string(),
                    Err(_) => {
                        log::warn!("Invalid shared memory name in animation frame");
                        return self.format_response(&cmd, Err(GraphicsError::FileReadFailed));
                    }
                };
                let shm_path = format!("/dev/shm/{}", shm_name);
                log::debug!("Reading animation frame from shared memory: {}", shm_path);
                match std::fs::read(&shm_path) {
                    Ok(data) => {
                        log::debug!("Read {} bytes from shared memory", data.len());
                        cmd.payload = data;
                    }
                    Err(e) => {
                        log::warn!("Failed to read animation frame shm {}: {}", shm_path, e);
                        return self.format_response(&cmd, Err(GraphicsError::FileReadFailed));
                    }
                }
                // Remove shared memory object after reading
                let _ = std::fs::remove_file(&shm_path);
            }
            Transmission::Direct => {
                // Payload already contains the data
            }
        }

        // Decompress payload if needed
        if let Err(e) = cmd.decompress_payload() {
            log::warn!("Failed to decompress animation frame payload: {:?}", e);
            return self.format_response(&cmd, Err(e));
        }

        // Get or decode the frame data
        let frame_data = match cmd.format {
            Format::Png => match cmd.decode_png() {
                Ok((w, h, data)) => {
                    // Update dimensions from decoded PNG
                    cmd.width = Some(w);
                    cmd.height = Some(h);
                    data
                }
                Err(e) => {
                    log::warn!("Failed to decode animation frame PNG: {:?}", e);
                    return self.format_response(&cmd, Err(e));
                }
            },
            Format::Rgba => cmd.payload.clone(),
            Format::Rgb => cmd.rgb_to_rgba(),
            Format::Gif => {
                // Unlikely, but handle it
                log::warn!("GIF format in animation frame - not supported");
                return self.format_response(&cmd, Err(GraphicsError::UnsupportedFormat));
            }
        };

        // Get the image and add the frame
        let image = match self.images.get_mut(&id) {
            Some(img) => img,
            None => {
                log::warn!("AnimationFrame for non-existent image {}", id);
                return self.format_response(&cmd, Err(GraphicsError::ImageNotFound));
            }
        };

        // Expected size for a full frame
        let expected_size = (image.width * image.height * 4) as usize;
        
        // Initialize animation if this image doesn't have one yet
        // This MUST happen before compositing so that frame 0 exists for c=1
        if image.animation.is_none() {
            // Debug: check base image alpha values
            let transparent_count = image.data.chunks(4).filter(|p| p.len() == 4 && p[3] < 255).count();
            let total_pixels = image.data.len() / 4;
            log::debug!(
                "Creating animation base frame: {}/{} pixels have alpha < 255, data len = {}",
                transparent_count,
                total_pixels,
                image.data.len()
            );
            
            let base_frame = AnimationFrame {
                data: image.data.clone(),
                duration_ms: 100, // Default for base frame
            };
            image.animation = Some(AnimationData {
                frames: vec![base_frame],
                current_frame: 0,
                frame_start: None,
                looping: true,
                total_duration_ms: 100,
                state: AnimationState::Loading,
                loops_remaining: None,
            });
        }
        
        // Composite the frame onto the base frame if needed
        // GIF animations typically use delta frames where transparent pixels
        // should show through to the previous frame
        let final_frame_data = if let Some(base_frame_num) = cmd.base_frame {
            // base_frame is 1-indexed (1 = root frame, 2 = second frame, etc.)
            // We need to get the base frame data and composite the new data on top
            let anim = image.animation.as_ref().unwrap(); // Safe: we just created it above
            let base_idx = if base_frame_num == 0 {
                0
            } else {
                (base_frame_num as usize).saturating_sub(1)
            };
            
            if base_idx < anim.frames.len() {
                let base_data = &anim.frames[base_idx].data;
                
                if frame_data.len() == expected_size && base_data.len() == expected_size {
                    // Both frames are full size - composite them
                    // composition_mode: 0 = alpha blend, 1 = overwrite
                    let mut composited = base_data.clone();
                    
                    for i in (0..expected_size).step_by(4) {
                        let src_a = frame_data[i + 3];
                        
                        if src_a == 255 {
                            // Fully opaque source - just copy
                            composited[i] = frame_data[i];
                            composited[i + 1] = frame_data[i + 1];
                            composited[i + 2] = frame_data[i + 2];
                            composited[i + 3] = 255;
                        } else if src_a > 0 {
                            if cmd.composition_mode == 1 {
                                // Overwrite mode - replace pixel
                                composited[i] = frame_data[i];
                                composited[i + 1] = frame_data[i + 1];
                                composited[i + 2] = frame_data[i + 2];
                                composited[i + 3] = frame_data[i + 3];
                            } else {
                                // Alpha blend mode (default)
                                let src_r = frame_data[i] as u32;
                                let src_g = frame_data[i + 1] as u32;
                                let src_b = frame_data[i + 2] as u32;
                                let src_a32 = src_a as u32;
                                
                                let dst_r = composited[i] as u32;
                                let dst_g = composited[i + 1] as u32;
                                let dst_b = composited[i + 2] as u32;
                                let dst_a = composited[i + 3] as u32;
                                
                                // Standard alpha compositing: out = src + dst * (1 - src_a)
                                let inv_a = 255 - src_a32;
                                composited[i] = ((src_r * src_a32 + dst_r * inv_a) / 255) as u8;
                                composited[i + 1] = ((src_g * src_a32 + dst_g * inv_a) / 255) as u8;
                                composited[i + 2] = ((src_b * src_a32 + dst_b * inv_a) / 255) as u8;
                                composited[i + 3] = (src_a32 + dst_a * inv_a / 255).min(255) as u8;
                            }
                        }
                        // else: src_a == 0, keep base pixel (already in composited)
                    }
                    
                    // Debug: check alpha values
                    let transparent_count = composited.chunks(4).filter(|p| p.len() == 4 && p[3] < 255).count();
                    let total_pixels = composited.len() / 4;
                    if transparent_count > 0 {
                        log::debug!(
                            "After compositing: {}/{} pixels have alpha < 255",
                            transparent_count,
                            total_pixels
                        );
                    }
                    
                    composited
                } else if frame_data.len() < expected_size && base_data.len() == expected_size {
                    // Partial frame data - just use base for now
                    log::debug!(
                        "Frame data size {} < expected {}, using base frame {}",
                        frame_data.len(),
                        expected_size,
                        base_frame_num
                    );
                    base_data.clone()
                } else {
                    // Pad/truncate to expected size
                    let mut data = frame_data;
                    data.resize(expected_size, 0);
                    data
                }
            } else {
                // Base frame doesn't exist yet (shouldn't happen now), pad the data
                log::warn!("Base frame {} doesn't exist, padding data", base_frame_num);
                let mut data = frame_data;
                data.resize(expected_size, 0);
                data
            }
        } else if frame_data.len() != expected_size {
            // No base frame specified, ensure correct size
            log::debug!(
                "Frame data size {} != expected {}, resizing",
                frame_data.len(),
                expected_size
            );
            let mut data = frame_data;
            data.resize(expected_size, 0);
            data
        } else {
            frame_data
        };

        // Frame gap - use provided value or default to 100ms
        // Negative values mean "gapless" (use previous frame's timing)
        let duration_ms = cmd.frame_gap.unwrap_or(100).max(0) as u32;
        let duration_ms = if duration_ms == 0 { 100 } else { duration_ms };

        // Create the frame
        let frame = AnimationFrame {
            data: final_frame_data,
            duration_ms,
        };

        // Add the new frame (animation is guaranteed to exist now)
        if let Some(ref mut anim) = image.animation {
            let frame_num = cmd.edit_frame.unwrap_or(0);
            
            if frame_num > 0 && (frame_num as usize) <= anim.frames.len() {
                // Replace existing frame (1-indexed)
                anim.frames[frame_num as usize - 1] = frame;
            } else {
                // Append new frame
                anim.total_duration_ms += duration_ms as u64;
                anim.frames.push(frame);
            }
            
            log::debug!(
                "Added animation frame to image {}: now {} frames, {}ms total",
                id,
                anim.frames.len(),
                anim.total_duration_ms
            );
        }

        self.dirty = true;
        
        // Return OK response (quiet mode respected)
        if cmd.quiet >= 1 {
            None
        } else {
            Some(format!("\x1b_Gi={};OK\x1b\\", id))
        }
    }

    /// Handle an animation control command (a=a).
    /// This controls playback of an animated image.
    fn handle_animation_control(&mut self, cmd: &GraphicsCommand) -> Option<String> {
        let id = match cmd.image_id {
            Some(id) => id,
            None => {
                log::warn!("AnimationControl without image_id");
                return self.format_response(cmd, Err(GraphicsError::MissingId));
            }
        };

        log::debug!(
            "AnimationControl: id={}, state={:?}, base_frame={:?}, loop_count={:?}",
            id,
            cmd.animation_state,
            cmd.base_frame,
            cmd.loop_count
        );

        let image = match self.images.get_mut(&id) {
            Some(img) => img,
            None => {
                log::warn!("AnimationControl for non-existent image {}", id);
                return self.format_response(cmd, Err(GraphicsError::ImageNotFound));
            }
        };

        if let Some(ref mut anim) = image.animation {
            // Handle animation state (s key)
            if let Some(state) = cmd.animation_state {
                anim.state = match state {
                    1 => {
                        log::debug!("Animation {} stopped", id);
                        AnimationState::Stopped
                    }
                    2 => {
                        log::debug!("Animation {} loading", id);
                        AnimationState::Loading
                    }
                    3 => {
                        log::debug!("Animation {} running ({} frames)", id, anim.frames.len());
                        // Reset frame start when starting animation
                        anim.frame_start = None;
                        AnimationState::Running
                    }
                    _ => anim.state.clone(),
                };
            }

            // Handle current frame (c key in control context)
            if let Some(frame_num) = cmd.base_frame {
                if frame_num > 0 && (frame_num as usize) <= anim.frames.len() {
                    anim.current_frame = frame_num as usize - 1; // 1-indexed to 0-indexed
                    // Update image data to show this frame
                    image.data = anim.frames[anim.current_frame].data.clone();
                    anim.frame_start = None; // Reset timing
                    log::debug!("Animation {} jumped to frame {}", id, frame_num);
                }
            }

            // Handle loop count (v key)
            if let Some(loop_count) = cmd.loop_count {
                if loop_count == 0 {
                    anim.looping = true;
                    anim.loops_remaining = None; // Infinite
                } else {
                    anim.looping = true;
                    anim.loops_remaining = Some(loop_count);
                }
                log::debug!("Animation {} loop count set to {:?}", id, anim.loops_remaining);
            }

            self.dirty = true;
        } else {
            log::warn!("AnimationControl for non-animated image {}", id);
        }

        // AnimationControl commands don't need OK responses by default
        // Only errors are sent. This matches Kitty behavior - kitten icat
        // doesn't set q= for a=a commands, but expects no OK response.
        None
    }

    /// Store an image from command data.
    fn store_image(
        &mut self,
        cmd: &mut GraphicsCommand,
    ) -> Result<u32, GraphicsError> {
        // Track if we loaded from a file (for WebM which needs path)
        let mut file_path: Option<String> = None;

        // Handle file-based transmission
        match cmd.transmission {
            Transmission::File | Transmission::TempFile => {
                // Payload contains a file path
                let path = std::str::from_utf8(&cmd.payload)
                    .map_err(|_| GraphicsError::FileReadFailed)?;
                let path = path.trim().to_string(); // Clone to avoid borrow issues

                log::debug!("Reading image from file: {}", path);

                // Detect format from file extension
                let path_lower = path.to_lowercase();
                if path_lower.ends_with(".gif") {
                    cmd.format = Format::Gif;
                } else if path_lower.ends_with(".webm")
                    || path_lower.ends_with(".mp4")
                    || path_lower.ends_with(".mkv")
                    || path_lower.ends_with(".avi")
                    || path_lower.ends_with(".mov")
                {
                    // For video files, we'll handle them specially
                    #[cfg(feature = "webm")]
                    {
                        file_path = Some(path.clone());
                    }
                    #[cfg(not(feature = "webm"))]
                    {
                        log::warn!("WebM support not enabled, treating as PNG");
                    }
                } else if path_lower.ends_with(".png") {
                    cmd.format = Format::Png;
                }

                // Read the file (unless it's a video file that needs the path)
                if file_path.is_none() {
                    let file_data = std::fs::read(&path)
                        .map_err(|_| GraphicsError::FileReadFailed)?;
                    cmd.payload = file_data;
                }

                // Delete temp file after reading
                if cmd.transmission == Transmission::TempFile && file_path.is_none() {
                    let _ = std::fs::remove_file(&path);
                }
            }
            Transmission::SharedMemory => {
                // Payload contains a shared memory object name
                let shm_name = std::str::from_utf8(&cmd.payload)
                    .map_err(|_| GraphicsError::FileReadFailed)?;
                let shm_name = shm_name.trim();

                log::debug!("Reading image from shared memory: {}", shm_name);

                // On Linux, shared memory objects are in /dev/shm/
                let shm_path = format!("/dev/shm/{}", shm_name);

                // Read the shared memory file
                let file_data = std::fs::read(&shm_path).map_err(|e| {
                    log::error!(
                        "Failed to read shared memory {}: {}",
                        shm_path,
                        e
                    );
                    GraphicsError::FileReadFailed
                })?;

                // Remove the shared memory object after reading
                let _ = std::fs::remove_file(&shm_path);

                cmd.payload = file_data;
            }
            Transmission::Direct => {
                // Payload is already the data
                // Try to detect format from magic bytes if format is default
                if cmd.format == Format::Rgba && cmd.payload.len() >= 6 {
                    if &cmd.payload[0..6] == b"GIF89a" || &cmd.payload[0..6] == b"GIF87a" {
                        cmd.format = Format::Gif;
                    }
                }
            }
        }

        // Decompress if needed (but not for video files)
        if file_path.is_none() {
            cmd.decompress_payload()?;
        }

        // Decode image data - some formats may include animation data
        let (width, height, data, animation) = if let Some(path) = file_path {
            // Handle video files via FFmpeg
            #[cfg(feature = "webm")]
            {
                decode_webm(&path)?
            }
            #[cfg(not(feature = "webm"))]
            {
                let _ = path;
                return Err(GraphicsError::UnsupportedFormat);
            }
        } else {
            match cmd.format {
                Format::Png => {
                    let (w, h, d) = cmd.decode_png()?;
                    (w, h, d, None)
                }
                Format::Rgba => {
                    let w = cmd.width.ok_or(GraphicsError::MissingDimensions)?;
                    let h = cmd.height.ok_or(GraphicsError::MissingDimensions)?;
                    (w, h, cmd.payload.clone(), None)
                }
                Format::Rgb => {
                    let w = cmd.width.ok_or(GraphicsError::MissingDimensions)?;
                    let h = cmd.height.ok_or(GraphicsError::MissingDimensions)?;
                    (w, h, cmd.rgb_to_rgba(), None)
                }
                Format::Gif => decode_gif(&cmd.payload)?,
            }
        };

        // Assign ID if not provided
        let id = cmd.image_id.unwrap_or_else(|| {
            let id = self.next_id;
            self.next_id += 1;
            id
        });

        self.images.insert(
            id,
            ImageData {
                id,
                width,
                height,
                data,
                animation,
            },
        );
        self.dirty = true;

        Ok(id)
    }

    /// Place an image at the cursor position.
    /// Returns (cols, rows) dimensions of the placement.
    fn place_image(
        &mut self,
        cmd: &GraphicsCommand,
        cursor_col: usize,
        cursor_row: usize,
        cell_width: f32,
        cell_height: f32,
    ) -> (usize, usize) {
        let id = match cmd.image_id {
            Some(id) => id,
            None => return (0, 0),
        };

        let image = match self.images.get(&id) {
            Some(img) => img,
            None => return (0, 0),
        };

        // Calculate display size
        let src_width = if cmd.src_width == 0 {
            image.width
        } else {
            cmd.src_width
        };
        let src_height = if cmd.src_height == 0 {
            image.height
        } else {
            cmd.src_height
        };

        // Calculate columns and rows if not specified
        let cols = if cmd.cols == 0 {
            ((src_width as f32) / cell_width).ceil() as usize
        } else {
            cmd.cols as usize
        };
        let rows = if cmd.rows == 0 {
            ((src_height as f32) / cell_height).ceil() as usize
        } else {
            cmd.rows as usize
        };

        // Don't create actual placement for virtual placements (U=1)
        // Virtual placements are referenced by Unicode placeholders
        if cmd.unicode_placeholder {
            log::debug!(
                "Virtual placement for image id={}, cols={} rows={}",
                id,
                cols,
                rows
            );
            return (cols, rows);
        }

        let placement = ImagePlacement {
            image_id: id,
            placement_id: cmd.placement_id.unwrap_or(0),
            col: cursor_col,
            row: cursor_row,
            cols,
            rows,
            z_index: cmd.z_index,
            src_x: cmd.src_x,
            src_y: cmd.src_y,
            src_width,
            src_height,
            x_offset: cmd.x_offset,
            y_offset: cmd.y_offset,
        };

        // Remove existing placement with same ID if present
        if cmd.placement_id.is_some() {
            self.placements.retain(|p| {
                p.image_id != id || p.placement_id != placement.placement_id
            });
        }

        self.placements.push(placement);
        self.dirty = true;

        (cols, rows)
    }

    /// Format a response for the application.
    fn format_response(
        &self,
        cmd: &GraphicsCommand,
        result: Result<u32, GraphicsError>,
    ) -> Option<String> {
        // Quiet mode 2 suppresses all responses
        if cmd.quiet >= 2 {
            return None;
        }

        match result {
            Ok(id) => {
                // Quiet mode 1 suppresses OK responses
                if cmd.quiet >= 1 {
                    None
                } else {
                    Some(format!("\x1b_Gi={};OK\x1b\\", id))
                }
            }
            Err(e) => {
                let msg = match e {
                    GraphicsError::Base64DecodeFailed => {
                        "ENODATA:base64 decode failed"
                    }
                    GraphicsError::DecompressionFailed => {
                        "ENODATA:decompression failed"
                    }
                    GraphicsError::PngDecodeFailed => {
                        "ENODATA:PNG decode failed"
                    }
                    GraphicsError::GifDecodeFailed => {
                        "ENODATA:GIF decode failed"
                    }
                    GraphicsError::VideoDecodeFailed => {
                        "ENODATA:video decode failed"
                    }
                    GraphicsError::MissingDimensions => {
                        "EINVAL:missing dimensions"
                    }
                    GraphicsError::InvalidData => "ENODATA:invalid data",
                    GraphicsError::ImageNotFound => "ENOENT:image not found",
                    GraphicsError::MissingId => "EINVAL:missing image id",
                    GraphicsError::FileReadFailed => "ENOENT:file read failed",
                    GraphicsError::UnsupportedFormat => {
                        "EINVAL:unsupported format"
                    }
                };
                let id = cmd.image_id.unwrap_or(0);
                Some(format!("\x1b_Gi={};{}\x1b\\", id, msg))
            }
        }
    }

    /// Get all images.
    pub fn images(&self) -> &HashMap<u32, ImageData> {
        &self.images
    }

    /// Get all active placements.
    pub fn placements(&self) -> &[ImagePlacement] {
        &self.placements
    }

    /// Get an image by ID.
    pub fn get_image(&self, id: u32) -> Option<&ImageData> {
        self.images.get(&id)
    }

    /// Clear the dirty flag.
    pub fn clear_dirty(&mut self) {
        self.dirty = false;
    }

    /// Update animations and return list of image IDs that changed frames.
    /// This should be called every frame to advance animations.
    pub fn update_animations(&mut self) -> Vec<u32> {
        let now = Instant::now();
        let mut changed = Vec::new();

        for (id, image) in self.images.iter_mut() {
            if let Some(ref mut anim) = image.animation {
                // Only advance if animation is running
                if anim.state != AnimationState::Running {
                    continue;
                }

                // Initialize frame start time if not set
                if anim.frame_start.is_none() {
                    anim.frame_start = Some(now);
                    log::debug!("Animation {} started, {} frames, first frame {}ms", 
                        id, anim.frames.len(), anim.frames[0].duration_ms);
                }

                let frame_start = anim.frame_start.unwrap();
                let elapsed = now.duration_since(frame_start).as_millis() as u32;
                let current_frame_duration = anim.frames[anim.current_frame].duration_ms;

                if elapsed >= current_frame_duration {
                    // Advance to next frame
                    let old_frame = anim.current_frame;
                    let next_frame = anim.current_frame + 1;
                    if next_frame >= anim.frames.len() {
                        if anim.looping {
                            // Check loop count
                            if let Some(ref mut loops) = anim.loops_remaining {
                                if *loops > 0 {
                                    *loops -= 1;
                                    anim.current_frame = 0;
                                } else {
                                    // No more loops, stop
                                    anim.state = AnimationState::Stopped;
                                    continue;
                                }
                            } else {
                                // Infinite looping
                                anim.current_frame = 0;
                            }
                        }
                        // else: stay on last frame
                    } else {
                        anim.current_frame = next_frame;
                    }

                    log::debug!("Animation {} frame {} -> {} (elapsed {}ms >= {}ms)", 
                        id, old_frame, anim.current_frame, elapsed, current_frame_duration);

                    // Update the image data with the new frame
                    image.data = anim.frames[anim.current_frame].data.clone();
                    anim.frame_start = Some(now);
                    changed.push(*id);
                }
            }
        }

        if !changed.is_empty() {
            self.dirty = true;
        }

        changed
    }

    /// Check if any images have running animations.
    pub fn has_animations(&self) -> bool {
        self.images.values().any(|img| {
            img.animation
                .as_ref()
                .map(|a| a.state == AnimationState::Running && a.frames.len() > 1)
                .unwrap_or(false)
        })
    }

    /// Get mutable access to an image by ID.
    pub fn get_image_mut(&mut self, id: u32) -> Option<&mut ImageData> {
        self.images.get_mut(&id)
    }
}

/// Simple base64 decoder.
fn base64_decode(input: &str) -> Result<Vec<u8>, GraphicsError> {
    const DECODE_TABLE: [i8; 256] = {
        let mut table = [-1i8; 256];
        let chars =
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut i = 0;
        while i < 64 {
            table[chars[i] as usize] = i as i8;
            i += 1;
        }
        table
    };

    let input = input.as_bytes();
    let mut output = Vec::with_capacity(input.len() * 3 / 4);
    let mut buffer = 0u32;
    let mut bits = 0;

    for &byte in input {
        if byte == b'=' || byte == b'\n' || byte == b'\r' || byte == b' ' {
            continue;
        }
        let value = DECODE_TABLE[byte as usize];
        if value < 0 {
            return Err(GraphicsError::Base64DecodeFailed);
        }
        buffer = (buffer << 6) | (value as u32);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            output.push((buffer >> bits) as u8);
            buffer &= (1 << bits) - 1;
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_basic_command() {
        let data = b"Ga=T,f=100,i=1;iVBORw0KGgo=";
        let cmd = GraphicsCommand::parse(data).unwrap();
        assert_eq!(cmd.action, Action::TransmitAndDisplay);
        assert_eq!(cmd.format, Format::Png);
        assert_eq!(cmd.image_id, Some(1));
    }

    #[test]
    fn test_parse_query() {
        let data = b"Ga=q,i=31;";
        let cmd = GraphicsCommand::parse(data).unwrap();
        assert_eq!(cmd.action, Action::Query);
        assert_eq!(cmd.image_id, Some(31));
    }

    #[test]
    fn test_base64_decode() {
        let decoded = base64_decode("SGVsbG8=").unwrap();
        assert_eq!(decoded, b"Hello");
    }
}
