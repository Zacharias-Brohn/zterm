//! Test ligature detection - compare individual vs combined shaping
use rustybuzz::{Face, UnicodeBuffer, Feature};
use ttf_parser::Tag;
use fontdue::Font;
use std::fs;

fn main() {
    let path = "/usr/share/fonts/TTF/0xProtoNerdFontMono-Regular.ttf";
    println!("Using font: {}", path);
    
    let font_data = fs::read(path).expect("Failed to read font");
    let face = Face::from_slice(&font_data, 0).expect("Failed to parse font");
    let fontdue_font = Font::from_bytes(&font_data[..], fontdue::FontSettings::default()).unwrap();
    
    let font_size = 16.0;
    let units_per_em = face.units_per_em() as f32;
    
    println!("Font units per em: {}", units_per_em);
    
    // Get cell width from a regular character
    let (hyphen_metrics, _) = fontdue_font.rasterize('-', font_size);
    let cell_width = hyphen_metrics.advance_width;
    println!("Cell width (from '-'): {:.2}px", cell_width);
    
    let features = vec![
        Feature::new(Tag::from_bytes(b"liga"), 1, ..),
        Feature::new(Tag::from_bytes(b"calt"), 1, ..),
        Feature::new(Tag::from_bytes(b"dlig"), 1, ..),
    ];
    
    let test_strings = ["->", "=>", "==", "!=", ">=", "<="];
    
    for s in &test_strings {
        // Shape combined string
        let mut buffer = UnicodeBuffer::new();
        buffer.push_str(s);
        let combined = rustybuzz::shape(&face, &features, buffer);
        let combined_infos = combined.glyph_infos();
        let combined_positions = combined.glyph_positions();
        
        // Shape each character individually  
        let mut individual_glyphs = Vec::new();
        for c in s.chars() {
            let mut buf = UnicodeBuffer::new();
            buf.push_str(&c.to_string());
            let shaped = rustybuzz::shape(&face, &features, buf);
            individual_glyphs.push(shaped.glyph_infos()[0].glyph_id);
        }
        
        println!("\n'{}' analysis:", s);
        println!("  Combined glyphs:   {:?}", combined_infos.iter().map(|i| i.glyph_id).collect::<Vec<_>>());
        println!("  Individual glyphs: {:?}", individual_glyphs);
        
        // Show advances for each glyph
        for (i, (info, pos)) in combined_infos.iter().zip(combined_positions.iter()).enumerate() {
            let advance_px = pos.x_advance as f32 * font_size / units_per_em;
            println!("  Glyph {}: id={}, advance={} units ({:.2}px)", i, info.glyph_id, pos.x_advance, advance_px);
            
            // Rasterize and show metrics
            let (metrics, _) = fontdue_font.rasterize_indexed(info.glyph_id as u16, font_size);
            println!("    Rasterized: {}x{} px, xmin={}, ymin={}, advance_width={:.2}", 
                metrics.width, metrics.height, metrics.xmin, metrics.ymin, metrics.advance_width);
        }
        
        // Check if any glyph was substituted
        let has_substitution = combined_infos.iter().zip(individual_glyphs.iter())
            .any(|(combined, &individual)| combined.glyph_id != individual);
        println!("  Has substitution: {}", has_substitution);
    }
    
    // Also test what Kitty does - check glyph names
    println!("\n=== Checking glyph names via ttf-parser ===");
    let ttf_face = ttf_parser::Face::parse(&font_data, 0).unwrap();
    
    // Shape "->" and get glyph names
    let mut buffer = UnicodeBuffer::new();
    buffer.push_str("->");
    let combined = rustybuzz::shape(&face, &features, buffer);
    for info in combined.glyph_infos() {
        let glyph_id = ttf_parser::GlyphId(info.glyph_id as u16);
        if let Some(name) = ttf_face.glyph_name(glyph_id) {
            println!("  Glyph {} name: {}", info.glyph_id, name);
        } else {
            println!("  Glyph {} has no name", info.glyph_id);
        }
    }
}
