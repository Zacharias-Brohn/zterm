use zterm::terminal::Terminal;
use zterm::vt_parser::Parser;
use std::time::Instant;
use std::io::Write;

const ASCII_PRINTABLE: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ  `~!@#$%^&*()_+-=[]{}\\|;:'\",<.>/?";
const CONTROL_CHARS: &[u8] = b"\n\t";

// Match Kitty's default repetitions
const REPETITIONS: usize = 100;

fn random_string(len: usize, rng: &mut u64) -> Vec<u8> {
    let alphabet_len = (ASCII_PRINTABLE.len() + CONTROL_CHARS.len()) as u64;
    let mut result = Vec::with_capacity(len);
    for _ in 0..len {
        // Simple LCG random
        *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let idx = ((*rng >> 33) % alphabet_len) as usize;
        if idx < ASCII_PRINTABLE.len() {
            result.push(ASCII_PRINTABLE[idx]);
        } else {
            result.push(CONTROL_CHARS[idx - ASCII_PRINTABLE.len()]);
        }
    }
    result
}

/// Run a benchmark with multiple repetitions like Kitty does
fn run_benchmark<F>(name: &str, data: &[u8], repetitions: usize, mut setup: F) 
where
    F: FnMut() -> (Terminal, Parser),
{
    let data_size = data.len();
    let total_size = data_size * repetitions;
    
    // Warmup run
    let (mut terminal, mut parser) = setup();
    parser.parse(data, &mut terminal);
    
    // Timed runs
    let start = Instant::now();
    for _ in 0..repetitions {
        let (mut terminal, mut parser) = setup();
        parser.parse(data, &mut terminal);
    }
    let elapsed = start.elapsed();
    
    let mb = total_size as f64 / 1024.0 / 1024.0;
    let rate = mb / elapsed.as_secs_f64();
    
    println!("  {:<24} : {:>6.2}s @ {:.1} MB/s ({} reps, {:.2} MB each)", 
        name, elapsed.as_secs_f64(), rate, repetitions, data_size as f64 / 1024.0 / 1024.0);
}

fn main() {
    println!("=== ZTerm VT Parser Benchmark ===");
    println!("Matching Kitty's kitten __benchmark__ methodology\n");
    
    // Benchmark 1: Only ASCII chars (matches Kitty's simple_ascii)
    println!("--- Only ASCII chars ---");
    let target_sz = 1024 * 2048 + 13;
    let mut rng: u64 = 12345;
    let mut ascii_data = Vec::with_capacity(target_sz);
    let alphabet = [ASCII_PRINTABLE, CONTROL_CHARS].concat();
    for _ in 0..target_sz {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let idx = ((rng >> 33) % alphabet.len() as u64) as usize;
        ascii_data.push(alphabet[idx]);
    }
    
    run_benchmark("Only ASCII chars", &ascii_data, REPETITIONS, || {
        (Terminal::new(80, 25, 20000), Parser::new())
    });
    
    // Benchmark 2: CSI codes with few chars (matches Kitty's ascii_with_csi)
    println!("\n--- CSI codes with few chars ---");
    let target_sz = 1024 * 1024 + 17;
    let mut csi_data = Vec::with_capacity(target_sz + 100);
    let mut rng: u64 = 12345; // Fixed seed for reproducibility
    
    while csi_data.len() < target_sz {
        // Simple LCG random for chunk selection
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let q = ((rng >> 33) % 100) as u32;
        
        match q {
            0..=9 => {
                // 10%: random ASCII text (1-72 chars)
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let len = ((rng >> 33) % 72 + 1) as usize;
                csi_data.extend(random_string(len, &mut rng));
            }
            10..=29 => {
                // 20%: cursor movement
                csi_data.extend_from_slice(b"\x1b[m\x1b[?1h\x1b[H");
            }
            30..=39 => {
                // 10%: basic SGR attributes
                csi_data.extend_from_slice(b"\x1b[1;2;3;4:3;31m");
            }
            40..=49 => {
                // 10%: SGR with 256-color + RGB (colon-separated subparams)
                csi_data.extend_from_slice(b"\x1b[38:5:24;48:2:125:136:147m");
            }
            50..=59 => {
                // 10%: SGR with underline color
                csi_data.extend_from_slice(b"\x1b[58;5;44;2m");
            }
            60..=79 => {
                // 20%: cursor movement + erase
                csi_data.extend_from_slice(b"\x1b[m\x1b[10A\x1b[3E\x1b[2K");
            }
            _ => {
                // 20%: reset + cursor + repeat + mode
                csi_data.extend_from_slice(b"\x1b[39m\x1b[10`a\x1b[100b\x1b[?1l");
            }
        }
    }
    csi_data.extend_from_slice(b"\x1b[m");
    
    run_benchmark("CSI codes with few chars", &csi_data, REPETITIONS, || {
        (Terminal::new(80, 25, 20000), Parser::new())
    });
    
    // Benchmark 3: Long escape codes (matches Kitty's long_escape_codes)
    println!("\n--- Long escape codes ---");
    let mut long_esc_data = Vec::new();
    let long_content: String = (0..8024).map(|i| ASCII_PRINTABLE[i % ASCII_PRINTABLE.len()] as char).collect();
    for _ in 0..1024 {
        // OSC 6 - document reporting, ignored after parsing
        long_esc_data.extend_from_slice(b"\x1b]6;");
        long_esc_data.extend_from_slice(long_content.as_bytes());
        long_esc_data.push(0x07); // BEL terminator
    }
    
    run_benchmark("Long escape codes", &long_esc_data, REPETITIONS, || {
        (Terminal::new(80, 25, 20000), Parser::new())
    });
    
    println!("\n=== Benchmark Complete ===");
    println!("\nNote: These benchmarks include terminal state updates but NOT GPU rendering.");
    println!("Compare with: kitten __benchmark__  (without --render flag)");
}
