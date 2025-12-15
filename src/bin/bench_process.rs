use zterm::terminal::Terminal;
use std::time::Instant;
use std::io::Write;

fn main() {
    // Generate seq 1 100000 output
    let mut data = Vec::new();
    for i in 1..=100000 {
        writeln!(&mut data, "{}", i).unwrap();
    }
    println!("Data size: {} bytes", data.len());
    
    // Test with different terminal sizes to see scroll impact
    for rows in [24, 100, 1000] {
        let mut terminal = Terminal::new(80, rows, 10000);
        let start = Instant::now();
        terminal.process(&data);
        let elapsed = start.elapsed();
        println!("Terminal {}x{}: {:?} ({:.2} MB/s)", 
            80, rows,
            elapsed, 
            (data.len() as f64 / 1024.0 / 1024.0) / elapsed.as_secs_f64()
        );
    }
    
    // Test with scrollback disabled
    println!("\nWith scrollback disabled:");
    let mut terminal = Terminal::new(80, 24, 0);
    let start = Instant::now();
    terminal.process(&data);
    let elapsed = start.elapsed();
    println!("Terminal 80x24, no scrollback: {:?} ({:.2} MB/s)", 
        elapsed, 
        (data.len() as f64 / 1024.0 / 1024.0) / elapsed.as_secs_f64()
    );
}
