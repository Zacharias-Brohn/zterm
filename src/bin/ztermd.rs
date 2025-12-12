//! ZTerm Daemon - Background process that manages terminal sessions.

use zterm::daemon::Daemon;

fn main() {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    
    log::info!("ZTerm daemon starting...");
    
    match Daemon::new() {
        Ok(mut daemon) => {
            if let Err(e) = daemon.run() {
                log::error!("Daemon error: {}", e);
                std::process::exit(1);
            }
        }
        Err(e) => {
            log::error!("Failed to start daemon: {}", e);
            std::process::exit(1);
        }
    }
    
    log::info!("ZTerm daemon exiting");
}
