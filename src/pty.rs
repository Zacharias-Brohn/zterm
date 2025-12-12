//! PTY (pseudo-terminal) handling for shell communication.

use rustix::fs::{fcntl_setfl, OFlags};
use rustix::io::{read, write, Errno};
use rustix::pty::{grantpt, openpt, ptsname, unlockpt, OpenptFlags};
use std::ffi::CString;
use std::os::fd::{AsFd, BorrowedFd, OwnedFd};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum PtyError {
    #[error("Failed to open PTY master: {0}")]
    OpenMaster(#[source] rustix::io::Errno),
    #[error("Failed to grant PTY: {0}")]
    Grant(#[source] rustix::io::Errno),
    #[error("Failed to unlock PTY: {0}")]
    Unlock(#[source] rustix::io::Errno),
    #[error("Failed to get PTS name: {0}")]
    PtsName(#[source] rustix::io::Errno),
    #[error("Failed to open PTS: {0}")]
    OpenSlave(#[source] rustix::io::Errno),
    #[error("Failed to fork: {0}")]
    Fork(#[source] std::io::Error),
    #[error("Failed to execute shell: {0}")]
    Exec(#[source] std::io::Error),
    #[error("I/O error: {0}")]
    Io(#[source] std::io::Error),
}

/// Represents the master side of a PTY pair.
pub struct Pty {
    master: OwnedFd,
    child_pid: rustix::process::Pid,
}

impl Pty {
    /// Creates a new PTY and spawns a shell process.
    pub fn spawn(shell: Option<&str>) -> Result<Self, PtyError> {
        // Open the PTY master
        let master = openpt(OpenptFlags::RDWR | OpenptFlags::NOCTTY | OpenptFlags::CLOEXEC)
            .map_err(PtyError::OpenMaster)?;

        // Set non-blocking mode on master
        fcntl_setfl(&master, OFlags::NONBLOCK).map_err(|e| PtyError::Io(e.into()))?;

        // Grant and unlock the PTY
        grantpt(&master).map_err(PtyError::Grant)?;
        unlockpt(&master).map_err(PtyError::Unlock)?;

        // Get the slave name
        let slave_name = ptsname(&master, Vec::new()).map_err(PtyError::PtsName)?;

        // Fork the process
        // SAFETY: We're careful to only use async-signal-safe functions in the child
        let fork_result = unsafe { libc::fork() };

        match fork_result {
            -1 => Err(PtyError::Fork(std::io::Error::last_os_error())),
            0 => {
                // Child process
                Self::setup_child(&slave_name, shell);
            }
            pid => {
                // Parent process
                let child_pid = unsafe { rustix::process::Pid::from_raw_unchecked(pid) };
                Ok(Self { master, child_pid })
            }
        }
    }

    /// Sets up the child process (runs in forked child).
    fn setup_child(slave_name: &CString, shell: Option<&str>) -> ! {
        // Create a new session
        unsafe { libc::setsid() };

        // Open the slave PTY using libc for async-signal-safety
        let slave_fd = unsafe { libc::open(slave_name.as_ptr(), libc::O_RDWR) };
        if slave_fd < 0 {
            unsafe { libc::_exit(1) };
        }

        // Set as controlling terminal
        unsafe { libc::ioctl(slave_fd, libc::TIOCSCTTY, 0) };

        // Duplicate slave to stdin/stdout/stderr
        unsafe {
            libc::dup2(slave_fd, 0);
            libc::dup2(slave_fd, 1);
            libc::dup2(slave_fd, 2);
        }

        // Close the original slave fd if it's not 0, 1, or 2
        if slave_fd > 2 {
            unsafe { libc::close(slave_fd) };
        }

        // Determine which shell to use
        let shell_path = shell
            .map(String::from)
            .or_else(|| std::env::var("SHELL").ok())
            .unwrap_or_else(|| "/bin/sh".to_string());

        let shell_cstr = CString::new(shell_path.clone()).expect("Invalid shell path");
        let shell_name = std::path::Path::new(&shell_path)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("sh");

        // Login shell (prepend with -)
        let login_shell = CString::new(format!("-{}", shell_name)).expect("Invalid shell name");

        // Execute the shell
        let args = [login_shell.as_ptr(), std::ptr::null()];

        unsafe {
            libc::execvp(shell_cstr.as_ptr(), args.as_ptr());
        }

        // If exec fails, exit
        std::process::exit(1);
    }

    /// Returns a reference to the master file descriptor.
    pub fn master_fd(&self) -> BorrowedFd<'_> {
        self.master.as_fd()
    }

    /// Reads data from the PTY master.
    /// Returns Ok(0) if no data is available (non-blocking).
    pub fn read(&self, buf: &mut [u8]) -> Result<usize, PtyError> {
        match read(&self.master, buf) {
            Ok(n) => Ok(n),
            Err(Errno::AGAIN) => Ok(0), // WOULDBLOCK is same as AGAIN on Linux
            Err(e) => Err(PtyError::Io(e.into())),
        }
    }

    /// Writes data to the PTY master.
    pub fn write(&self, buf: &[u8]) -> Result<usize, PtyError> {
        write(&self.master, buf).map_err(|e| PtyError::Io(e.into()))
    }

    /// Resizes the PTY window.
    pub fn resize(&self, cols: u16, rows: u16) -> Result<(), PtyError> {
        let winsize = libc::winsize {
            ws_row: rows,
            ws_col: cols,
            ws_xpixel: 0,
            ws_ypixel: 0,
        };

        let fd = std::os::fd::AsRawFd::as_raw_fd(&self.master);
        let result = unsafe { libc::ioctl(fd, libc::TIOCSWINSZ, &winsize) };

        if result == -1 {
            Err(PtyError::Io(std::io::Error::last_os_error()))
        } else {
            Ok(())
        }
    }

    /// Returns the child process ID.
    pub fn child_pid(&self) -> rustix::process::Pid {
        self.child_pid
    }
}

impl Drop for Pty {
    fn drop(&mut self) {
        // Send SIGHUP to the child process
        unsafe {
            libc::kill(self.child_pid.as_raw_nonzero().get(), libc::SIGHUP);
        }
    }
}
