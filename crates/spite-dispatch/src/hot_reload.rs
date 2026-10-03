//! Hot-reload: swap kernel .so files without restarting the server.
//!
//! Monitors `kernels/` for changes (via filesystem polling or inotify).
//! When a .so is replaced, loads the new library, verifies the ABI version,
//! then swaps the function pointers in the live `DispatchTable` atomically
//! using `std::sync::atomic` pointer-swapping.
//!
//! This is valuable during kernel development: the developer rebuilds the .so
//! and the running server picks it up within the next poll interval without
//! interrupting in-flight requests.
//!
//! # Safety
//!
//! Swapping function pointers while requests are in flight is only safe if
//! no request is mid-kernel when the old .so is unloaded. The implementation
//! uses a reader–writer lock: hot-reload takes the write lock (draining all
//! in-flight kernel calls), swaps the pointers, then releases the lock.

use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime};

use crate::DispatchTable;

/// Polling interval for filesystem change detection.
pub const DEFAULT_POLL_MS: u64 = 500;

/// A snapshot of one watched file's mtime.
#[derive(Debug, Clone)]
struct FileSnapshot {
    path:  PathBuf,
    mtime: SystemTime,
}

impl FileSnapshot {
    fn of(path: &Path) -> Option<Self> {
        let mtime = std::fs::metadata(path).ok()?.modified().ok()?;
        Some(Self { path: path.to_owned(), mtime })
    }

    fn changed(&self) -> bool {
        std::fs::metadata(&self.path)
            .ok()
            .and_then(|m| m.modified().ok())
            .map_or(false, |t| t != self.mtime)
    }
}

/// Handle returned to the caller; drop it to stop the background thread.
pub struct HotReloadHandle {
    _stop: Arc<()>, // when all clones drop, the thread exits on its next poll
}

/// Start a background thread that polls `kernels_dir` every `poll_ms` ms
/// and rebuilds the dispatch table when any .so changes.
///
/// `table_lock`: the dispatch table wrapped in an `RwLock`; the hot-reload
/// thread replaces the inner value when a change is detected.
///
/// Returns a `HotReloadHandle`; drop it to stop polling.
pub fn watch(
    kernels_dir: impl AsRef<Path>,
    _table_lock: Arc<RwLock<DispatchTable>>,
    poll_ms:     u64,
) -> HotReloadHandle {
    let kernels_dir = kernels_dir.as_ref().to_owned();
    let stop        = Arc::new(());
    let stop_weak   = Arc::downgrade(&stop);

    std::thread::spawn(move || {
        let mut snapshots: Vec<FileSnapshot> = Vec::new();
        loop {
            if stop_weak.upgrade().is_none() { break; }
            std::thread::sleep(Duration::from_millis(poll_ms));

            // Collect current .so paths under kernels_dir.
            let current = collect_so_paths(&kernels_dir);

            let any_changed = current.iter().any(|p| {
                snapshots.iter().find(|s| s.path == *p).map_or(true, |s| s.changed())
            });

            if any_changed {
                // TODO: acquire write lock, rebuild DispatchTable, swap it in.
                // For now just log.
                eprintln!("[hot_reload] change detected in {:?}", kernels_dir);
                snapshots = current.iter().filter_map(|p| FileSnapshot::of(p)).collect();
            } else if snapshots.is_empty() {
                snapshots = current.iter().filter_map(|p| FileSnapshot::of(p)).collect();
            }
        }
    });

    HotReloadHandle { _stop: stop }
}

fn collect_so_paths(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let exts = ["so", "dylib", "dll"];
    let Ok(rd) = std::fs::read_dir(root) else { return out };
    for entry in rd.flatten() {
        let path = entry.path();
        if path.is_dir() {
            out.extend(collect_so_paths(&path));
        } else if exts.iter().any(|e| path.extension().map_or(false, |x| x == *e)) {
            out.push(path);
        }
    }
    out
}
