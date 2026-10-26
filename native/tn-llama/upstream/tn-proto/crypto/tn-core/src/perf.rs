//! Internal stage-timing instrumentation (`TN_PERF_TRACE`) for the write hot
//! path; not a protocol surface. See [`crate::Runtime`] for the actual
//! event API. Reach here directly only to read or reset stage timings.
//!
//! Enabled when the `TN_PERF_TRACE` env var is set at process start.
//! Each named stage accumulates `count` and `total_ns` into a static
//! `Mutex<HashMap>` keyed by `&'static str`. Snapshot via
//! [`snapshot`]; reset via [`reset`].
//!
//! A mutex protects the shared map while stage names are inserted and
//! counters are updated. Stage names do not require a fixed compile-time list.
//!
//! When disabled, [`time_stage`] checks the atomic enable flag and calls
//! the supplied function without reading the clock or recording a duration.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

static ENABLED: AtomicBool = AtomicBool::new(false);

fn counters() -> &'static Mutex<HashMap<&'static str, StageStats>> {
    static CELL: OnceLock<Mutex<HashMap<&'static str, StageStats>>> = OnceLock::new();
    CELL.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Per-stage accumulator.
#[derive(Default, Clone, Copy)]
pub struct StageStats {
    /// Number of times the stage executed.
    pub count: u64,
    /// Sum of stage wall-clock durations in nanoseconds.
    pub total_ns: u64,
}

/// Initialize from the env. Call once at process start (e.g. from
/// `Runtime::init`). Idempotent.
pub fn init_from_env() {
    let on = std::env::var("TN_PERF_TRACE")
        .ok()
        .filter(|s| !s.is_empty() && s != "0")
        .is_some();
    ENABLED.store(on, Ordering::Relaxed);
}

/// Whether instrumentation was enabled through the environment.
#[inline]
pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// Time `f`, attributing the elapsed nanoseconds to `stage`.
///
/// When disabled, call `f()` after checking the enable flag. When enabled,
/// measure its duration and update the stage counters under a mutex.
#[inline]
pub fn time_stage<F, R>(stage: &'static str, f: F) -> R
where
    F: FnOnce() -> R,
{
    if !enabled() {
        return f();
    }
    let t0 = Instant::now();
    let r = f();
    let dt_ns = t0.elapsed().as_nanos() as u64;
    record(stage, dt_ns);
    r
}

/// Attribute `ns` nanoseconds to `stage` directly. Useful when the
/// caller already captured an `Instant` (e.g. inside a closure that
/// can `return` early) and wants to record without re-wrapping.
///
/// When [`enabled`] is false this still acquires the mutex — use
/// [`enabled`] to guard the timing capture itself in tight code paths.
#[inline]
pub fn record_ns(stage: &'static str, ns: u64) {
    record(stage, ns);
}

fn record(stage: &'static str, ns: u64) {
    let Ok(mut g) = counters().lock() else {
        return;
    };
    let s = g.entry(stage).or_default();
    s.count += 1;
    s.total_ns = s.total_ns.saturating_add(ns);
}

/// Snapshot the current accumulator state.
///
/// Returned vector is sorted by `total_ns` descending so the most
/// expensive stages appear first.
pub fn snapshot() -> Vec<(&'static str, StageStats)> {
    let g = counters().lock().expect("perf counters mutex poisoned");
    let mut out: Vec<(&'static str, StageStats)> = g.iter().map(|(k, v)| (*k, *v)).collect();
    out.sort_by(|a, b| b.1.total_ns.cmp(&a.1.total_ns));
    out
}

/// Zero all accumulators. Useful when the caller wants to discard
/// warmup costs before a measurement window.
pub fn reset() {
    let mut g = counters().lock().expect("perf counters mutex poisoned");
    g.clear();
}
