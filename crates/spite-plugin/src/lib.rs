//! Priority-chain plugin registry.
//!
//! The central override mechanism for the entire spite engine.
//! The same philosophy as the kernel dispatcher — specific beats generic,
//! anyone can slot in at any level, the fallback always works — but
//! generalized to *every* subsystem, not just GPU compute ops.
//!
//! # How it works
//!
//! Every pluggable subsystem exposes a trait (e.g. `Sampler`, `Tokenize`,
//! `Cache`, `Schedule`).  Implementations are registered against a
//! [`PluginKey`] that describes *when* they apply.  At resolution time the
//! registry walks registrations from most-specific to least-specific and
//! returns the first match.
//!
//! ```text
//! Registry resolution order  (higher specificity wins)
//! ──────────────────────────────────────────────────────
//!  key { model_arch: "llama4", task: "chat" }   ← most specific
//!  key { model_arch: "llama4" }
//!  key { task: "chat" }
//!  key { }                                       ← wildcard / global default
//! ```
//!
//! This mirrors the kernel dispatch chain:
//! ```text
//!  kernels/llama/llama4/sm_89/rtx_4090/Q4_K_M/  ← most specific
//!  ...
//!  kernels/generic/generic/                       ← always present
//! ```
//!
//! # Example
//!
//! ```rust
//! use spite_plugin::{Registry, PluginKey};
//!
//! trait Greet: Send + Sync { fn hello(&self) -> &str; }
//! struct EnglishGreeting;
//! impl Greet for EnglishGreeting { fn hello(&self) -> &str { "hello" } }
//! struct FrenchGreeting;
//! impl Greet for FrenchGreeting { fn hello(&self) -> &str { "bonjour" } }
//!
//! let mut reg: Registry<dyn Greet> = Registry::new();
//! reg.set_default(Box::new(EnglishGreeting));
//! reg.register(
//!     PluginKey::for_task("fr_chat"),
//!     Box::new(FrenchGreeting),
//! );
//!
//! let q = PluginKey::for_task("fr_chat");
//! assert_eq!(reg.resolve(&q).unwrap().hello(), "bonjour");
//!
//! let q2 = PluginKey::default();
//! assert_eq!(reg.resolve(&q2).unwrap().hello(), "hello");
//! ```

/// Match context for registry resolution.
///
/// Each `None` field is a wildcard — it matches any value.
/// A key with more `Some` fields is more specific and wins over a more
/// general registration.
#[derive(Clone, Default, PartialEq, Eq, Hash, Debug)]
pub struct PluginKey {
    /// GGUF model architecture string: "llama4", "mistral4", "gemma4", …
    pub model_arch: Option<String>,
    /// GPU architecture string: "sm_89", "rdna3", "metal", …
    pub gpu_arch: Option<String>,
    /// Task label: "chat", "completion", "embed", "rerank", …
    pub task: Option<String>,
}

impl PluginKey {
    /// Convenience: a key that matches only the given model arch.
    pub fn for_model(arch: impl Into<String>) -> Self {
        Self {
            model_arch: Some(arch.into()),
            ..Default::default()
        }
    }

    /// Convenience: a key that matches only the given task.
    pub fn for_task(task: impl Into<String>) -> Self {
        Self {
            task: Some(task.into()),
            ..Default::default()
        }
    }

    /// Convenience: a key that matches model + task.
    pub fn for_model_task(arch: impl Into<String>, task: impl Into<String>) -> Self {
        Self {
            model_arch: Some(arch.into()),
            task: Some(task.into()),
            ..Default::default()
        }
    }

    /// Number of non-wildcard fields — higher means more specific.
    pub fn specificity(&self) -> u32 {
        [
            self.model_arch.is_some(),
            self.gpu_arch.is_some(),
            self.task.is_some(),
        ]
        .iter()
        .filter(|&&b| b)
        .count() as u32
    }

    /// Does this key (as a pattern) match the given query?
    /// `None` in `self` is a wildcard: it matches any value in `query`.
    pub fn matches(&self, query: &PluginKey) -> bool {
        fn field_ok(pattern: &Option<String>, value: &Option<String>) -> bool {
            match (pattern, value) {
                (None, _) => true,            // wildcard
                (Some(p), Some(v)) => p == v, // exact match
                (Some(_), None) => false,     // pattern requires a value
            }
        }
        field_ok(&self.model_arch, &query.model_arch)
            && field_ok(&self.gpu_arch, &query.gpu_arch)
            && field_ok(&self.task, &query.task)
    }
}

/// A priority-ordered registry of trait-object implementations.
///
/// `T` is an unsized trait: `dyn Sampler`, `dyn Cache`, etc.
/// Implementations registered with higher-specificity keys win.
/// When two registrations have equal specificity, the one registered
/// **first** wins (stable ordering).
pub struct Registry<T: ?Sized + 'static> {
    /// `(key, implementation)` pairs, sorted by descending specificity.
    entries: Vec<(PluginKey, Box<T>)>,
}

impl<T: ?Sized + 'static> Default for Registry<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: ?Sized + 'static> Registry<T> {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Register `implementation` under `key`.
    ///
    /// The registry re-sorts after each insertion so `resolve` always
    /// sees the most-specific match first.  This is a startup operation
    /// (called during engine initialisation, not per-request), so the
    /// O(n log n) sort cost is acceptable.
    pub fn register(&mut self, key: PluginKey, implementation: Box<T>) {
        self.entries.push((key, implementation));
        // Stable sort: equal-specificity entries keep insertion order.
        self.entries
            .sort_by_key(|b| std::cmp::Reverse(b.0.specificity()));
    }

    /// Register `implementation` as the global default (wildcard key).
    ///
    /// Any query that matches nothing more specific will land here.
    pub fn set_default(&mut self, implementation: Box<T>) {
        self.register(PluginKey::default(), implementation);
    }

    /// Return the highest-priority implementation whose key matches `query`.
    ///
    /// Returns `None` only if the registry is empty.
    pub fn resolve(&self, query: &PluginKey) -> Option<&T> {
        self.entries
            .iter()
            .find(|(key, _)| key.matches(query))
            .map(|(_, imp)| imp.as_ref())
    }

    /// Mutable version of `resolve`.
    pub fn resolve_mut(&mut self, query: &PluginKey) -> Option<&mut T> {
        self.entries
            .iter_mut()
            .find(|(key, _)| key.matches(query))
            .map(|(_, imp)| imp.as_mut())
    }

    /// Number of registered implementations.
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    trait Greeter: Send + Sync {
        fn greet(&self) -> &'static str;
    }

    struct En;
    struct Fr;
    struct FrChat;

    impl Greeter for En {
        fn greet(&self) -> &'static str {
            "hello"
        }
    }
    impl Greeter for Fr {
        fn greet(&self) -> &'static str {
            "bonjour"
        }
    }
    impl Greeter for FrChat {
        fn greet(&self) -> &'static str {
            "salut"
        }
    }

    fn make_registry() -> Registry<dyn Greeter> {
        let mut r: Registry<dyn Greeter> = Registry::new();
        r.set_default(Box::new(En));
        r.register(PluginKey::for_task("fr"), Box::new(Fr));
        r.register(PluginKey::for_model_task("x", "fr"), Box::new(FrChat));
        r
    }

    #[test]
    fn wildcard_is_fallback() {
        let r = make_registry();
        let q = PluginKey::default();
        assert_eq!(r.resolve(&q).unwrap().greet(), "hello");
    }

    #[test]
    fn task_beats_wildcard() {
        let r = make_registry();
        let q = PluginKey::for_task("fr");
        assert_eq!(r.resolve(&q).unwrap().greet(), "bonjour");
    }

    #[test]
    fn model_task_beats_task() {
        let r = make_registry();
        let q = PluginKey::for_model_task("x", "fr");
        assert_eq!(r.resolve(&q).unwrap().greet(), "salut");
    }

    #[test]
    fn unknown_task_falls_through_to_wildcard() {
        let r = make_registry();
        let q = PluginKey::for_task("de");
        assert_eq!(r.resolve(&q).unwrap().greet(), "hello");
    }

    #[test]
    fn empty_registry_returns_none() {
        let r: Registry<dyn Greeter> = Registry::new();
        assert!(r.resolve(&PluginKey::default()).is_none());
    }

    #[test]
    fn specificity_ordering() {
        assert!(
            PluginKey::for_model_task("m", "t").specificity()
                > PluginKey::for_task("t").specificity()
        );
        assert!(PluginKey::for_task("t").specificity() > PluginKey::default().specificity());
    }
}
