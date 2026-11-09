//! One-off experiment: compare the old sBPF JIT with the new ("parrot") one on real workloads.
//!
//! `Executable` has a single slot for the compiled program, and both JITs store their output in
//! it, so a single executable can only ever run one of them. Each cache entry therefore carries a
//! second `Executable` of the same ELF that is compiled with the parrot JIT.
//!
//! Which of the two executables runs a program is decided by [`with_jit`]: the SVM executes every
//! transaction batch twice, once under each JIT, and compares the outcomes (see
//! `solana_svm::jit_comparison`). Running whole batches rather than individual program invocations
//! keeps the executions independent: a program run has plenty of side effects (accounts, CPIs,
//! logs, return data, compute meter) that could not be undone in the middle of a transaction.
//!
//! Enable with `AGAVE_SBPF_JIT_COMPARE_PATH=<file>`. The per-program statistics are written (as
//! CSV with a header) to that file whenever [`ProgramCache::output_entry_stats`] runs.
//!
//! [`ProgramCache::output_entry_stats`]: crate::program_cache::global::ProgramCache::output_entry_stats

use {
    crate::invoke_context::InvokeContext,
    solana_sbpf::elf::Executable,
    std::{
        cell::Cell,
        ffi::OsString,
        fmt::Write,
        sync::{
            Arc, LazyLock, Mutex,
            atomic::{AtomicU64, Ordering},
        },
    },
};

type ProgramExecutable = Executable<InvokeContext<'static, 'static>>;

pub const PATH_ENV: &str = "AGAVE_SBPF_JIT_COMPARE_PATH";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JitKind {
    Old,
    Parrot,
}

thread_local! {
    static ACTIVE_JIT: Cell<JitKind> = const { Cell::new(JitKind::Old) };
}

/// Run `f`, executing all programs on this thread with the JIT `kind`.
pub fn with_jit<R>(kind: JitKind, f: impl FnOnce() -> R) -> R {
    struct Restore(JitKind);
    impl Drop for Restore {
        fn drop(&mut self) {
            ACTIVE_JIT.set(self.0);
        }
    }
    let _restore = Restore(ACTIVE_JIT.replace(kind));
    f()
}

/// Whether the experiment is enabled.
pub fn enabled() -> bool {
    stats_path().is_some()
}

/// Where to write the comparison, if the experiment is enabled.
pub fn stats_path() -> Option<&'static OsString> {
    static PATH: LazyLock<Option<OsString>> = LazyLock::new(|| std::env::var_os(PATH_ENV));
    PATH.as_ref()
}

#[derive(Debug)]
struct ExecutionStats {
    calls: AtomicU64,
    time_ns: AtomicU64,
    /// BPF instructions executed, as reported by the VM.
    ///
    /// The same program takes wildly different paths between invocations, so this is the thing
    /// to normalize the time by.
    insns: AtomicU64,
    min_ns: AtomicU64,
    max_ns: AtomicU64,
}

impl Default for ExecutionStats {
    fn default() -> Self {
        Self {
            calls: AtomicU64::new(0),
            time_ns: AtomicU64::new(0),
            insns: AtomicU64::new(0),
            min_ns: AtomicU64::new(u64::MAX),
            max_ns: AtomicU64::new(0),
        }
    }
}

impl ExecutionStats {
    fn record(&self, time_ns: u64, insns: u64) {
        let ord = Ordering::Relaxed;
        self.calls.fetch_add(1, ord);
        self.time_ns.fetch_add(time_ns, ord);
        self.insns.fetch_add(insns, ord);
        self.min_ns.fetch_min(time_ns, ord);
        self.max_ns.fetch_max(time_ns, ord);
    }

    fn write_csv(&self, out: &mut String) {
        let ord = Ordering::Relaxed;
        let calls = self.calls.load(ord);
        let min_ns = if calls == 0 { 0 } else { self.min_ns.load(ord) };
        let _ = write!(
            out,
            "{calls},{},{},{min_ns},{}",
            self.time_ns.load(ord),
            self.insns.load(ord),
            self.max_ns.load(ord),
        );
    }
}

/// Per-program state of the comparison. Lives in [`ProgramStatistics`].
///
/// [`ProgramStatistics`]: crate::program_cache::metrics::ProgramStatistics
#[derive(Default)]
pub struct JitComparison {
    /// The same program as the cache entry's own, compiled with the parrot JIT instead.
    parrot_executable: Mutex<Option<Arc<ProgramExecutable>>>,

    sbpf_version: AtomicU64,
    text_bytes: AtomicU64,
    old_compile_ns: AtomicU64,
    old_code_bytes: AtomicU64,
    parrot_compile_ns: AtomicU64,
    parrot_code_bytes: AtomicU64,

    old: ExecutionStats,
    parrot: ExecutionStats,
}

impl std::fmt::Debug for JitComparison {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JitComparison").finish_non_exhaustive()
    }
}

impl JitComparison {
    /// Record how the entry's own executable got compiled with the old JIT and compile
    /// another copy of the program with the parrot JIT.
    pub fn compile(&self, elf_bytes: &[u8], old: &ProgramExecutable, old_compile_ns: u64) {
        if stats_path().is_none() {
            return;
        }
        // A program that the parrot JIT fails to handle is left out of the comparison rather than
        // making the whole program fail to load.
        if let Err(err) = self.try_compile(elf_bytes, old, old_compile_ns) {
            log::error!("parrot JIT comparison: failed to compile a program: {err}");
        }
    }

    fn try_compile(
        &self,
        elf_bytes: &[u8],
        old: &ProgramExecutable,
        old_compile_ns: u64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use solana_sbpf::verifier::RequisiteVerifier;
        let ord = Ordering::Relaxed;
        self.sbpf_version.store(old.get_sbpf_version() as u64, ord);
        self.text_bytes
            .store(old.get_text_bytes().1.len() as u64, ord);
        self.old_compile_ns.store(old_compile_ns, ord);
        self.old_code_bytes.store(
            old.get_compiled_program()
                .map_or(0, |p| p.machine_code_length() as u64),
            ord,
        );

        let parrot = Executable::load(elf_bytes, Arc::clone(old.get_loader()))?;
        parrot.verify::<RequisiteVerifier>()?;
        let start = solana_svm_measure::measure::Measure::start("parrot_compile_time");
        parrot.parrot_compile()?;
        self.parrot_compile_ns.store(start.end_as_ns(), ord);
        self.parrot_code_bytes.store(
            parrot
                .get_compiled_program()
                .map_or(0, |p| p.machine_code_length() as u64),
            ord,
        );
        *self.parrot_executable.lock().unwrap() = Some(Arc::new(parrot));
        Ok(())
    }

    /// The executable to run on this thread: the entry's own, or the parrot one if that is what
    /// [`with_jit`] asked for.
    pub fn select(&self) -> (JitKind, Option<Arc<ProgramExecutable>>) {
        match ACTIVE_JIT.get() {
            JitKind::Old => (JitKind::Old, None),
            // Programs that parrot could not compile still run, with the old JIT.
            JitKind::Parrot => match self.parrot_executable.lock().unwrap().clone() {
                Some(parrot) => (JitKind::Parrot, Some(parrot)),
                None => (JitKind::Old, None),
            },
        }
    }

    pub fn executed(&self, kind: JitKind, time_ns: u64, insns: u64) {
        match kind {
            JitKind::Old => self.old.record(time_ns, insns),
            JitKind::Parrot => self.parrot.record(time_ns, insns),
        }
    }

    /// The parrot executable is only useful while the entry is loaded.
    pub fn release(&self) {
        *self.parrot_executable.lock().unwrap() = None;
    }

    pub const CSV_HEADER: &'static str = "program,sbpf_version,text_bytes,\
        old_compile_ns,old_code_bytes,parrot_compile_ns,parrot_code_bytes,\
        old_calls,old_exec_ns,old_insns,old_min_ns,old_max_ns,\
        parrot_calls,parrot_exec_ns,parrot_insns,parrot_min_ns,parrot_max_ns\n";

    /// Append this program's row, unless the program has never been compiled in the experiment.
    pub fn write_csv_row(&self, program: &dyn std::fmt::Display, out: &mut String) {
        let ord = Ordering::Relaxed;
        if self.parrot_compile_ns.load(ord) == 0 {
            return;
        }
        let _ = write!(
            out,
            "{program},{},{},{},{},{},{},",
            self.sbpf_version.load(ord),
            self.text_bytes.load(ord),
            self.old_compile_ns.load(ord),
            self.old_code_bytes.load(ord),
            self.parrot_compile_ns.load(ord),
            self.parrot_code_bytes.load(ord),
        );
        self.old.write_csv(out);
        out.push(',');
        self.parrot.write_csv(out);
        out.push('\n');
    }
}
