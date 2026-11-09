//! Executes each transaction batch under both the old and the parrot sBPF JIT and compares the
//! outcomes, see [`solana_program_runtime::program_cache::jit_comparison`].
//!
//! Besides logging mismatches, every transaction is also recorded to
//! `<AGAVE_SBPF_JIT_COMPARE_PATH>.transactions.csv` along with the time it took to process under
//! either JIT, and the details of every mismatch go to `<AGAVE_SBPF_JIT_COMPARE_PATH>.mismatches.log`
//! in addition to the regular log.

use {
    crate::{
        transaction_execution_result::ExecutedTransaction,
        transaction_processing_result::{ProcessedTransaction, TransactionProcessingResult},
    },
    log::error,
    solana_clock::Slot,
    solana_program_runtime::program_cache::jit_comparison::stats_path,
    std::{
        cell::RefCell,
        fmt::{Display, Write as _},
        fs::File,
        io::Write as _,
        panic::{AssertUnwindSafe, catch_unwind, resume_unwind},
        sync::{
            LazyLock, Mutex,
            mpsc::{Sender, channel},
        },
    },
};

type Job = Box<dyn FnOnce() + Send + 'static>;

/// A thread that outlives the batches it runs jobs for.
///
/// Programs get their stack, heap and call frames from a pool that lives in thread-local storage.
/// A thread spawned for each batch would start every one of them with an empty pool and cold
/// caches, which is a severe penalty on the side of the comparison that runs on it.
struct Helper(Sender<Job>);

impl Helper {
    fn spawn() -> Self {
        let (sender, receiver) = channel::<Job>();
        std::thread::Builder::new()
            .name("solJitCmpHelper".into())
            // Programs and syscalls run on the native stack.
            .stack_size(16 * 1024 * 1024)
            .spawn(move || {
                // Ends when the thread that owns the helper does.
                while let Ok(job) = receiver.recv() {
                    job();
                }
            })
            .expect("failed to spawn the helper thread for the JIT comparison");
        Self(sender)
    }
}

thread_local! {
    /// Every thread that executes batches gets a helper of its own.
    static HELPER: Helper = Helper::spawn();
}

/// Run `on_helper` on this thread's helper thread and `on_current` on this thread at the same
/// time, and return when both are done.
pub(crate) fn join_with_helper<A: Send, B>(
    on_helper: impl FnOnce() -> A + Send,
    on_current: impl FnOnce() -> B,
) -> (A, B) {
    let (result_sender, result_receiver) = channel();
    let job: Box<dyn FnOnce() + Send + '_> = Box::new(move || {
        let _ = result_sender.send(catch_unwind(AssertUnwindSafe(on_helper)));
    });
    // SAFETY: the job borrows from the caller's stack. It is guaranteed to be done with those
    // borrows by the time this function returns: it either reports its result below, or the
    // helper thread is gone along with the job (dropped unrun, or run to the end by the time it
    // is closed), in which case `recv` fails. A panic in `on_current` is held back until then.
    let job: Job = unsafe { std::mem::transmute(job) };
    HELPER.with(|helper| helper.0.send(job).expect("the JIT comparison helper died"));
    let current = catch_unwind(AssertUnwindSafe(on_current));
    let helped = result_receiver
        .recv()
        .expect("the JIT comparison helper died without a result");
    match (helped, current) {
        (Ok(helped), Ok(current)) => (helped, current),
        (Err(panic), _) | (_, Err(panic)) => resume_unwind(panic),
    }
}

/// How long a single transaction of a batch took to process.
#[derive(Clone, Copy, Default)]
pub(crate) struct TxTiming {
    /// Validation, loading, executing and everything else done for the transaction.
    pub total_ns: u64,
    /// Just the part of that which loads missing programs and executes the transaction.
    pub execution_ns: u64,
}

thread_local! {
    static TX_TIMINGS: RefCell<Vec<TxTiming>> = const { RefCell::new(Vec::new()) };
}

/// Record how long the next transaction of the batch being processed on this thread took.
pub(crate) fn record_tx_timing(timing: TxTiming) {
    TX_TIMINGS.with_borrow_mut(|timings| timings.push(timing));
}

/// Take the timings of the transactions of the batch processed on this thread, in order.
///
/// Transactions past the point where a batch got aborted have none.
pub(crate) fn take_tx_timings() -> Vec<TxTiming> {
    TX_TIMINGS.take()
}

static TRANSACTION_LOG: LazyLock<Option<Mutex<File>>> = LazyLock::new(|| {
    let mut path = stats_path()?.clone();
    path.push(".transactions.csv");
    let result = File::create(&path).and_then(|mut file| {
        writeln!(
            file,
            "signature,slot,outcome,executed_units,matched,\
             old_total_ns,old_execution_ns,parrot_total_ns,parrot_execution_ns"
        )?;
        Ok(file)
    });
    match result {
        Ok(file) => Some(Mutex::new(file)),
        Err(e) => {
            error!("JIT comparison: cannot create {path:?}: {e}");
            None
        }
    }
});

static MISMATCH_LOG: LazyLock<Option<Mutex<File>>> = LazyLock::new(|| {
    let mut path = stats_path()?.clone();
    path.push(".mismatches.log");
    match File::create(&path) {
        Ok(file) => Some(Mutex::new(file)),
        Err(e) => {
            error!("JIT comparison: cannot create {path:?}: {e}");
            None
        }
    }
});

/// A transaction as processed under both JITs.
pub(crate) struct Compared<'a> {
    pub signature: &'a dyn Display,
    pub old: &'a TransactionProcessingResult,
    pub old_timing: Option<TxTiming>,
    pub parrot: &'a TransactionProcessingResult,
    pub parrot_timing: Option<TxTiming>,
}

/// Compare the transactions of a batch as processed under both JITs, log the mismatches and
/// record the transactions.
pub(crate) fn compare_batch<'a>(slot: Slot, batch: impl IntoIterator<Item = Compared<'a>>) {
    // Make sure that the files exist even if there is nothing to put in them.
    let _ = (&*TRANSACTION_LOG, &*MISMATCH_LOG);
    let mut rows = String::new();
    for tx in batch {
        let mut differences = String::new();
        compare_results(tx.old, tx.parrot, &mut differences);
        let matched = differences.is_empty();
        if !matched {
            let report = format!(
                "JIT comparison mismatch in transaction {} in slot {slot} (old JIT vs parrot \
                 JIT):{differences}\n",
                tx.signature,
            );
            error!("{}", report.trim_end());
            if let Some(log) = &*MISMATCH_LOG {
                if let Err(e) = log.lock().unwrap().write_all(report.as_bytes()) {
                    error!("JIT comparison: cannot write the mismatch log: {e}");
                }
            }
        }
        let (outcome, executed_units) = match tx.old {
            Ok(ProcessedTransaction::Executed(executed)) => {
                ("executed", executed.execution_details.executed_units)
            }
            Ok(ProcessedTransaction::FeesOnly(_)) => ("fees-only", 0),
            Ok(ProcessedTransaction::NoOp(_)) => ("no-op", 0),
            Err(_) => ("error", 0),
        };
        let _ = write!(
            rows,
            "{},{slot},{outcome},{executed_units},{}",
            tx.signature, matched as u8
        );
        for timing in [tx.old_timing, tx.parrot_timing] {
            match timing {
                Some(t) => {
                    let _ = write!(rows, ",{},{}", t.total_ns, t.execution_ns);
                }
                None => rows.push_str(",,"),
            }
        }
        rows.push('\n');
    }
    if let Some(log) = &*TRANSACTION_LOG {
        if let Err(e) = log.lock().unwrap().write_all(rows.as_bytes()) {
            error!("JIT comparison: cannot write the transaction log: {e}");
        }
    }
}

fn compare_results(
    old: &TransactionProcessingResult,
    parrot: &TransactionProcessingResult,
    out: &mut String,
) {
    match (old, parrot) {
        (Ok(ProcessedTransaction::Executed(old)), Ok(ProcessedTransaction::Executed(parrot))) => {
            compare_executed(old, parrot, out)
        }
        (Ok(ProcessedTransaction::FeesOnly(old)), Ok(ProcessedTransaction::FeesOnly(parrot))) => {
            if old != parrot {
                let _ = write!(out, "\n  fees-only outcome: {old:?} vs {parrot:?}");
            }
        }
        (Ok(ProcessedTransaction::NoOp(old)), Ok(ProcessedTransaction::NoOp(parrot))) => {
            if old != parrot {
                let _ = write!(out, "\n  no-op outcome: {old:?} vs {parrot:?}");
            }
        }
        (Err(old), Err(parrot)) => {
            if old != parrot {
                let _ = write!(out, "\n  error: {old:?} vs {parrot:?}");
            }
        }
        (old, parrot) => {
            let kind = |r: &TransactionProcessingResult| match r {
                Ok(ProcessedTransaction::Executed(_)) => "executed",
                Ok(ProcessedTransaction::FeesOnly(_)) => "fees-only",
                Ok(ProcessedTransaction::NoOp(_)) => "no-op",
                Err(_) => "error",
            };
            let _ = write!(out, "\n  outcome: {} vs {}", kind(old), kind(parrot));
        }
    }
}

fn compare_executed(old: &ExecutedTransaction, parrot: &ExecutedTransaction, out: &mut String) {
    let (o, p) = (&old.execution_details, &parrot.execution_details);
    if o.status != p.status {
        let _ = write!(out, "\n  status: {:?} vs {:?}", o.status, p.status);
    }
    if o.executed_units != p.executed_units {
        let _ = write!(
            out,
            "\n  executed units: {} vs {}",
            o.executed_units, p.executed_units
        );
    }
    if o.return_data != p.return_data {
        let _ = write!(
            out,
            "\n  return data: {:?} vs {:?}",
            o.return_data, p.return_data
        );
    }
    if o.log_messages != p.log_messages {
        let _ = write!(
            out,
            "\n  log messages: {:?} vs {:?}",
            o.log_messages, p.log_messages
        );
    }
    if o.inner_instructions != p.inner_instructions {
        let _ = write!(out, "\n  inner instructions differ");
    }
    if o.accounts_deltas != p.accounts_deltas {
        let _ = write!(
            out,
            "\n  accounts deltas: {:?} vs {:?}",
            o.accounts_deltas, p.accounts_deltas
        );
    }
    let (o, p) = (&old.loaded_transaction, &parrot.loaded_transaction);
    if o.accounts.len() != p.accounts.len() {
        let _ = write!(
            out,
            "\n  number of accounts: {} vs {}",
            o.accounts.len(),
            p.accounts.len()
        );
    }
    for ((old_key, old_account), (parrot_key, parrot_account)) in o.accounts.iter().zip(&p.accounts)
    {
        if old_key != parrot_key || old_account != parrot_account {
            let _ = write!(out, "\n  account {old_key} differs");
        }
    }
    if o.touched_flags != p.touched_flags {
        let _ = write!(out, "\n  touched accounts differ");
    }
    if o.rollback_accounts != p.rollback_accounts {
        let _ = write!(out, "\n  rollback accounts differ");
    }
    if o.fee_details != p.fee_details {
        let _ = write!(
            out,
            "\n  fee details: {:?} vs {:?}",
            o.fee_details, p.fee_details
        );
    }
}
