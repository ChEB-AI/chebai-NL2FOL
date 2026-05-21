import multiprocessing
import os
import queue
import sys
import time
import traceback
from importlib import import_module

from chemlog.fol_classification.model_checking import ModelCheckerOutcome
from gavel.logic import logic
from gavel.logic.logic import QuantifiedFormula

from nl_2_fol.inference import PRINT_TRACES
from nl_2_fol.inference.fol_reasoner import GavelFOLReasoner
from nl_2_fol.inference.learner.custom_exceptions import StopProgramException
from nl_2_fol.inference.preprocessing import c3po_slim_data as dm

assert sys.version_info >= (3, 11), "Python 3.11 or newer is required."

__all__ = ["check_if_definition_matches_samples"]

WorkerError = tuple[
    str,
    str,
    str | None,
    str | None,
    tuple | None,
    dict | None,
]

MAX_TIMEOUTS = 50


def _parse_error_event(event: tuple) -> WorkerError:
    """Parse the structured worker error event."""
    (
        _,
        error_message,
        error_trace,
        exc_module,
        exc_qualname,
        exc_args,
        exc_state,
    ) = event
    normalized_args = exc_args if isinstance(exc_args, tuple) else None
    normalized_state = exc_state if isinstance(exc_state, dict) else None
    return (
        error_message,
        error_trace,
        exc_module,
        exc_qualname,
        normalized_args,
        normalized_state,
    )


def _resolve_exception_class(
    module_name: str | None,
    qualname: str | None,
) -> type[BaseException] | None:
    if not module_name or not qualname:
        return None

    try:
        module = import_module(module_name)
        resolved_obj = module
        for attr in qualname.split("."):
            resolved_obj = getattr(resolved_obj, attr)
        if isinstance(resolved_obj, type) and issubclass(resolved_obj, BaseException):
            return resolved_obj
    except Exception:
        return None
    return None


def _raise_worker_error(worker_label: str, worker_error: WorkerError) -> None:
    error_message, error_trace, exc_module, exc_qualname, exc_args, exc_state = (
        worker_error
    )
    if PRINT_TRACES:
        print(error_message, error_trace, sep="\n")

    exception_cls = _resolve_exception_class(exc_module, exc_qualname)
    if exception_cls is not None:
        try:
            # Avoid custom __init__ signatures (e.g., MissingPredicateException expects
            # a set) by restoring BaseException args/state directly.
            rebuilt_exception = exception_cls.__new__(exception_cls)
            BaseException.__init__(
                rebuilt_exception,
                *(exc_args if exc_args is not None else (error_message,)),
            )
            if exc_state:
                rebuilt_exception.__dict__.update(exc_state)
        except Exception:
            rebuilt_exception = exception_cls(error_message)

        if error_trace:
            rebuilt_exception.add_note(
                "Original worker traceback (from subprocess):\n" + error_trace
            )
        raise rebuilt_exception

    fallback_exception = Exception(
        f"{worker_label} sample matching subprocess failed with error: {error_message}"
    )
    if error_trace:
        fallback_exception.add_note(
            "Original worker traceback (from subprocess):\n" + error_trace
        )
    raise fallback_exception


def check_if_definition_matches_samples(
    gavel: GavelFOLReasoner,
    sample_matching_timeout_seconds: int | None,
    chemical_class: dm.ChemicalClass,
    tptp_def: QuantifiedFormula,
    pos_samples: set[dm.ChemicalStructure],
    neg_samples: set[dm.ChemicalStructure],
    temp_additional_defs: dict[
        str, tuple[list[logic.Variable], logic.QuantifiedFormula]
    ]
    | None = None,
) -> tuple[dict[str, set[dm.SMILES_STRING]], dict[str, set[dm.ChemicalStructure]]]:
    # Track definite outcomes
    matched_pos_samples: set[dm.SMILES_STRING] = set()  # TPs
    unmatched_neg_samples: set[dm.SMILES_STRING] = set()  # TNs
    unmatched_pos_samples: set[dm.SMILES_STRING] = set()  # FNs
    matched_neg_samples: set[dm.SMILES_STRING] = set()  # FPs

    # Track inferred outcomes
    inferred_match_pos: set[dm.SMILES_STRING] = set()
    inferred_match_neg: set[dm.SMILES_STRING] = set()
    inferred_no_match_pos: set[dm.SMILES_STRING] = set()
    inferred_no_match_neg: set[dm.SMILES_STRING] = set()

    # Track error/timeout/unknown outcomes
    timeout_pos: set[dm.SMILES_STRING] = set()
    timeout_neg: set[dm.SMILES_STRING] = set()
    error_pos: set[dm.SMILES_STRING] = set()
    error_neg: set[dm.SMILES_STRING] = set()
    unknown_pos: set[dm.SMILES_STRING] = set()
    unknown_neg: set[dm.SMILES_STRING] = set()

    processed_pos_smiles: set[dm.SMILES_STRING] = set()
    processed_neg_smiles: set[dm.SMILES_STRING] = set()
    pos_worker_error: WorkerError | None = None
    neg_worker_error: WorkerError | None = None
    pos_worker_completed = False
    neg_worker_completed = False

    pos_samples_list = list(pos_samples)
    neg_samples_list = list(neg_samples)
    print(
        "\n[sample-matching] Starting validation for "
        f"{chemical_class.name} | "
        f"pos={len(pos_samples_list)} neg={len(neg_samples_list)} "
        f"timeout={sample_matching_timeout_seconds if sample_matching_timeout_seconds is not None else 'none'}",
        flush=True,
    )
    ctx = multiprocessing.get_context("fork")
    pos_result_queue = ctx.Queue()
    neg_result_queue = ctx.Queue()
    pos_worker = ctx.Process(
        target=check_positive_samples_worker,
        args=(
            pos_result_queue,
            gavel,
            tptp_def,
            pos_samples_list,
            temp_additional_defs,
        ),
    )
    neg_worker = ctx.Process(
        target=check_negative_samples_worker,
        args=(
            neg_result_queue,
            gavel,
            tptp_def,
            neg_samples_list,
            temp_additional_defs,
        ),
    )

    def drain_pos_queue() -> None:
        nonlocal pos_worker_error
        nonlocal pos_worker_completed
        while True:
            try:
                event = pos_result_queue.get_nowait()
            except queue.Empty:
                break

            event_type = event[0]
            if event_type == "pos_checked":
                _, smiles, outcome = event
                processed_pos_smiles.add(smiles)
                if outcome == "match":
                    matched_pos_samples.add(smiles)  # TP
                elif outcome == "no_match":
                    unmatched_pos_samples.add(smiles)  # FN
                elif outcome == "inferred_match":
                    inferred_match_pos.add(smiles)
                elif outcome == "inferred_no_match":
                    inferred_no_match_pos.add(smiles)
                elif outcome == "timeout":
                    timeout_pos.add(smiles)
                elif outcome == "error":
                    error_pos.add(smiles)
                else:  # unknown
                    unknown_pos.add(smiles)
            elif event_type == "done":
                pos_worker_completed = True
                print(
                    f"[sample-matching for {chemical_class.name}] Positive worker reported done.",
                    flush=True,
                )
            elif event_type == "error":
                pos_worker_error = _parse_error_event(event)
                error_message = pos_worker_error[0]
                print(
                    f"[sample-matching for {chemical_class.name}] Positive worker reported error: "
                    f"{error_message}",
                    flush=True,
                )

    def drain_neg_queue() -> None:
        nonlocal neg_worker_error
        nonlocal neg_worker_completed
        while True:
            try:
                event = neg_result_queue.get_nowait()
            except queue.Empty:
                break

            event_type = event[0]
            if event_type == "neg_checked":
                _, smiles, outcome = event
                processed_neg_smiles.add(smiles)
                if outcome == "match":
                    matched_neg_samples.add(smiles)  # FP
                elif outcome == "no_match":
                    unmatched_neg_samples.add(smiles)  # TN
                elif outcome == "inferred_match":
                    inferred_match_neg.add(smiles)
                elif outcome == "inferred_no_match":
                    inferred_no_match_neg.add(smiles)
                elif outcome == "timeout":
                    timeout_neg.add(smiles)
                elif outcome == "error":
                    error_neg.add(smiles)
                else:  # unknown
                    unknown_neg.add(smiles)
            elif event_type == "done":
                neg_worker_completed = True
                print(
                    f"[sample-matching for {chemical_class.name}] Negative worker reported done.",
                    flush=True,
                )
            elif event_type == "error":
                neg_worker_error = _parse_error_event(event)
                error_message = neg_worker_error[0]
                print(
                    f"[sample-matching for {chemical_class.name}] Negative worker reported error: "
                    f"{error_message}",
                    flush=True,
                )

    pos_worker.start()
    neg_worker.start()
    print(
        f"[sample-matching for {chemical_class.name}] Spawned workers "
        f"(pos_pid={pos_worker.pid}, neg_pid={neg_worker.pid}).",
        flush=True,
    )

    deadline = (
        None
        if sample_matching_timeout_seconds is None
        or sample_matching_timeout_seconds <= 0
        else time.monotonic() + sample_matching_timeout_seconds
    )
    timed_out = False
    last_progress_report = time.monotonic()

    while pos_worker.is_alive() or neg_worker.is_alive():
        if deadline is None:
            join_timeout = 0.5
        else:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            join_timeout = min(0.5, remaining)
        if pos_worker.is_alive():
            pos_worker.join(timeout=join_timeout)
        if neg_worker.is_alive():
            neg_worker.join(timeout=join_timeout)
        drain_pos_queue()
        drain_neg_queue()

        # If a worker accumulates too many timeouts, terminate that worker to avoid runaway work.
        if len(timeout_pos) > MAX_TIMEOUTS and pos_worker.is_alive():
            print(
                f"[sample-matching for {chemical_class.name}] Positive worker exceeded {MAX_TIMEOUTS} timeouts (count={len(timeout_pos)}); terminating positive and sibling negative worker.",
                flush=True,
            )
            # terminate positive worker
            pos_worker.terminate()
            pos_worker.join(timeout=2)
            if pos_worker.is_alive():
                pos_worker.kill()
                pos_worker.join()
            pos_worker_completed = True

            # also terminate negative worker to avoid orphaned work
            if neg_worker.is_alive():
                print(
                    f"[sample-matching for {chemical_class.name}] Terminating negative worker due to positive worker timeout cutoff.",
                    flush=True,
                )
                neg_worker.terminate()
                neg_worker.join(timeout=2)
                if neg_worker.is_alive():
                    neg_worker.kill()
                    neg_worker.join()
                neg_worker_completed = True

        if len(timeout_neg) > MAX_TIMEOUTS and neg_worker.is_alive():
            print(
                f"[sample-matching for {chemical_class.name}] Negative worker exceeded {MAX_TIMEOUTS} timeouts (count={len(timeout_neg)}); terminating negative and sibling positive worker.",
                flush=True,
            )
            # terminate negative worker
            neg_worker.terminate()
            neg_worker.join(timeout=2)
            if neg_worker.is_alive():
                neg_worker.kill()
                neg_worker.join()
            neg_worker_completed = True

            # also terminate positive worker to avoid orphaned work
            if pos_worker.is_alive():
                print(
                    f"[sample-matching for {chemical_class.name}] Terminating positive worker due to negative worker timeout cutoff.",
                    flush=True,
                )
                pos_worker.terminate()
                pos_worker.join(timeout=2)
                if pos_worker.is_alive():
                    pos_worker.kill()
                    pos_worker.join()
                pos_worker_completed = True

        now = time.monotonic()
        if deadline is not None and now - last_progress_report >= 5:
            print(
                f"[sample-matching for {chemical_class.name}] Progress "
                f"pos={len(processed_pos_smiles)}/{len(pos_samples_list)} "
                f"neg={len(processed_neg_smiles)}/{len(neg_samples_list)} "
                f"remaining={max(0, int(deadline - now))}s",
                flush=True,
            )
            last_progress_report = now

    if timed_out and (pos_worker.is_alive() or neg_worker.is_alive()):
        print(
            f"[sample-matching for {chemical_class.name}] Sample matching subprocesses exceeded "
            f"{sample_matching_timeout_seconds} seconds and was terminated."
        )
        if pos_worker.is_alive():
            pos_worker.terminate()
            pos_worker.join(timeout=2)
            if pos_worker.is_alive():
                pos_worker.kill()
                pos_worker.join()
        if neg_worker.is_alive():
            neg_worker.terminate()
            neg_worker.join(timeout=2)
            if neg_worker.is_alive():
                neg_worker.kill()
                neg_worker.join()

    drain_pos_queue()
    drain_neg_queue()

    if pos_worker_error is not None:
        _raise_worker_error("Positive", pos_worker_error)

    if neg_worker_error is not None:
        _raise_worker_error("Negative", neg_worker_error)

    if (
        not pos_worker_completed
        and not timed_out
        and pos_worker.exitcode not in (0, None)
    ):
        raise StopProgramException(
            "Positive sample matching subprocess exited unexpectedly with "
            f"exit code: {pos_worker.exitcode}."
        )
    if (
        not neg_worker_completed
        and not timed_out
        and neg_worker.exitcode not in (0, None)
    ):
        raise StopProgramException(
            "Negative sample matching subprocess exited unexpectedly with "
            f"exit code: {neg_worker.exitcode}."
        )
    processed_pos_samples = {
        chemical
        for chemical in pos_samples
        if chemical.smiles in processed_pos_smiles
        and chemical.smiles in (matched_pos_samples | unmatched_pos_samples)
    }
    processed_neg_samples = {
        chemical
        for chemical in neg_samples
        if chemical.smiles in processed_neg_smiles
        and chemical.smiles in (matched_neg_samples | unmatched_neg_samples)
    }

    if len(processed_pos_samples) == 0 and len(processed_neg_samples) == 0:
        raise TimeoutError(
            "No samples were processed within "
            f"{sample_matching_timeout_seconds} seconds while validating definition "
            f"of {chemical_class.name}. Try reducing formula complexity."
        )

    if timed_out:
        print(
            f"[sample-matching for {chemical_class.name}] Sample matching timed out; returning partial results."
        )

    print(
        f"[sample-matching for {chemical_class.name}] Unmatched positive (FN) samples: {len(unmatched_pos_samples)}/"
        f"{len(processed_pos_samples)} processed "
        f"(total available: {len(pos_samples)})"
    )
    print(
        f"[sample-matching for {chemical_class.name}] Matched negative (FP) samples: {len(matched_neg_samples)}/"
        f"{len(processed_neg_samples)} processed "
        f"(total available: {len(neg_samples)})"
    )
    # Return all outcome tracking data with the standard TP/FP/TN/FN outcomes
    return {
        "matched_pos_samples": matched_pos_samples,  # TPs
        "unmatched_neg_samples": unmatched_neg_samples,  # TNs
        "unmatched_pos_samples": unmatched_pos_samples,  # FNs
        "matched_neg_samples": matched_neg_samples,  # FPs
        "inferred_match_pos": inferred_match_pos,
        "inferred_match_neg": inferred_match_neg,
        "inferred_no_match_pos": inferred_no_match_pos,
        "inferred_no_match_neg": inferred_no_match_neg,
        "timeout_pos": timeout_pos,
        "timeout_neg": timeout_neg,
        "error_pos": error_pos,
        "error_neg": error_neg,
        "unknown_pos": unknown_pos,
        "unknown_neg": unknown_neg,
    }, {
        "processed_pos_samples": processed_pos_samples,  # All processed positives
        "processed_neg_samples": processed_neg_samples,  # All processed negatives
    }


def _check_samples_worker(
    result_queue,
    gavel: GavelFOLReasoner,
    tptp_def: QuantifiedFormula,
    samples: list[dm.ChemicalStructure],
    event_type: str,
    temp_additional_defs: dict[
        str, tuple[list[logic.Variable], logic.QuantifiedFormula]
    ]
    | None = None,
) -> None:
    label = "positive" if event_type == "pos_checked" else "negative"
    total_samples = len(samples)

    def is_matched(chemical: dm.ChemicalStructure) -> str:
        """Returns categorized outcome: 'match', 'no_match', 'inferred_match', 'inferred_no_match', 'timeout', 'error', or 'unknown'."""
        outcome = gavel.does_mol_match_tptp_definition(
            chemical.mol,
            tptp_def,
            temp_additional_defs=temp_additional_defs,
        )
        if outcome == ModelCheckerOutcome.MODEL_FOUND:
            return "match"
        elif outcome == ModelCheckerOutcome.NO_MODEL:
            return "no_match"
        elif outcome == ModelCheckerOutcome.MODEL_FOUND_INFERRED:
            return "inferred_match"
        elif outcome == ModelCheckerOutcome.NO_MODEL_INFERRED:
            return "inferred_no_match"
        elif outcome == ModelCheckerOutcome.TIMEOUT:
            return "timeout"
        elif outcome == ModelCheckerOutcome.ERROR:
            return "error"
        else:  # ModelCheckerOutcome.UNKNOWN or any other value
            return "unknown"

    try:
        print(
            f"[sample-matching:{label}] Worker PID={os.getpid()} starting "
            f"{total_samples} samples.",
            flush=True,
        )
        for idx, chemical in enumerate(samples, start=1):
            matched = is_matched(chemical)
            # Record all outcome types, not just True/False
            result_queue.put((event_type, chemical.smiles, matched))
            if idx == 1 or idx % 25 == 0 or idx == total_samples:
                print(
                    f"[sample-matching:{label}] "
                    f"processed {idx}/{total_samples} "
                    f"(smiles={chemical.smiles}, matched={matched})",
                    flush=True,
                )

        result_queue.put(("done",))
        print(
            f"[sample-matching:{label}] Worker PID={os.getpid()} completed.",
            flush=True,
        )
    except Exception as e:
        print(
            f"[sample-matching:{label}] Worker PID={os.getpid()} failed: {e}",
            flush=True,
        )
        error_trace = traceback.format_exc()
        exception_type = type(e)
        serialized_error_event = (
            "error",
            str(e),
            error_trace,
            exception_type.__module__,
            exception_type.__qualname__,
            e.args,
            dict(e.__dict__),
        )
        try:
            result_queue.put(serialized_error_event)
        except Exception:
            fallback_error_event = (
                "error",
                str(e),
                error_trace,
                None,
                None,
                e.args,
                None,
            )
            result_queue.put(fallback_error_event)


def check_positive_samples_worker(
    result_queue,
    gavel: GavelFOLReasoner,
    tptp_def: QuantifiedFormula,
    pos_samples: list[dm.ChemicalStructure],
    temp_additional_defs: dict[
        str, tuple[list[logic.Variable], logic.QuantifiedFormula]
    ]
    | None = None,
) -> None:
    _check_samples_worker(
        result_queue,
        gavel,
        tptp_def,
        pos_samples,
        "pos_checked",
        temp_additional_defs,
    )


def check_negative_samples_worker(
    result_queue,
    gavel: GavelFOLReasoner,
    tptp_def: QuantifiedFormula,
    neg_samples: list[dm.ChemicalStructure],
    temp_additional_defs: dict[
        str, tuple[list[logic.Variable], logic.QuantifiedFormula]
    ]
    | None = None,
) -> None:
    _check_samples_worker(
        result_queue,
        gavel,
        tptp_def,
        neg_samples,
        "neg_checked",
        temp_additional_defs,
    )
