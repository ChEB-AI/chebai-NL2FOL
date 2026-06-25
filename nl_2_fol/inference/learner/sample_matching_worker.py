import logging
import multiprocessing
import os
import queue
import sys
import time
import traceback
from importlib import import_module

import pandas as pd
import tqdm
from chemlog.fol_classification.model_checking import ModelCheckerOutcome
from gavel.logic.logic import QuantifiedFormula

from nl_2_fol.inference import PRINT_TRACES
from nl_2_fol.inference.fol_reasoner import (
    ASPModelChecker,
    ChemlogModelChecker,
    FOLDefinition,
)
from nl_2_fol.inference.learner.custom_exceptions import StopProgramException
from nl_2_fol.inference.preprocessing import c3po_slim_data as dm

assert sys.version_info >= (3, 11), "Python 3.11 or newer is required."

logger = logging.getLogger(__name__)

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


def _raise_worker_error(
    chemical_class: dm.ChemicalClass, worker_label: str, worker_error: WorkerError
) -> None:
    error_message, error_trace, exc_module, exc_qualname, exc_args, exc_state = (
        worker_error
    )
    if PRINT_TRACES:
        logger.error("%s\n%s", error_message, error_trace)

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
                f"[{chemical_class.id}:{chemical_class.name}:{worker_label}] Original worker traceback (from subprocess):\n"
                + error_trace
            )
        raise rebuilt_exception

    fallback_exception = Exception(
        f"{chemical_class.id}:{chemical_class.name}:{worker_label} sample matching subprocess failed with error: {error_message}"
    )
    if error_trace:
        fallback_exception.add_note(
            f"[{chemical_class.id}:{chemical_class.name}:{worker_label}] Original worker traceback (from subprocess):\n"
            + error_trace
        )
    raise fallback_exception


def check_if_definition_matches_samples_clingo(
    model_checker: ASPModelChecker,
    sample_matching_timeout_seconds: int | None,
    chemical_class: dm.ChemicalClass,
    asp_def: str,
    pos_samples: set[dm.ChemicalStructure],
    neg_samples: set[dm.ChemicalStructure],
    temp_additional_defs: dict[str, FOLDefinition] | None = None,
    split: str = "train",
) -> tuple[dict[str, set[dm.SMILES_STRING]], dict[str, set[dm.ChemicalStructure]]]:
    # normally, the index would be the actual ChEBI ID. here, a dummy index is used instead since ChemicalStructure does not store the ChEBI ID
    indexed_samples = {
        i: s for i, s in enumerate(list(pos_samples) + list(neg_samples))
    }
    molecules_df = pd.DataFrame(
        {"mol": [s.mol for s in indexed_samples.values()]},
        index=list(indexed_samples.keys()),
    )

    matched_ids = model_checker.do_molecules_match_asp_definition(
        molecules_df,
        asp_def,
        temp_additional_defs=temp_additional_defs,
        timeout=sample_matching_timeout_seconds,
    )

    if matched_ids is None:
        raise Exception(
            f"[{chemical_class.id}:{chemical_class.name}] Model checker returned None for definition: {asp_def}"
        )
    print(f"Matched IDs: (total {len(matched_ids)})")
    matched_pos_samples = {
        i for i in matched_ids if i in indexed_samples and i < len(pos_samples)
    }
    matched_neg_samples = {
        i for i in matched_ids if i in indexed_samples and i >= len(pos_samples)
    }
    unmatched_pos_samples = {
        i
        for i in indexed_samples
        if i < len(pos_samples) and i not in matched_pos_samples
    }
    unmatched_neg_samples = {
        i
        for i in indexed_samples
        if i >= len(pos_samples) and i not in matched_neg_samples
    }

    print(
        f"#TPs: {len(matched_pos_samples)}, #FNs: {len(unmatched_pos_samples)}, #TNs: {len(unmatched_neg_samples)}, #FPs: {len(matched_neg_samples)}"
    )

    return {
        "matched_pos_samples": set(
            indexed_samples[s].smiles for s in matched_pos_samples
        ),  # TPs
        "unmatched_neg_samples": set(
            indexed_samples[s].smiles for s in unmatched_neg_samples
        ),  # TNs
        "unmatched_pos_samples": set(
            indexed_samples[s].smiles for s in unmatched_pos_samples
        ),  # FNs
        "matched_neg_samples": set(
            indexed_samples[s].smiles for s in matched_neg_samples
        ),  # FPs
        "inferred_match_pos": set(),
        "inferred_match_neg": set(),
        "inferred_no_match_pos": set(),
        "inferred_no_match_neg": set(),
        "timeout_pos": set(),
        "timeout_neg": set(),
        "error_pos": set(),
        "error_neg": set(),
        "unknown_pos": set(),
        "unknown_neg": set(),
    }, {
        "processed_pos_samples": pos_samples,
        "processed_neg_samples": neg_samples,
    }


def check_if_definition_matches_samples(
    gavel: ChemlogModelChecker,
    sample_matching_timeout_seconds: int | None,
    chemical_class: dm.ChemicalClass,
    tptp_def: QuantifiedFormula,
    pos_samples: set[dm.ChemicalStructure],
    neg_samples: set[dm.ChemicalStructure],
    temp_additional_defs: dict[str, FOLDefinition] | None = None,
    split: str = "train",
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
    logger.info(
        "[%s:%s:sample-matching] Starting validation for %s | pos=%d neg=%d timeout=%s",
        chemical_class.id,
        chemical_class.name,
        chemical_class.name,
        len(pos_samples_list),
        len(neg_samples_list),
        sample_matching_timeout_seconds
        if sample_matching_timeout_seconds is not None
        else "none",
    )
    ctx = multiprocessing.get_context("fork")
    pos_result_queue = ctx.Queue()
    neg_result_queue = ctx.Queue()
    pos_worker = ctx.Process(
        target=check_positive_samples_worker,
        args=(
            chemical_class,
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
            chemical_class,
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
                    logger.warning(
                        "[%s:%s] Positive worker reported timeout for SMILES: %s",
                        chemical_class.id,
                        chemical_class.name,
                        smiles,
                    )
                elif outcome == "error":
                    error_pos.add(smiles)
                else:  # unknown
                    unknown_pos.add(smiles)
            elif event_type == "done":
                pos_worker_completed = True
                logger.info(
                    "[%s:%s] Positive worker reported done.",
                    chemical_class.id,
                    chemical_class.name,
                )
            elif event_type == "error":
                pos_worker_error = _parse_error_event(event)
                error_message = pos_worker_error[0]
                logger.error(
                    "[%s:%s] Positive worker reported error: %s",
                    chemical_class.id,
                    chemical_class.name,
                    error_message,
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
                    logger.warning(
                        "[%s:%s] Negative worker reported timeout for SMILES: %s",
                        chemical_class.id,
                        chemical_class.name,
                        smiles,
                    )
                    timeout_neg.add(smiles)
                elif outcome == "error":
                    error_neg.add(smiles)
                else:  # unknown
                    unknown_neg.add(smiles)
            elif event_type == "done":
                neg_worker_completed = True
                logger.info(
                    "[%s:%s] Negative worker reported done.",
                    chemical_class.id,
                    chemical_class.name,
                )
            elif event_type == "error":
                neg_worker_error = _parse_error_event(event)
                error_message = neg_worker_error[0]
                logger.error(
                    "[%s:%s] Negative worker reported error: %s",
                    chemical_class.id,
                    chemical_class.name,
                    error_message,
                )

    pos_worker.start()
    neg_worker.start()
    logger.info(
        "[%s:%s] Spawned workers (pos_pid=%s, neg_pid=%s).",
        chemical_class.id,
        chemical_class.name,
        pos_worker.pid,
        neg_worker.pid,
    )

    deadline = (
        None
        if sample_matching_timeout_seconds is None
        or sample_matching_timeout_seconds <= 0
        else time.monotonic() + sample_matching_timeout_seconds
    )
    timed_out = False
    max_timeout_threshold_reached = False
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

        if len(timeout_pos) > MAX_TIMEOUTS or len(timeout_neg) > MAX_TIMEOUTS:
            # If a worker accumulates too many timeouts, terminate that worker to avoid runaway work.
            # See: https://github.com/sfluegel05/chemlog-peptides/issues/16
            logger.warning(
                "[%s:%s] Worker exceeded %d timeouts (pos=%d, neg=%d, max=%d); "
                "terminating positive and negative workers to prevent runaway processing.",
                chemical_class.id,
                chemical_class.name,
                MAX_TIMEOUTS,
                len(timeout_pos),
                len(timeout_neg),
                max(len(timeout_pos), len(timeout_neg)),
            )
            if pos_worker.is_alive():
                # terminate positive worker
                pos_worker.terminate()
                pos_worker.join(timeout=2)
                if pos_worker.is_alive():
                    pos_worker.kill()
                    pos_worker.join()
                pos_worker_completed = True

            # also terminate negative worker to avoid orphaned work
            if neg_worker.is_alive():
                neg_worker.terminate()
                neg_worker.join(timeout=2)
                if neg_worker.is_alive():
                    neg_worker.kill()
                    neg_worker.join()
                neg_worker_completed = True

            drain_pos_queue()
            drain_neg_queue()
            # Set all matched samples to unknown since we can't trust the results
            # anymore after too many timeouts
            matched_pos_samples: set[dm.SMILES_STRING] = set()  # TPs
            unmatched_neg_samples: set[dm.SMILES_STRING] = set()  # TNs
            unmatched_pos_samples: set[dm.SMILES_STRING] = set()  # FNs
            matched_neg_samples: set[dm.SMILES_STRING] = set()  # FPs

            # Move processed samples or remaining samples to timeouts as the FOL exceeds
            # Max timeouts threshold, which indicates that the definition is likely
            # too complex to validate within reasonable time.
            # timeout_pos: set[dm.SMILES_STRING] = {
            #     chemical.smiles for chemical in pos_samples
            # }
            # timeout_neg: set[dm.SMILES_STRING] = {
            #     chemical.smiles for chemical in neg_samples
            # }
            max_timeout_threshold_reached = True

        now = time.monotonic()
        if deadline is not None and now - last_progress_report >= 5:
            logger.info(
                "[%s:%s] Progress pos=%d/%d neg=%d/%d remaining=%ds",
                chemical_class.id,
                chemical_class.name,
                len(processed_pos_smiles),
                len(pos_samples_list),
                len(processed_neg_smiles),
                len(neg_samples_list),
                max(0, int(deadline - now)),
            )
            last_progress_report = now

    if timed_out and (pos_worker.is_alive() or neg_worker.is_alive()):
        logger.warning(
            "[%s:%s] Sample matching subprocesses exceeded %s seconds and was terminated.",
            chemical_class.id,
            chemical_class.name,
            sample_matching_timeout_seconds,
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
        _raise_worker_error(chemical_class, "Positive", pos_worker_error)

    if neg_worker_error is not None:
        _raise_worker_error(chemical_class, "Negative", neg_worker_error)

    if (
        not pos_worker_completed
        and not timed_out
        and pos_worker.exitcode not in (0, None)
        and not max_timeout_threshold_reached
    ):
        raise StopProgramException(
            f"[{chemical_class.id}:{chemical_class.name}] Positive sample matching subprocess exited unexpectedly with "
            f"exit code: {pos_worker.exitcode}."
        )
    if (
        not neg_worker_completed
        and not timed_out
        and neg_worker.exitcode not in (0, None)
        and not max_timeout_threshold_reached
    ):
        raise StopProgramException(
            f"[{chemical_class.id}:{chemical_class.name}] Negative sample matching subprocess exited unexpectedly with "
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

    if split == "train" and (
        len(processed_pos_samples) == 0 or len(processed_neg_samples) == 0
    ):
        raise TimeoutError(
            f"[{chemical_class.id}:{chemical_class.name}] No samples were processed within "
            f"{sample_matching_timeout_seconds} seconds while validating definition "
            f"of {chemical_class.name}. Try reducing formula complexity."
        )

    if timed_out:
        logger.warning(
            "[%s:%s] Sample matching timed out; returning partial results.",
            chemical_class.id,
            chemical_class.name,
        )

    logger.info(
        "[%s:%s] Unmatched positive (FN) samples: %d/%d processed (total available: %d)",
        chemical_class.id,
        chemical_class.name,
        len(unmatched_pos_samples),
        len(processed_pos_samples),
        len(pos_samples),
    )
    logger.info(
        "[%s:%s] Matched negative (FP) samples: %d/%d processed (total available: %d)",
        chemical_class.id,
        chemical_class.name,
        len(matched_neg_samples),
        len(processed_neg_samples),
        len(neg_samples),
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
        "processed_pos_samples": processed_pos_samples,
        "processed_neg_samples": processed_neg_samples,
    }


def _check_samples_worker(
    chemical_class: dm.ChemicalClass,
    result_queue,
    gavel: ChemlogModelChecker,
    tptp_def: QuantifiedFormula,
    samples: list[dm.ChemicalStructure],
    event_type: str,
    temp_additional_defs: dict[str, FOLDefinition] | None = None,
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
        logger.info(
            "[%s:%s] Worker PID=%s starting %d %s samples.",
            chemical_class.id,
            chemical_class.name,
            os.getpid(),
            total_samples,
            label,
        )
        with tqdm.tqdm(
            samples,
            total=total_samples,
            desc=f"[ChEBI:{chemical_class.id}:{chemical_class.name}:{label}:Pid:{os.getpid()}]",
        ) as progress:
            for chemical in progress:
                matched = is_matched(chemical)
                # Record all outcome types, not just True/False
                result_queue.put((event_type, chemical.smiles, matched))

        result_queue.put(("done",))
        logger.info(
            "[%s:%s:%s] Worker PID=%s completed.",
            chemical_class.id,
            chemical_class.name,
            label,
            os.getpid(),
        )
    except Exception as e:
        logger.exception(
            "[%s:%s:%s] Worker PID=%s failed: %s",
            chemical_class.id,
            chemical_class.name,
            label,
            os.getpid(),
            e,
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
    chemical_class: dm.ChemicalClass,
    result_queue,
    gavel: ChemlogModelChecker,
    tptp_def: QuantifiedFormula,
    pos_samples: list[dm.ChemicalStructure],
    temp_additional_defs: dict[str, FOLDefinition] | None = None,
) -> None:
    _check_samples_worker(
        chemical_class,
        result_queue,
        gavel,
        tptp_def,
        pos_samples,
        "pos_checked",
        temp_additional_defs,
    )


def check_negative_samples_worker(
    chemical_class: dm.ChemicalClass,
    result_queue,
    gavel: ChemlogModelChecker,
    tptp_def: QuantifiedFormula,
    neg_samples: list[dm.ChemicalStructure],
    temp_additional_defs: dict[str, FOLDefinition] | None = None,
) -> None:
    _check_samples_worker(
        chemical_class,
        result_queue,
        gavel,
        tptp_def,
        neg_samples,
        "neg_checked",
        temp_additional_defs,
    )
