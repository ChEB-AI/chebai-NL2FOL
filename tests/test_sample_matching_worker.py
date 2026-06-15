"""Tests for check_if_definition_matches_samples subprocess timeout and error paths."""

import itertools
import queue
from unittest.mock import MagicMock, patch

import pytest
from rdkit import Chem

from nl_2_fol.inference.learner.custom_exceptions import (
    MissingPredicateException,
    StopProgramException,
)
from nl_2_fol.inference.learner.sample_matching_worker import (
    check_if_definition_matches_samples,
)
from nl_2_fol.inference.preprocessing.c3po_slim_data import ChemicalStructure


def _make_structure(name: str, smiles: str) -> ChemicalStructure:
    return ChemicalStructure(name=name, smiles=smiles, mol=Chem.MolFromSmiles(smiles))


def _make_mock_context(
    pos_queue: queue.Queue,
    neg_queue: queue.Queue,
    pos_proc: MagicMock,
    neg_proc: MagicMock,
) -> MagicMock:
    """Return a mock multiprocessing context wired to the given queues and processes."""
    mock_ctx = MagicMock()
    mock_ctx.Queue.side_effect = [pos_queue, neg_queue]
    mock_ctx.Process.side_effect = [pos_proc, neg_proc]
    return mock_ctx


def _make_error_event(
    message: str,
    traceback_text: str,
    exc_type: type[BaseException] | None = None,
    exc_args: tuple | None = None,
    exc_state: dict | None = None,
) -> tuple:
    return (
        "error",
        message,
        traceback_text,
        exc_type.__module__ if exc_type is not None else None,
        exc_type.__qualname__ if exc_type is not None else None,
        exc_args,
        exc_state,
    )


class TestCheckIfDefinitionMatchesSamples:
    """Tests for subprocess timeout and error-handling paths."""

    @pytest.fixture
    def common_inputs(self):
        """Minimal inputs for check_if_definition_matches_samples."""
        gavel = MagicMock()
        chemical_class = MagicMock()
        chemical_class.name = "TestClass"
        tptp_def = MagicMock()
        pos_sample = _make_structure("benzene", "c1ccccc1")
        neg_sample = _make_structure("ethanol", "CCO")
        return {
            "gavel": gavel,
            "chemical_class": chemical_class,
            "tptp_def": tptp_def,
            "pos_samples": {pos_sample},
            "neg_samples": {neg_sample},
        }

    @pytest.fixture
    def timeout_proc_mocks(self):
        """Process mocks for the timeout scenario.

        The workers stay alive long enough for the deadline check to fire, then
        remain alive so the timeout cleanup path can terminate them.
        """
        mock_pos_proc = MagicMock()
        mock_pos_proc.is_alive.return_value = True
        mock_neg_proc = MagicMock()
        mock_neg_proc.is_alive.return_value = True
        return mock_pos_proc, mock_neg_proc

    # ------------------------------------------------------------------ #
    # (a) Timeout scenarios                                                #
    # ------------------------------------------------------------------ #

    @patch("nl_2_fol.inference.learner.sample_matching_worker.multiprocessing")
    def test_timeout_with_only_positive_partial_results_raises_timeout_error(
        self, mock_mp, common_inputs, timeout_proc_mocks
    ):
        """When no negative samples are processed before timeout, TimeoutError is raised."""
        mock_pos_proc, mock_neg_proc = timeout_proc_mocks

        # pos queue holds one result emitted before the timeout
        pos_q: queue.Queue = queue.Queue()
        pos_q.put(("pos_checked", "c1ccccc1", "match"))  # benzene matched → TP

        neg_q: queue.Queue = queue.Queue()  # no neg results before timeout

        mock_mp.get_context.return_value = _make_mock_context(
            pos_q, neg_q, mock_pos_proc, mock_neg_proc
        )

        with patch(
            "nl_2_fol.inference.learner.sample_matching_worker.time.monotonic",
            side_effect=itertools.count(100),
        ):
            with pytest.raises(TimeoutError, match="No samples were processed"):
                check_if_definition_matches_samples(
                    gavel=common_inputs["gavel"],
                    sample_matching_timeout_seconds=1,
                    chemical_class=common_inputs["chemical_class"],
                    tptp_def=common_inputs["tptp_def"],
                    pos_samples=common_inputs["pos_samples"],
                    neg_samples=common_inputs["neg_samples"],
                )

        mock_pos_proc.terminate.assert_called_once()
        mock_neg_proc.terminate.assert_called_once()

    @patch("nl_2_fol.inference.learner.sample_matching_worker.multiprocessing")
    def test_timeout_with_partial_false_negatives(
        self, mock_mp, common_inputs, timeout_proc_mocks
    ):
        """Partial positive results still raise when no negative samples are processed."""
        mock_pos_proc, mock_neg_proc = timeout_proc_mocks

        pos_q: queue.Queue = queue.Queue()
        pos_q.put(("pos_checked", "c1ccccc1", "no_match"))  # benzene not matched → FN

        neg_q: queue.Queue = queue.Queue()

        mock_mp.get_context.return_value = _make_mock_context(
            pos_q, neg_q, mock_pos_proc, mock_neg_proc
        )

        with patch(
            "nl_2_fol.inference.learner.sample_matching_worker.time.monotonic",
            side_effect=itertools.count(100),
        ):
            with pytest.raises(TimeoutError, match="No samples were processed"):
                check_if_definition_matches_samples(
                    gavel=common_inputs["gavel"],
                    sample_matching_timeout_seconds=1,
                    chemical_class=common_inputs["chemical_class"],
                    tptp_def=common_inputs["tptp_def"],
                    pos_samples=common_inputs["pos_samples"],
                    neg_samples=common_inputs["neg_samples"],
                )

    @patch("nl_2_fol.inference.learner.sample_matching_worker.multiprocessing")
    def test_timeout_with_no_results_raises_timeout_error(
        self, mock_mp, common_inputs, timeout_proc_mocks
    ):
        """When no samples are processed before the timeout, TimeoutError is raised."""
        mock_pos_proc, mock_neg_proc = timeout_proc_mocks

        pos_q: queue.Queue = queue.Queue()  # empty — no results before timeout
        neg_q: queue.Queue = queue.Queue()  # empty

        mock_mp.get_context.return_value = _make_mock_context(
            pos_q, neg_q, mock_pos_proc, mock_neg_proc
        )

        with patch(
            "nl_2_fol.inference.learner.sample_matching_worker.time.monotonic",
            side_effect=itertools.count(100),
        ):
            with pytest.raises(TimeoutError, match="No samples were processed"):
                check_if_definition_matches_samples(
                    gavel=common_inputs["gavel"],
                    sample_matching_timeout_seconds=1,
                    chemical_class=common_inputs["chemical_class"],
                    tptp_def=common_inputs["tptp_def"],
                    pos_samples=common_inputs["pos_samples"],
                    neg_samples=common_inputs["neg_samples"],
                )

        mock_pos_proc.terminate.assert_called_once()
        mock_neg_proc.terminate.assert_called_once()

    # ------------------------------------------------------------------ #
    # (b) Worker error-propagation scenarios                               #
    # ------------------------------------------------------------------ #

    @patch("nl_2_fol.inference.learner.sample_matching_worker.multiprocessing")
    def test_pos_worker_error_is_propagated(self, mock_mp, common_inputs):
        """An exception raised inside the positive-sample worker propagates to the caller."""
        pos_q: queue.Queue = queue.Queue()
        pos_q.put(
            _make_error_event(
                "model check failed",
                "Traceback (most recent call last)...",
            )
        )

        neg_q: queue.Queue = queue.Queue()
        neg_q.put(("done",))

        mock_pos_proc = MagicMock()
        mock_pos_proc.is_alive.return_value = False  # already exited
        mock_pos_proc.exitcode = 0

        mock_neg_proc = MagicMock()
        mock_neg_proc.is_alive.return_value = False
        mock_neg_proc.exitcode = 0

        mock_mp.get_context.return_value = _make_mock_context(
            pos_q, neg_q, mock_pos_proc, mock_neg_proc
        )

        with pytest.raises(
            Exception, match="Positive sample matching subprocess failed"
        ):
            check_if_definition_matches_samples(
                gavel=common_inputs["gavel"],
                sample_matching_timeout_seconds=10,
                chemical_class=common_inputs["chemical_class"],
                tptp_def=common_inputs["tptp_def"],
                pos_samples=common_inputs["pos_samples"],
                neg_samples=common_inputs["neg_samples"],
            )

    @patch("nl_2_fol.inference.learner.sample_matching_worker.multiprocessing")
    def test_pos_worker_error_message_contains_details(self, mock_mp, common_inputs):
        """The propagated exception message includes the original error text."""
        error_text = "rdkit segmentation fault"
        traceback_text = "Traceback ..."

        pos_q: queue.Queue = queue.Queue()
        pos_q.put(_make_error_event(error_text, traceback_text))

        neg_q: queue.Queue = queue.Queue()
        neg_q.put(("done",))

        mock_pos_proc = MagicMock()
        mock_pos_proc.is_alive.return_value = False
        mock_pos_proc.exitcode = 0

        mock_neg_proc = MagicMock()
        mock_neg_proc.is_alive.return_value = False
        mock_neg_proc.exitcode = 0

        mock_mp.get_context.return_value = _make_mock_context(
            pos_q, neg_q, mock_pos_proc, mock_neg_proc
        )

        with pytest.raises(Exception) as exc_info:
            check_if_definition_matches_samples(
                gavel=common_inputs["gavel"],
                sample_matching_timeout_seconds=10,
                chemical_class=common_inputs["chemical_class"],
                tptp_def=common_inputs["tptp_def"],
                pos_samples=common_inputs["pos_samples"],
                neg_samples=common_inputs["neg_samples"],
            )

        assert error_text in str(exc_info.value)
        assert traceback_text not in str(exc_info.value)

    @patch("nl_2_fol.inference.learner.sample_matching_worker.multiprocessing")
    def test_pos_worker_error_preserves_original_exception_type(
        self, mock_mp, common_inputs
    ):
        """Worker metadata allows parent to raise MissingPredicateException directly."""
        original_exc = MissingPredicateException({"UnknownPredicate"})

        pos_q: queue.Queue = queue.Queue()
        pos_q.put(
            _make_error_event(
                str(original_exc),
                "Traceback ...",
                type(original_exc),
                original_exc.args,
                dict(original_exc.__dict__),
            )
        )

        neg_q: queue.Queue = queue.Queue()
        neg_q.put(("done",))

        mock_pos_proc = MagicMock()
        mock_pos_proc.is_alive.return_value = False
        mock_pos_proc.exitcode = 0

        mock_neg_proc = MagicMock()
        mock_neg_proc.is_alive.return_value = False
        mock_neg_proc.exitcode = 0

        mock_mp.get_context.return_value = _make_mock_context(
            pos_q, neg_q, mock_pos_proc, mock_neg_proc
        )

        with pytest.raises(MissingPredicateException) as exc_info:
            check_if_definition_matches_samples(
                gavel=common_inputs["gavel"],
                sample_matching_timeout_seconds=10,
                chemical_class=common_inputs["chemical_class"],
                tptp_def=common_inputs["tptp_def"],
                pos_samples=common_inputs["pos_samples"],
                neg_samples=common_inputs["neg_samples"],
            )

        assert exc_info.value.missing_predicates == {"UnknownPredicate"}

    @patch("nl_2_fol.inference.learner.sample_matching_worker.multiprocessing")
    def test_neg_worker_error_is_propagated(self, mock_mp, common_inputs):
        """An exception raised inside the negative-sample worker propagates to the caller."""
        pos_q: queue.Queue = queue.Queue()
        pos_q.put(("done",))

        neg_q: queue.Queue = queue.Queue()
        neg_q.put(_make_error_event("neg model check failed", "Traceback ..."))

        mock_pos_proc = MagicMock()
        mock_pos_proc.is_alive.return_value = False
        mock_pos_proc.exitcode = 0

        mock_neg_proc = MagicMock()
        mock_neg_proc.is_alive.return_value = False
        mock_neg_proc.exitcode = 0

        mock_mp.get_context.return_value = _make_mock_context(
            pos_q, neg_q, mock_pos_proc, mock_neg_proc
        )

        with pytest.raises(
            Exception, match="Negative sample matching subprocess failed"
        ):
            check_if_definition_matches_samples(
                gavel=common_inputs["gavel"],
                sample_matching_timeout_seconds=10,
                chemical_class=common_inputs["chemical_class"],
                tptp_def=common_inputs["tptp_def"],
                pos_samples=common_inputs["pos_samples"],
                neg_samples=common_inputs["neg_samples"],
            )

    @patch("nl_2_fol.inference.learner.sample_matching_worker.multiprocessing")
    def test_pos_worker_error_takes_precedence_over_partial_results(
        self, mock_mp, common_inputs
    ):
        """Error event is raised even when some pos results arrived before the error."""
        pos_q: queue.Queue = queue.Queue()
        pos_q.put(("pos_checked", "c1ccccc1", True))  # partial result before error
        pos_q.put(_make_error_event("mid-batch failure", "Traceback ..."))

        neg_q: queue.Queue = queue.Queue()
        neg_q.put(("done",))

        mock_pos_proc = MagicMock()
        mock_pos_proc.is_alive.return_value = False
        mock_pos_proc.exitcode = 0

        mock_neg_proc = MagicMock()
        mock_neg_proc.is_alive.return_value = False
        mock_neg_proc.exitcode = 0

        mock_mp.get_context.return_value = _make_mock_context(
            pos_q, neg_q, mock_pos_proc, mock_neg_proc
        )

        with pytest.raises(
            Exception, match="Positive sample matching subprocess failed"
        ):
            check_if_definition_matches_samples(
                gavel=common_inputs["gavel"],
                sample_matching_timeout_seconds=10,
                chemical_class=common_inputs["chemical_class"],
                tptp_def=common_inputs["tptp_def"],
                pos_samples=common_inputs["pos_samples"],
                neg_samples=common_inputs["neg_samples"],
            )

    # ------------------------------------------------------------------ #
    # Unexpected (non-zero exit code) worker crash                         #
    # ------------------------------------------------------------------ #

    @patch("nl_2_fol.inference.learner.sample_matching_worker.multiprocessing")
    def test_unexpected_pos_worker_exit_raises_stop_program_exception(
        self, mock_mp, common_inputs
    ):
        """A non-zero exit code from the pos worker (no error event) raises StopProgramException."""
        pos_q: queue.Queue = (
            queue.Queue()
        )  # no events — crashed before sending anything

        neg_q: queue.Queue = queue.Queue()
        neg_q.put(("done",))

        mock_pos_proc = MagicMock()
        mock_pos_proc.is_alive.return_value = False
        mock_pos_proc.exitcode = 1  # non-zero → unexpected crash

        mock_neg_proc = MagicMock()
        mock_neg_proc.is_alive.return_value = False
        mock_neg_proc.exitcode = 0

        mock_mp.get_context.return_value = _make_mock_context(
            pos_q, neg_q, mock_pos_proc, mock_neg_proc
        )

        with pytest.raises(
            StopProgramException,
            match="Positive sample matching subprocess exited unexpectedly",
        ):
            check_if_definition_matches_samples(
                gavel=common_inputs["gavel"],
                sample_matching_timeout_seconds=10,
                chemical_class=common_inputs["chemical_class"],
                tptp_def=common_inputs["tptp_def"],
                pos_samples=common_inputs["pos_samples"],
                neg_samples=common_inputs["neg_samples"],
            )

    @patch("nl_2_fol.inference.learner.sample_matching_worker.multiprocessing")
    def test_unexpected_neg_worker_exit_raises_stop_program_exception(
        self, mock_mp, common_inputs
    ):
        """A non-zero exit code from the neg worker (no error event) raises StopProgramException."""
        pos_q: queue.Queue = queue.Queue()
        pos_q.put(("done",))

        neg_q: queue.Queue = queue.Queue()  # no events — crashed

        mock_pos_proc = MagicMock()
        mock_pos_proc.is_alive.return_value = False
        mock_pos_proc.exitcode = 0

        mock_neg_proc = MagicMock()
        mock_neg_proc.is_alive.return_value = False
        mock_neg_proc.exitcode = 2  # non-zero → unexpected crash

        mock_mp.get_context.return_value = _make_mock_context(
            pos_q, neg_q, mock_pos_proc, mock_neg_proc
        )

        with pytest.raises(
            StopProgramException,
            match="Negative sample matching subprocess exited unexpectedly",
        ):
            check_if_definition_matches_samples(
                gavel=common_inputs["gavel"],
                sample_matching_timeout_seconds=10,
                chemical_class=common_inputs["chemical_class"],
                tptp_def=common_inputs["tptp_def"],
                pos_samples=common_inputs["pos_samples"],
                neg_samples=common_inputs["neg_samples"],
            )
