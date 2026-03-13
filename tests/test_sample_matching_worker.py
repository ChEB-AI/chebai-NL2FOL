"""Tests for check_if_definition_matches_samples subprocess timeout and error paths."""

import queue
from unittest.mock import MagicMock, patch

import pytest
from rdkit import Chem

from nl_2_fol.inference.learner.custom_exceptions import StopProgramException
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

        is_alive() call sequence when a timeout occurs before workers finish:
          - pos: while-check(T) → outer-timeout-check(T) → inner-pos-check(T) → post-terminate(F)
          - neg: inner-neg-check(T) → post-terminate(F)
        """
        mock_pos_proc = MagicMock()
        mock_pos_proc.is_alive.side_effect = [True, True, True, False]
        mock_neg_proc = MagicMock()
        mock_neg_proc.is_alive.side_effect = [True, False]
        return mock_pos_proc, mock_neg_proc

    # ------------------------------------------------------------------ #
    # (a) Timeout scenarios                                                #
    # ------------------------------------------------------------------ #

    @patch("nl_2_fol.inference.learner.sample_matching_worker.multiprocessing")
    def test_timeout_returns_partial_results(
        self, mock_mp, common_inputs, timeout_proc_mocks
    ):
        """When workers are terminated due to timeout, already-queued results are returned."""
        mock_pos_proc, mock_neg_proc = timeout_proc_mocks

        # pos queue holds one result emitted before the timeout
        pos_q: queue.Queue = queue.Queue()
        pos_q.put(("pos_checked", "c1ccccc1", True))  # benzene matched → TP

        neg_q: queue.Queue = queue.Queue()  # no neg results before timeout

        mock_mp.get_context.return_value = _make_mock_context(
            pos_q, neg_q, mock_pos_proc, mock_neg_proc
        )

        # A negative timeout ensures the deadline has already passed when the
        # loop first checks remaining time, forcing an immediate timeout.
        result = check_if_definition_matches_samples(
            gavel=common_inputs["gavel"],
            sample_matching_timeout_seconds=-1,
            chemical_class=common_inputs["chemical_class"],
            tptp_def=common_inputs["tptp_def"],
            pos_samples=common_inputs["pos_samples"],
            neg_samples=common_inputs["neg_samples"],
        )

        unmatched_pos, matched_neg, processed_pos, processed_neg = result

        # benzene was matched (True) so it is not a false-negative
        assert len(unmatched_pos) == 0
        # no neg results were processed before the timeout
        assert len(matched_neg) == 0
        assert len(processed_neg) == 0
        # benzene was processed
        assert len(processed_pos) == 1
        assert next(iter(processed_pos)).smiles == "c1ccccc1"

        mock_pos_proc.terminate.assert_called_once()
        mock_neg_proc.terminate.assert_called_once()

    @patch("nl_2_fol.inference.learner.sample_matching_worker.multiprocessing")
    def test_timeout_with_partial_false_negatives(
        self, mock_mp, common_inputs, timeout_proc_mocks
    ):
        """Partial results that include unmatched positive samples are reported correctly."""
        mock_pos_proc, mock_neg_proc = timeout_proc_mocks

        pos_q: queue.Queue = queue.Queue()
        pos_q.put(("pos_checked", "c1ccccc1", False))  # benzene not matched → FN

        neg_q: queue.Queue = queue.Queue()

        mock_mp.get_context.return_value = _make_mock_context(
            pos_q, neg_q, mock_pos_proc, mock_neg_proc
        )

        unmatched_pos, matched_neg, processed_pos, processed_neg = (
            check_if_definition_matches_samples(
                gavel=common_inputs["gavel"],
                sample_matching_timeout_seconds=-1,
                chemical_class=common_inputs["chemical_class"],
                tptp_def=common_inputs["tptp_def"],
                pos_samples=common_inputs["pos_samples"],
                neg_samples=common_inputs["neg_samples"],
            )
        )

        assert "c1ccccc1" in unmatched_pos
        assert len(processed_pos) == 1

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

        with pytest.raises(TimeoutError, match="No samples were processed"):
            check_if_definition_matches_samples(
                gavel=common_inputs["gavel"],
                sample_matching_timeout_seconds=-1,
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
            ("error", "model check failed", "Traceback (most recent call last)...")
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
        pos_q.put(("error", error_text, traceback_text))

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
    def test_neg_worker_error_is_propagated(self, mock_mp, common_inputs):
        """An exception raised inside the negative-sample worker propagates to the caller."""
        pos_q: queue.Queue = queue.Queue()
        pos_q.put(("done",))

        neg_q: queue.Queue = queue.Queue()
        neg_q.put(("error", "neg model check failed", "Traceback ..."))

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
        pos_q.put(("error", "mid-batch failure", "Traceback ..."))

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
