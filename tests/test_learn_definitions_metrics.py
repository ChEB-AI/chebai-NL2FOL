from nl_2_fol.inference.learner.learn_definitions import LearnDefinitions


def test_get_metrics_handles_no_positive_predictions() -> None:
    learner = LearnDefinitions.__new__(LearnDefinitions)

    # TP=0 and FP=0 => PPV denominator would be zero without safeguards.
    metrics = learner._get_metrics(
        unmatched_pos_samples={"p1", "p2"},
        matched_neg_samples=set(),
        pos_samples={"p1", "p2"},
        neg_samples={"n1", "n2"},
    )

    assert metrics.TP == 0
    assert metrics.FP == 0
    assert metrics.FN == 2
    assert metrics.TN == 2
    assert metrics.F1 == 0.0
    assert metrics.PPV == 0.0
    assert metrics.NPV == 0.5


def test_get_metrics_handles_zero_f1_denominator() -> None:
    learner = LearnDefinitions.__new__(LearnDefinitions)

    # F1 denominator is zero when TP=FP=FN=0.
    metrics = learner._get_metrics(
        unmatched_pos_samples=set(),
        matched_neg_samples=set(),
        pos_samples=set(),
        neg_samples={"n1"},
    )

    assert metrics.F1 == 0.0
    assert metrics.PPV == 0.0
    assert metrics.NPV == 1.0
