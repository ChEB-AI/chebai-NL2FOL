import argparse
import ast
import pickle
from pathlib import Path

from nl_2_fol.inference.learner.definition_model import DefinitionLearningResults

DEFAULT_PICKLE_FILE = (
    Path(__file__).resolve().parent.parent
    / "learner"
    / "learned"
    / "claude-opus-4-6"
    / "learned_definitions.pkl"
)


def print_pickle_contents(pickle_file_path, class_name="all"):
    """Print the contents of a pickle file containing DefinitionLearningResults."""

    with open(pickle_file_path, "rb") as f:
        data: DefinitionLearningResults = pickle.load(f)

    for _, learned_def in data.learned_definitions.items():
        if class_name != "all" and learned_def.name != class_name:
            continue

        print(f"Learned definition for predicate: {learned_def.name}")
        print(
            f"Pred variables: {[str(var) for var in learned_def.learned_FOL.pred_variables]}"
        )
        print(f"Metrics: {learned_def.train_metrics}")
        print(f"Formula: {learned_def.learned_FOL.formula}")
        print(f"Learned success: {learned_def.learn_success}")
        for his in learned_def.prompts_history:
            print(his)

        content = learned_def.prompts_history["conversation_history"][-1]["content"]
        parsed_dict = ast.literal_eval(content)
        if "FOL_formula" in parsed_dict:
            print(f"Generated FOL formula: {parsed_dict['FOL_formula']}")
        print("---" * 10)

    for name, add_def in data.additional_definitions.items():
        if class_name != "all" and name != class_name:
            continue
        print(f"Additional definition for predicate: {name}")
        print(
            f"Pred variables: {[str(var) for var in add_def.fol_formula.pred_variables]}"
        )
        print(f"Formula: {add_def.fol_formula.formula}")
        print(f"Learned success: {add_def.learn_success}")
        print("---" * 10)

    print(f"Number of learned definitions: {len(data.learned_definitions)}")
    print(f"Number of additional definitions: {len(data.additional_definitions)}")


def print_learned_definition_stats(pickle_file_path, metric_name="F1"):
    """Print score distribution statistics for learned definitions."""

    with open(pickle_file_path, "rb") as f:
        data: DefinitionLearningResults = pickle.load(f)

    requested_metric = metric_name.upper()
    scores = []
    skipped = 0

    for learned_def in data.learned_definitions.values():
        metric_value = getattr(learned_def.train_metrics, requested_metric, None)
        if metric_value is None:
            skipped += 1
            continue

        try:
            scores.append(float(metric_value))
        except (TypeError, ValueError):
            skipped += 1

    total = len(scores)
    print(f"Metric: {requested_metric}")
    print(f"Total learned definitions: {len(data.learned_definitions)}")
    print(f"Definitions with valid {requested_metric}: {total}")
    print(f"Definitions skipped (missing/invalid metric): {skipped}")

    if total == 0:
        print("No valid scores found. Nothing to summarize.")
        return

    perfect = sum(1 for score in scores if score == 1.0)
    gt_08 = sum(1 for score in scores if 0.8 <= score < 1.0)
    between_06_08 = sum(1 for score in scores if 0.6 <= score < 0.8)
    between_04_06 = sum(1 for score in scores if 0.4 <= score < 0.6)
    between_02_04 = sum(1 for score in scores if 0.2 <= score < 0.4)
    between_00_02 = sum(1 for score in scores if 0.0 < score < 0.2)
    equal_to_0 = sum(1 for score in scores if score == 0.0)

    print("Score buckets:")
    print(f"  score == 1.0: {perfect} ({(perfect / total) * 100:.2f}%)")
    print(f"  0.8 <= score < 1.0: {gt_08} ({(gt_08 / total) * 100:.2f}%)")
    print(
        f"  0.6 <= score < 0.8: {between_06_08} ({(between_06_08 / total) * 100:.2f}%)"
    )
    print(
        f"  0.4 <= score < 0.6: {between_04_06} ({(between_04_06 / total) * 100:.2f}%)"
    )
    print(
        f"  0.2 <= score < 0.4: {between_02_04} ({(between_02_04 / total) * 100:.2f}%)"
    )
    print(
        f"  0.0 < score < 0.2: {between_00_02} ({(between_00_02 / total) * 100:.2f}%)"
    )
    print(f"  score == 0.0: {equal_to_0} ({(equal_to_0 / total) * 100:.2f}%)")


def delete_class_from_pickle(
    pickle_file_path, class_name, output_file_path=None, create_backup=True
):
    """Delete a class from learned/additional definitions and save updated pickle."""

    if class_name == "all":
        raise ValueError(
            "Refusing to delete all classes. Please pass a specific class."
        )

    input_path = Path(pickle_file_path)
    original_pickle_bytes = input_path.read_bytes()
    data: DefinitionLearningResults = pickle.loads(original_pickle_bytes)

    target = str(class_name).strip()

    learned_keys_to_delete = [
        key
        for key, learned_def in data.learned_definitions.items()
        if str(key) == target or learned_def.name == target
    ]
    additional_keys_to_delete = [
        name for name in data.additional_definitions if name == target
    ]

    for key in learned_keys_to_delete:
        del data.learned_definitions[key]

    for key in additional_keys_to_delete:
        del data.additional_definitions[key]

    total_deleted = len(learned_keys_to_delete) + len(additional_keys_to_delete)
    if total_deleted == 0:
        print(f"No class found for '{class_name}'. Nothing deleted.")
        return False

    output_path = Path(output_file_path) if output_file_path else input_path

    if create_backup and output_path.resolve() == input_path.resolve():
        backup_path = input_path.with_suffix(input_path.suffix + ".bak")
        backup_path.write_bytes(original_pickle_bytes)
        print(f"Backup written to: {backup_path}")

    with open(output_path, "wb") as f:
        pickle.dump(data, f)

    print(f"Deleted {len(learned_keys_to_delete)} learned definition(s).")
    print(f"Deleted {len(additional_keys_to_delete)} additional definition(s).")
    print(f"Updated pickle written to: {output_path}")
    return True


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Show or delete class definitions from learned_definitions pickle."
    )
    parser.add_argument(
        "--pickle-file",
        type=Path,
        default=DEFAULT_PICKLE_FILE,
        help="Path to learned_definitions pickle file.",
    )

    subparsers = parser.add_subparsers(dest="command")

    show_parser = subparsers.add_parser("show", help="Show learned content.")
    show_parser.add_argument(
        "--class-name",
        default="all",
        help="Class name to filter (default: all).",
    )

    delete_parser = subparsers.add_parser("delete", help="Delete a class.")
    delete_parser.add_argument(
        "class_name",
        help="Class name or learned definition key to delete.",
    )
    delete_parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Optional output pickle path. Defaults to in-place update.",
    )
    delete_parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create a backup when editing in place.",
    )

    stats_parser = subparsers.add_parser(
        "stats", help="Show score statistics for learned definitions."
    )
    stats_parser.add_argument(
        "--metric",
        default="F1",
        help="Metric name from train_metrics to summarize (default: F1).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.command in (None, "show"):
        class_name = args.class_name if args.command == "show" else "all"
        print_pickle_contents(args.pickle_file, class_name=class_name)
    elif args.command == "delete":
        delete_class_from_pickle(
            args.pickle_file,
            class_name=args.class_name,
            output_file_path=args.output_file,
            create_backup=not args.no_backup,
        )
    elif args.command == "stats":
        print_learned_definition_stats(
            args.pickle_file,
            metric_name=args.metric,
        )
