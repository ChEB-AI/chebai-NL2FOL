import argparse
import pickle
from pathlib import Path

from nl_2_fol.inference.fol_reasoner import GavelFOLReasoner
from nl_2_fol.inference.learner.definition_model import (
    AdditionalDefinition,
    DefinitionLearningResults,
    FOLFormula,
)


def print_pickle_contents(
    pickle_file_path,
    class_name="all",
    show_system_prompt=False,
    show_conversation_history=False,
):
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
        if show_system_prompt:
            print("System prompt:")
            print(learned_def.prompts_history["system_prompt"])

        if show_conversation_history:
            print("Conversation history:")
            for c_his in learned_def.prompts_history["conversation_history"]:
                print(c_his)
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
        print(f"Used for CHEBI IDs: {add_def.used_for}")
        print("---" * 10)

    print(f"Number of learned definitions: {len(data.learned_definitions)}")
    print(f"Number of additional definitions: {len(data.additional_definitions)}")


def print_learned_definition_stats(pickle_file_path, metric_name="F1"):
    """Print score distribution statistics for learned definitions."""

    with open(pickle_file_path, "rb") as f:
        data: DefinitionLearningResults = pickle.load(f)

    requested_metric = metric_name.upper()
    scores, val_scores = [], []
    failed = 0

    for learned_def in data.learned_definitions.values():
        train_metric_value = getattr(learned_def.train_metrics, requested_metric)
        if not learned_def.learn_success:
            failed += 1
        scores.append(float(train_metric_value))

        if learned_def.val_metrics is not None:
            val_metric_value = getattr(learned_def.val_metrics, requested_metric)
            val_scores.append(float(val_metric_value))

    def print_stats(scores, total, dataset_name):
        perfect = sum(1 for score in scores if score == 1.0)
        gt_08 = sum(1 for score in scores if 0.8 <= score < 1.0)
        between_06_08 = sum(1 for score in scores if 0.6 <= score < 0.8)
        between_04_06 = sum(1 for score in scores if 0.4 <= score < 0.6)
        between_02_04 = sum(1 for score in scores if 0.2 <= score < 0.4)
        between_00_02 = sum(1 for score in scores if 0.0 < score < 0.2)
        equal_to_0 = sum(1 for score in scores if score == 0.0)

        print("------------ Score buckets for " + dataset_name, "-------------")
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
        if dataset_name == "training set":
            print(
                f"  score == 0.0: {equal_to_0} (out of which {failed} failed to learn) ({(equal_to_0 / total) * 100:.2f}%)"
            )
        elif dataset_name == "validation set":
            print(f"  score == 0.0: {equal_to_0} ({(equal_to_0 / total) * 100:.2f}%)")
        else:
            raise ValueError("Unexpected dataset name for stats printing.")
        print("-------------------------------------------------------")

    train_total = len(scores)
    print(f"Metric: {requested_metric}")
    print(f"Total definitions: {len(data.learned_definitions)}")
    print(f"Definitions failed to learn: {failed} during learning process.")

    if train_total == 0:
        print("No valid scores found. Nothing to summarize.")
        return
    print_stats(scores, train_total, "training set")

    if val_scores:
        val_total = len(val_scores)
        print_stats(val_scores, val_total, "validation set")


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


def _parse_used_for_ids(used_for: str) -> list[int]:
    if not used_for.strip():
        return []

    ids: list[int] = []
    for value in used_for.split(","):
        cleaned = value.strip()
        if not cleaned:
            continue
        ids.append(int(cleaned))
    return ids


def upsert_additional_definition_in_pickle(
    pickle_file_path: str,
    predicate_name: str,
    fol_definition: str,
    used_for_ids: list[int],
    learn_success=True,
    output_file_path=None,
    create_backup=True,
    replace_used_for=False,
):
    """Add or update an additional definition in a DefinitionLearningResults pickle."""

    if not predicate_name.strip():
        raise ValueError("predicate_name cannot be empty.")

    input_path = Path(pickle_file_path)
    original_pickle_bytes = input_path.read_bytes()
    data: DefinitionLearningResults = pickle.loads(original_pickle_bytes)

    reasoner = GavelFOLReasoner()
    pred_variables, parsed_formula = reasoner.get_tptp_fol_definition(fol_definition)

    predicate_name = predicate_name.strip()
    normalized_used_for = list(dict.fromkeys(int(x) for x in used_for_ids))

    if (
        predicate_name in data.additional_definitions
        and not replace_used_for
        and data.additional_definitions[predicate_name].used_for
    ):
        merged = [
            *data.additional_definitions[predicate_name].used_for,
            *normalized_used_for,
        ]
        normalized_used_for = list(dict.fromkeys(int(x) for x in merged))

    data.additional_definitions[predicate_name] = AdditionalDefinition(
        fol_formula=FOLFormula(
            formula=parsed_formula,
            pred_variables=pred_variables,
        ),
        used_for=normalized_used_for,
        learn_success=learn_success,
    )

    output_path = Path(output_file_path) if output_file_path else input_path

    if create_backup and output_path.resolve() == input_path.resolve():
        backup_path = input_path.with_suffix(input_path.suffix + ".bak")
        backup_path.write_bytes(original_pickle_bytes)
        print(f"Backup written to: {backup_path}")

    with open(output_path, "wb") as f:
        pickle.dump(data, f)

    print(f"Upserted additional definition for predicate: {predicate_name}")
    print(f"Updated pickle written to: {output_path}")
    return True


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Show or delete class definitions from learned_definitions pickle."
    )
    parser.add_argument(
        "--pickle-file",
        type=Path,
        required=True,
        help="Path to learned_definitions pickle file.",
    )

    subparsers = parser.add_subparsers(dest="command")

    show_parser = subparsers.add_parser("show", help="Show learned content.")
    show_parser.add_argument(
        "--class-name",
        default="all",
        help="Class name to filter (default: all).",
    )
    show_parser.add_argument(
        "--system-prompt",
        action="store_true",
        help="Show system prompt from prompts history if available (default: False).",
    )
    show_parser.add_argument(
        "--conversation-history",
        action="store_true",
        help="Show conversation history from prompts history if available (default: False).",
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

    upsert_additional_parser = subparsers.add_parser(
        "upsert-additional",
        help="Add or update an additional definition.",
    )
    upsert_additional_parser.add_argument(
        "predicate_name",
        help="Predicate name for the additional definition.",
    )
    upsert_additional_parser.add_argument(
        "fol_definition",
        help=(
            'FOL definition string (e.g. "twoPlusCarbonCompound <=> ?[X, Y]: '
            '(c(X) & c(Y) & has_bond_to(X, Y) & X != Y)").'
        ),
    )
    upsert_additional_parser.add_argument(
        "--used-for",
        default="",
        help="Comma-separated CHEBI IDs where this additional definition is used.",
    )
    upsert_additional_parser.add_argument(
        "--learn-success",
        action="store_true",
        default=False,
        help="Mark definition as successful (default: False).",
    )
    upsert_additional_parser.add_argument(
        "--replace-used-for",
        action="store_true",
        help="Replace existing used_for list instead of merging.",
    )
    upsert_additional_parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Optional output pickle path. Defaults to in-place update.",
    )
    upsert_additional_parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create a backup when editing in place.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    # python nl_2_fol/inference/utils/show_learned_content.py --pickle-file=<file_path> show --class-name=hopanoid
    # python nl_2_fol/inference/utils/show_learned_content.py --pickle-file=<file_path> delete hopanoid
    # python nl_2_fol/inference/utils/show_learned_content.py --pickle-file=<file_path> stats
    # python nl_2_fol/inference/utils/show_learned_content.py --pickle-file=<file_path> upsert-additional twoPlusCarbonCompound "twoPlusCarbonCompound <=> ?[X, Y]: (c(X) & c(Y) & has_bond_to(X, Y) & X != Y)" --used-for=12345,56645 --learn-success
    if args.command in (None, "show"):
        class_name = args.class_name if args.command == "show" else "all"
        show_system_prompt = args.system_prompt if args.command == "show" else False
        show_conversation_history = (
            args.conversation_history if args.command == "show" else False
        )
        print_pickle_contents(
            args.pickle_file,
            class_name=class_name,
            show_system_prompt=show_system_prompt,
            show_conversation_history=show_conversation_history,
        )
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
    elif args.command == "upsert-additional":
        used_for_ids = _parse_used_for_ids(args.used_for)
        upsert_additional_definition_in_pickle(
            args.pickle_file,
            predicate_name=args.predicate_name,
            fol_definition=args.fol_definition,
            used_for_ids=used_for_ids,
            learn_success=args.learn_success,
            output_file_path=args.output_file,
            create_backup=not args.no_backup,
            replace_used_for=args.replace_used_for,
        )
