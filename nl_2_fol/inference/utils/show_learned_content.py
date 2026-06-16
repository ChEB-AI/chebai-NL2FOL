import argparse
import pickle
from pathlib import Path

from nl_2_fol.inference.fol_reasoner import GavelFOLReasoner
from nl_2_fol.inference.learner.definition_model import (
    AdditionalDefinition,
    DefinitionLearningResults,
    FOLFormula,
)
from nl_2_fol.inference.utils import print_metrics
from nl_2_fol.inference.utils.to_camel_case import to_camel_case


def print_pickle_contents(
    pickle_file_path,
    class_name="all",
    show_system_prompt=False,
    show_conversation_history=False,
):
    """Print the contents of a pickle file containing DefinitionLearningResults."""

    with open(pickle_file_path, "rb") as f:
        data: DefinitionLearningResults = pickle.load(f)

    FOUND_CLASS = False

    def _print_def(class_name):
        for chebi_id, learned_def in data.learned_definitions.items():
            if class_name != "all" and learned_def.name != class_name:
                continue

            nonlocal FOUND_CLASS
            FOUND_CLASS = True
            print(
                f"Learned definition for predicate: {learned_def.name} (CHEBI ID: {chebi_id})"
            )
            print(
                f"Pred variables: {[str(var) for var in learned_def.learned_FOL.pred_variables]}"
            )
            print("Training Metrics:", print_metrics(learned_def.train_metrics))
            if learned_def.val_metrics is not None:
                print("Validation Metrics:", print_metrics(learned_def.val_metrics))
            print(f"Formula: {learned_def.learned_FOL.formula}")
            print(f"Learned success: {learned_def.learn_success}")

            if (
                hasattr(learned_def, "additional_defs_used")
                and learned_def.additional_defs_used
            ):
                print("Additional definitions used:")
                for name, (
                    def_vars,
                    add_def,
                ) in learned_def.additional_defs_used.items():
                    print(
                        f"  {name} with variables {[str(var) for var in def_vars]} and formula: {add_def}"
                    )
            if show_system_prompt:
                print("System prompt:")
                print(learned_def.prompts_history["system_prompt"])

            if show_conversation_history:
                print("Conversation history:")
                conv_his = learned_def.prompts_history["conversation_history"]
                if conv_his is None or len(conv_his) == 0:
                    print("\tNo conversation history available.")
                else:
                    for c_his in conv_his:
                        print(c_his)
                        print()
            print("---" * 10)

        for name, add_def in data.additional_definitions.items():
            if class_name != "all" and name != class_name:
                continue
            FOUND_CLASS = True
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

    _print_def(class_name)
    if FOUND_CLASS is False:
        camel_cased_class_name = to_camel_case(class_name)
        print(
            f"Class name `{class_name}` was not found directly in the learned definitions file. "
            f"Trying camel-cased variant `{camel_cased_class_name}`."
        )
        _print_def(camel_cased_class_name)


def print_learned_definition_stats(pickle_file_path, metric_name="F1"):
    """Print score distribution statistics for learned definitions."""

    with open(pickle_file_path, "rb") as f:
        data: DefinitionLearningResults = pickle.load(f)

    requested_metric = metric_name.upper()
    scores, val_scores = [], []
    train_metric_records = []
    val_metric_records = []
    failed = 0

    for learned_def in data.learned_definitions.values():
        train_metrics = learned_def.train_metrics
        train_metric_value = getattr(train_metrics, requested_metric)
        if not learned_def.learn_success:
            failed += 1
        else:
            if train_metric_value > 0.0:
                # Exclude case D i.e. ignore failed classes and classes with 0 f1 scores
                # for micro and macro f1 calculation
                train_metric_records.append(train_metrics)

        scores.append(float(train_metric_value))

        if learned_def.val_metrics is not None:
            val_metrics = learned_def.val_metrics
            val_metric_value = getattr(val_metrics, requested_metric)
            val_scores.append(float(val_metric_value))
            val_metric_records.append(val_metrics)

    def calculate_micro_macro_metrics(metric_records):
        if not metric_records:
            return None

        total_tp = sum(record.TP for record in metric_records)
        total_fp = sum(record.FP for record in metric_records)
        total_fn = sum(record.FN for record in metric_records)

        micro_precision_denom = total_tp + total_fp
        micro_recall_denom = total_tp + total_fn
        micro_precision = (
            total_tp / micro_precision_denom if micro_precision_denom > 0 else 0.0
        )
        micro_recall = total_tp / micro_recall_denom if micro_recall_denom > 0 else 0.0
        micro_f1_denom = 2 * total_tp + total_fp + total_fn
        micro_f1 = (2 * total_tp / micro_f1_denom) if micro_f1_denom > 0 else 0.0

        per_record_precisions = [
            (record.TP / (record.TP + record.FP))
            if (record.TP + record.FP) > 0
            else 0.0
            for record in metric_records
        ]
        per_record_recalls = [
            (record.TP / (record.TP + record.FN))
            if (record.TP + record.FN) > 0
            else 0.0
            for record in metric_records
        ]
        macro_precision = sum(per_record_precisions) / len(per_record_precisions)
        macro_recall = sum(per_record_recalls) / len(per_record_recalls)
        macro_f1 = sum(float(record.F1) for record in metric_records) / len(
            metric_records
        )

        return {
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "record_count": len(metric_records),
        }

    def print_stats(scores, total, dataset_name, metric_records):
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

        metrics = calculate_micro_macro_metrics(metric_records)
        if metrics is None:
            print("No valid records available for micro/macro precision/recall/F1.")
            print("-------------------------------------------------------")
            return

        print("-------------------------------------------------------")
        print(
            f"{metrics['record_count']} valid records for micro/macro precision/recall/F1 calculation "
            f"(excluding failed and zero-score classes)."
        )
        print(
            f"{dataset_name.title()} micro-precision: {metrics['micro_precision']:.4f}"
        )
        print(f"{dataset_name.title()} micro-recall: {metrics['micro_recall']:.4f}")
        print(f"{dataset_name.title()} micro-F1: {metrics['micro_f1']:.4f}")
        print(
            f"{dataset_name.title()} macro-precision: {metrics['macro_precision']:.4f}"
        )
        print(f"{dataset_name.title()} macro-recall: {metrics['macro_recall']:.4f}")
        print(f"{dataset_name.title()} macro-F1: {metrics['macro_f1']:.4f}")
        print("-------------------------------------------------------")

    train_total = len(scores)
    print(f"Metric: {requested_metric}")
    print(f"Total definitions: {len(data.learned_definitions)}")
    print(f"Definitions failed to learn: {failed} during learning process.")

    if train_total == 0:
        print("No valid scores found. Nothing to summarize.")
        return
    print_stats(scores, train_total, "training set", train_metric_records)

    if val_scores:
        print(
            "Validation micro/macro metrics should only be computed with classes that are",
            "common between c3po and ours, see `compare_with_c3p.py` for the rationale.",
            "This validation are for informational purposes only.",
        )
        val_total = len(val_scores)
        print_stats(val_scores, val_total, "validation set", val_metric_records)


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

    print("Current content for the upserted predicate:")
    print_pickle_contents(
        output_path,
        class_name=predicate_name,
        show_system_prompt=False,
        show_conversation_history=False,
    )

    return True


def set_class_learn_false_in_pickle(
    pickle_file_path: str, class_name: str, output_file_path=None, create_backup=True
):
    """Set `learn_success` to False for a learned or additional definition in a pickle."""

    if class_name == "all":
        raise ValueError(
            "Refusing to modify all classes. Please pass a specific class."
        )

    input_path = Path(pickle_file_path)
    original_pickle_bytes = input_path.read_bytes()
    data: DefinitionLearningResults = pickle.loads(original_pickle_bytes)

    target = str(class_name).strip()
    modified = False
    learned_modified_keys = []
    additional_modified_keys = []

    for key, learned_def in data.learned_definitions.items():
        if str(key) == target or learned_def.name == target:
            if learned_def.learn_success is not False:
                learned_def.learn_success = False
                modified = True
            learned_modified_keys.append(key)

    for name, add_def in data.additional_definitions.items():
        if name == target:
            if add_def.learn_success is not False:
                add_def.learn_success = False
                modified = True
            additional_modified_keys.append(name)

    total_modified = len(learned_modified_keys) + len(additional_modified_keys)
    if total_modified == 0 and not modified:
        print(f"No class found for '{class_name}'. Nothing modified.")
        return False

    output_path = Path(output_file_path) if output_file_path else input_path

    if create_backup and output_path.resolve() == input_path.resolve():
        backup_path = input_path.with_suffix(input_path.suffix + ".bak")
        backup_path.write_bytes(original_pickle_bytes)
        print(f"Backup written to: {backup_path}")

    with open(output_path, "wb") as f:
        pickle.dump(data, f)

    print(
        f"Marked {len(learned_modified_keys)} learned definition(s) as learn_success=False."
    )
    print(
        f"Marked {len(additional_modified_keys)} additional definition(s) as learn_success=False."
    )
    print(f"Updated pickle written to: {output_path}")

    print("Current content for the updated predicate:")
    print_pickle_contents(
        output_path,
        class_name=target,
        show_system_prompt=False,
        show_conversation_history=False,
    )
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

    set_learn_false_parser = subparsers.add_parser(
        "set-learn-false",
        help="Mark a learned or additional definition's learn_success as False.",
    )
    set_learn_false_parser.add_argument(
        "class_name",
        help="Class name or learned definition key to mark as failed.",
    )
    set_learn_false_parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Optional output pickle path. Defaults to in-place update.",
    )
    set_learn_false_parser.add_argument(
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
    # python nl_2_fol/inference/utils/show_learned_content.py --pickle-file=<file_path> set-learn-false hopanoid
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
    elif args.command == "set-learn-false":
        set_class_learn_false_in_pickle(
            args.pickle_file,
            class_name=args.class_name,
            output_file_path=args.output_file,
            create_backup=not args.no_backup,
        )
