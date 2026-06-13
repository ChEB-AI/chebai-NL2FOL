import argparse
import pickle
from pathlib import Path

from nl_2_fol.inference.learner.definition_model import DefinitionLearningResults


def _load_definitions(pickle_path: Path) -> DefinitionLearningResults:
    with pickle_path.open("rb") as f:
        return pickle.load(f)


def _save_definitions(
    definitions: DefinitionLearningResults,
    output_path: Path,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Pass --overwrite to replace it."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(definitions, f)


def _has_validation_metric(
    definitions: DefinitionLearningResults,
    class_id: int,
) -> bool:
    return (
        class_id in definitions.learned_definitions
        and definitions.learned_definitions[class_id].val_metrics is not None
    )


def merge_validation_metrics(
    target_pickle_path: Path,
    source_pickle_path: Path,
    output_path: Path,
    overwrite: bool = False,
) -> None:
    target_definitions = _load_definitions(target_pickle_path)
    source_definitions = _load_definitions(source_pickle_path)

    merged_count = 0
    skipped_missing_count = 0
    skipped_empty_count = 0
    skipped_conflict_count = 0
    replaced_conflict_count = 0

    for class_id, learned_def in source_definitions.learned_definitions.items():
        if learned_def.val_metrics is None:
            skipped_empty_count += 1
            continue

        if class_id not in target_definitions.learned_definitions:
            raise ValueError(
                f"Class id {class_id} from second pickle not found in first pickle."
            )

        if _has_validation_metric(target_definitions, class_id):
            raise ValueError(
                f"Class id {class_id} already has validation metrics in first pickle."
            )
        else:
            merged_count += 1

        target_definitions.learned_definitions[
            class_id
        ].val_metrics = learned_def.val_metrics

    _save_definitions(
        definitions=target_definitions,
        output_path=output_path,
        overwrite=overwrite,
    )

    total_validated = sum(
        1
        for learned_def in target_definitions.learned_definitions.values()
        if learned_def.val_metrics is not None
    )

    print(f"Merged validation metrics written to: {output_path}")
    print(f"Added validation metrics from second pickle: {merged_count}")
    print(f"Replaced existing validation metrics: {replaced_conflict_count}")
    print(f"Skipped existing validation metrics: {skipped_conflict_count}")
    print(f"Skipped missing class ids: {skipped_missing_count}")
    print(f"Skipped empty validation metrics: {skipped_empty_count}")
    print(f"Total definitions with validation metrics: {total_validated}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge validation metrics from two DefinitionLearningResults pickle "
            "files into one output pickle."
        )
    )
    parser.add_argument(
        "target_pickle",
        type=Path,
        help=(
            "Base pickle file. Its learned definitions are kept, and validation "
            "metrics from the second pickle are merged into it."
        ),
    )
    parser.add_argument(
        "source_pickle",
        type=Path,
        help="Pickle file whose non-empty validation metrics should be merged.",
    )
    parser.add_argument(
        "output_pickle",
        type=Path,
        help="Path where the merged pickle should be written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_false",
        help="Replace output_pickle if it already exists.",
    )
    args = parser.parse_args()

    merge_validation_metrics(
        target_pickle_path=args.target_pickle,
        source_pickle_path=args.source_pickle,
        output_path=args.output_pickle,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
