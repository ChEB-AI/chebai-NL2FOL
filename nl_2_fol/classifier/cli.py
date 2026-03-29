import argparse
import json
from pathlib import Path

from nl_2_fol.classifier.classify import NL2FOLChebiClassifier


def _load_smiles_from_file(file_path: Path) -> list[str]:
    if not file_path.exists():
        raise FileNotFoundError(f"SMILES file not found: {file_path}")

    smiles: list[str] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            cleaned = line.strip()
            if not cleaned or cleaned.startswith("#"):
                continue
            smiles.append(cleaned)
    return smiles


def _collect_smiles(smiles_args: list[str], smiles_file: Path | None) -> list[str]:
    smiles: list[str] = []
    if smiles_args:
        smiles.extend(smiles_args)
    if smiles_file is not None:
        smiles.extend(_load_smiles_from_file(smiles_file))

    # Remove duplicates while preserving input order.
    return list(dict.fromkeys(smiles))


def _print_human_readable(results: list[dict]) -> None:
    for result in results:
        smiles = next(iter(result.keys()))
        classifications = result[smiles]

        if isinstance(classifications, dict) and "error" in classifications:
            print(f"{smiles}: ERROR - {', '.join(classifications['error'])}")
            continue

        if not classifications:
            print(f"{smiles}: no matching classes")
            continue

        print(f"{smiles}:")
        for cls in classifications:
            print(f"  - CHEBI:{cls['chebi_id']} | {cls['name']}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify molecules into ChEBI classes using learned NL2FOL definitions."
    )
    parser.add_argument(
        "--definitions-file",
        type=Path,
        help="Path to learned_definitions pickle file.",
        required=True,
    )
    parser.add_argument(
        "--smiles",
        action="append",
        default=[],
        help="SMILES string to classify. Can be repeated.",
    )
    parser.add_argument(
        "--smiles-file",
        type=Path,
        default=None,
        help="Path to a text file with one SMILES per line.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Optional path to write JSON results.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print output as JSON instead of human-readable text.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    smiles_list = _collect_smiles(args.smiles, args.smiles_file)
    if not smiles_list:
        print("No SMILES provided. Use --smiles and/or --smiles-file.")
        return 1

    classifier = NL2FOLChebiClassifier(str(args.definitions_file))
    results = classifier.classify_smiles_list(smiles_list)

    if args.output_file is not None:
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {args.output_file}")

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        _print_human_readable(results)

    return 0


if __name__ == "__main__":
    # ---- Example Usage --------------
    # python nl_2_fol/classifier/cli.py \
    # --definitions-file nl_2_fol/inference/learner/learned/claude-opus-4-6/learned_definitions.pkl \
    # --smiles "CCO" \
    # --smiles "CCCC(=O)O" \
    # --json
    raise SystemExit(main())
