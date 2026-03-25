import traceback
from functools import wraps

from nl_2_fol.inference.preprocessing import SMILES_STRING
from nl_2_fol.inference.preprocessing.c3po_slim_data import ChemicalStructure


def tptp_parse_exception(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"{func.__name__} failed: {e}")
            traceback.print_exc()
            # Error can be customized here for LLMs feedback,
            # for now we just print the error and return it
            raise e

    return wrapper


def mol_to_fol_exception(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"{func.__name__} failed: {e}")
            traceback.print_exc()
            # Error can be customized here for LLMs feedback,
            # for now we just print the error and return it
            raise e

    return wrapper


def model_check_exception(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"{func.__name__} failed: {e}")
            traceback.print_exc()
            # Error can be customized here for LLMs feedback,
            # for now we just print the error and return it
            raise e

    return wrapper


class StopProgramException(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


def stop_program_upon_failure(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"{func.__name__} failed: {e}")
            traceback.print_exc()
            # Error can be customized here for LLMs feedback,
            # for now we just print the error and return it
            raise StopProgramException(str(e)) from e

    return wrapper


class RetryException(Exception):
    pass


class MissingPredicateException(Exception):
    @stop_program_upon_failure
    def __init__(self, missing_predicates: set[str]) -> None:
        assert len(missing_predicates) > 0, (
            "Expected at least one missing predicate but got an empty set."
        )
        self.missing_predicates: set[str] = missing_predicates
        message = (
            f"Definition contains unknown predicates not in base predicates "
            f"or background definitions: {missing_predicates}"
        )
        super().__init__(message)


class LearnOutOfBoxPredicateException(Exception):
    @stop_program_upon_failure
    def __init__(self, predicates_to_learn: dict[str, str | None]) -> None:
        assert len(predicates_to_learn) > 0, (
            "Expected at least one predicate to learn but got an empty dictionary."
        )
        self.predicates_to_learn: dict[str, str | None] = predicates_to_learn
        predicates_details = "\n".join(
            f"  - Predicate: {name}"
            + (f"\n    Chemical Definition: {definition}" if definition else "")
            for name, definition in predicates_to_learn.items()
        )
        message = (
            "Below is the list of predicates whose FOL formulas are not defined.\n\n"
            f"{predicates_details}\n\n"
        )
        self.message = message
        super().__init__(self.message)


class LowF1ScoreException(Exception):
    """
    Exception raised when a generated FOL definition fails F1-score validation.

    Args:
        current_f1_score: The F1 score achieved by the generated definition.
        pos_samples: List of positive ChemicalStructure samples.
        neg_samples: List of negative ChemicalStructure samples.
        matched_neg_samples: List of SMILES strings for negative samples incorrectly matched (false positives).
        unmatched_pos_samples: List of SMILES strings for positive samples not matched (false negatives).
        max_examples: Maximum number of misclassified examples to include in error message.
        chebi_id_to_data_mapping: Mapping of chemical IDs to their data including definitions.
    """

    @stop_program_upon_failure
    def __init__(
        self,
        current_f1_score: float,
        pos_samples: set[ChemicalStructure],
        neg_samples: set[ChemicalStructure],
        matched_neg_samples: set[SMILES_STRING],
        unmatched_pos_samples: set[SMILES_STRING],
        max_examples: int,
        chebi_name_to_data_mapping: dict[str, dict],
    ) -> None:
        def get_chemical_details(
            chemicals: set[ChemicalStructure],
            matched_smiles: set[SMILES_STRING],
        ) -> list[tuple[str, str | None]]:
            chemical_details: list[tuple[str, str | None]] = []
            chemicals_without_def: list[tuple[str, str | None]] = []

            # First pass: collect chemicals with definitions
            for chemical in chemicals:
                if chemical.smiles in matched_smiles:
                    chemical_data = chebi_name_to_data_mapping.get(
                        chemical.name,
                        chebi_name_to_data_mapping.get(chemical.name.lower(), None),
                    )
                    chemical_def = None
                    if chemical_data:
                        chemical_def = chemical_data.get("definition", "")

                    if chemical_def:
                        chemical_details.append((chemical.name, chemical_def))
                        if len(chemical_details) >= max_examples:
                            return chemical_details
                    else:
                        chemicals_without_def.append((chemical.name, chemical_def))

            # Second pass: fill remaining slots with chemicals without definitions
            remaining_slots = max_examples - len(chemical_details)
            chemical_details.extend(chemicals_without_def[:remaining_slots])

            return chemical_details

        fp_percentage = (
            len(matched_neg_samples) / len(neg_samples) if neg_samples else 0.0
        )
        fn_percentage = (
            len(unmatched_pos_samples) / len(pos_samples) if pos_samples else 0.0
        )
        if (fn_percentage < 0.1 and fp_percentage > 0.1) or (
            fn_percentage == 0.0 and fp_percentage > 0.0
        ):
            # When FN is less than 10% but FP is more than 10%,
            # we prioritize showing FP examples as they are more prevalent
            error_priority = "FP"
        elif (fn_percentage > 0.1 and fp_percentage < 0.1) or (
            fn_percentage > 0.0 and fp_percentage == 0.0
        ):
            error_priority = "FN"

        elif fn_percentage == 0.0 and fp_percentage == 0.0:
            raise ValueError(
                "Both false positive and false negative percentages are zero, which is "
                "unexpected when F1 score is low. Please check the input data and "
                "calculations."
            )
        else:
            error_priority = "both"

        fp_details, fn_details = None, None
        fn_mol_names: list[tuple[str, str | None]] = []
        if error_priority == "FP" or error_priority == "both":
            fp_mol_names = get_chemical_details(
                chemicals=set(neg_samples),
                matched_smiles=matched_neg_samples,
            )
            assert len(fp_mol_names) > 0, (
                "Expected at least one false positive sample to show details for "
                "but found none. Please check the input data and mappings."
            )
            fp_details = "\n".join(
                f"\t- Chemical Name: {name}"
                + (f", Chemical Definition: {chem_def}" if chem_def else "")
                for name, chem_def in fp_mol_names
            )

        if error_priority == "FN" or error_priority == "both":
            fn_mol_names = get_chemical_details(
                chemicals=set(pos_samples),
                matched_smiles=unmatched_pos_samples,
            )
            assert len(fn_mol_names) > 0, (
                "Expected at least one false negative sample to show details for "
                "but found none. Please check the input data and mappings."
            )
            fn_details = "\n".join(
                f"\t- Chemical Name: {name}"
                + (f", Chemical Definition: {chem_def}" if chem_def else "")
                for name, chem_def in fn_mol_names
            )

        message_parts = [
            "The generated FOL definition did not meet the required F1 score threshold:\n"
            f"Current F1 Score: {current_f1_score:.2f}\n"
            "Please find below the names of some molecules and optionally their definitions"
            " that were misclassified:\n"
        ]
        if fp_details is not None:
            message_parts.append(f"False Positives (FP): \n{fp_details}\n")
        if fn_details is not None:
            message_parts.append(f"False Negatives (FN): \n{fn_details}\n")

        message = "".join(message_parts)

        super().__init__(message)


if __name__ == "__main__":
    # Example usage of the custom exceptions
    from rdkit import Chem

    try:
        raise MissingPredicateException({"UnknownPredicate1", "UnknownPredicate2"})
    except MissingPredicateException as e:
        print(f"Caught an exception: {e}")

    try:
        pos_samples = {
            ChemicalStructure(
                name="MoleculeA",
                smiles="C1=CC=CC=C1",
                mol=Chem.MolFromSmiles("C1=CC=CC=C1"),
            ),
            ChemicalStructure(
                name="MoleculeB",
                smiles="C1=CC=CC=C1O",
                mol=Chem.MolFromSmiles("C1=CC=CC=C1O"),
            ),
        }
        neg_samples = {
            ChemicalStructure(
                name="MoleculeC",
                smiles="C1=CC=CC=C1N",
                mol=Chem.MolFromSmiles("C1=CC=CC=C1N"),
            ),
            ChemicalStructure(
                name="MoleculeD",
                smiles="C1=CC=CC=C1F",
                mol=Chem.MolFromSmiles("C1=CC=CC=C1F"),
            ),
        }
        matched_neg_samples = {"C1=CC=CC=C1N"}  # False positive
        unmatched_pos_samples = {"C1=CC=CC=C1O", "C1=CC=CC=C1"}  # False negative

        raise LowF1ScoreException(
            current_f1_score=0.65,
            pos_samples=pos_samples,
            neg_samples=neg_samples,
            matched_neg_samples=matched_neg_samples,
            unmatched_pos_samples=unmatched_pos_samples,
            max_examples=2,
            chebi_name_to_data_mapping={
                "moleculec": {"definition": "Definition of MoleculeC"},
                "moleculed": {"definition": "Definition of MoleculeD"},
                "moleculea": {"definition": "Definition of MoleculeA"},
                "moleculeb": {"definition": ""},
            },
        )
    except LowF1ScoreException as e:
        print(f"Caught an exception: {e}")
