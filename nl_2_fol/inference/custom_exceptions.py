from functools import wraps

from nl_2_fol.inference.data_model import SMILES_STRING, ChemicalStructure


def tptp_parse_exception(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"{func.__name__} failed: {e}")
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
            # Error can be customized here for LLMs feedback,
            # for now we just print the error and return it
            raise StopProgramException(str(e)) from e

    return wrapper


class RetryException(Exception):
    pass


class MissingPredicateException(Exception):
    def __init__(self, missing_predicates: set[str]) -> None:
        self.missing_predicates: set[str] = missing_predicates
        message = (
            f"Definition contains unknown predicates not in base predicates "
            f"or background definitions: {missing_predicates}"
        )
        super().__init__(message)


class LearnOutOfBoxPredicateException(Exception):
    def __init__(self, predicates_to_learn: dict[str, str | None]) -> None:
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
        super().__init__(message)


class LowF1ScoreException(Exception):
    """
    Exception raised when a generated FOL definition fails F1-score validation.

    Args:
        pos_samples: List of positive ChemicalStructure samples used in validation.
        neg_samples: List of negative ChemicalStructure samples used in validation.
        matched_neg_samples: List of SMILES strings for negative samples incorrectly matched (false positives).
        unmatched_pos_samples: List of SMILES strings for positive samples not matched (false negatives).
        max_examples: Maximum number of misclassified examples to include in error message.
        chebi_id_to_data_mapping: Mapping of chemical IDs to their data including definitions.
    """

    def __init__(
        self,
        pos_samples: list[ChemicalStructure],
        neg_samples: list[ChemicalStructure],
        matched_neg_samples: list[SMILES_STRING],
        unmatched_pos_samples: list[SMILES_STRING],
        max_examples: int,
        chebi_id_to_data_mapping: dict[str, dict],
    ) -> None:
        def get_chemical_details(
            chemicals: set[ChemicalStructure],
            matched_smiles: list[SMILES_STRING],
        ) -> list[tuple[str, str | None]]:
            chemical_details: list[tuple[str, str | None]] = []
            for smiles in matched_smiles[:max_examples]:
                if smiles in chemicals:
                    chemical_data = chebi_id_to_data_mapping.get(
                        str(smiles).lower().strip(), None
                    )
                    chemical_def = None
                    if chemical_data:
                        chemical_def = chemical_data.get("definition", "")
                    chemical_details.append((smiles, chemical_def))
            return chemical_details

        fp_percentage = (
            len(matched_neg_samples) / len(neg_samples) if neg_samples else 0
        )
        fn_percentage = (
            len(unmatched_pos_samples) / len(pos_samples) if pos_samples else 0
        )
        if fn_percentage < 0.1 and fp_percentage > 0.1:
            # When FN is less than 10% but FP is more than 10%,
            # we prioritize showing FP examples as they are more prevalent
            error_priority = "FP"
        elif fn_percentage > 0.1 and fp_percentage < 0.1:
            error_priority = "FN"
        else:
            error_priority = "both"

        fp_details, fn_details = None, None
        fn_mol_names: list[tuple[str, str | None]] = []
        if error_priority == "FP" or error_priority == "both":
            fp_mol_names = get_chemical_details(
                chemicals=set(neg_samples),
                matched_smiles=matched_neg_samples,
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
            fn_details = "\n".join(
                f"\t- Chemical Name: {name}"
                + (f", Chemical Definition: {chem_def}" if chem_def else "")
                for name, chem_def in fn_mol_names
            )

        message_parts = [
            "The generated FOL definition did not meet the required F1 score threshold:\n"
            "Please find below the names of molecules and optionally their definitions"
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
        pos_samples = [
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
        ]
        neg_samples = [
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
        ]
        matched_neg_samples = ["C1=CC=CC=C1N"]  # False positive
        unmatched_pos_samples = ["C1=CC=CC=C1O", "C1=CC=CC=C1"]  # False negative

        raise LowF1ScoreException(
            pos_samples,
            neg_samples,
            matched_neg_samples,
            unmatched_pos_samples,
            max_examples=2,
            chebi_id_to_data_mapping={
                "moleculec": {"definition": "Definition of MoleculeC"},
                "moleculed": {"definition": "Definition of MoleculeD"},
                "moleculea": {"definition": "Definition of MoleculeA"},
                "moleculeb": {"definition": ""},
            },
        )
    except LowF1ScoreException as e:
        print(f"Caught an exception: {e}")
