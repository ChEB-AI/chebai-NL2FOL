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
            return e

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
            return e

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
            return e

    return wrapper


class MissingPredicateException(Exception):
    def __init__(self, missing_predicates: set) -> None:
        message = (
            f"Definition contains unknown predicates not in base predicates "
            f"or background definitions: {missing_predicates}"
        )
        super().__init__(message)


class LowF1ScoreException(Exception):
    """
    Initialize a custom exception for FOL definition F1-score validation failure.
    Args:
        pos_samples: List of positive ChemicalStructure samples used in validation.
        neg_samples: List of negative ChemicalStructure samples used in validation.
        matched_neg_samples: List of SMILES strings for negative samples that were incorrectly matched (false positives).
        unmatched_pos_samples: List of SMILES strings for positive samples that were not matched (false negatives).
        max_examples: Maximum number of misclassified molecule names to include in the error message. Defaults to 5.
    Returns:
        None
    Raises:
        Constructs an exception message detailing the F1 score threshold failure and lists up to max_examples
        of false positive and false negative molecule names for debugging purposes.
    """

    def __init__(
        self,
        pos_samples: list[ChemicalStructure],
        neg_samples: list[ChemicalStructure],
        matched_neg_samples: list[SMILES_STRING],
        unmatched_pos_samples: list[SMILES_STRING],
        max_examples: int = 5,
    ) -> None:

        fp_mol_names = [
            chemical.name
            for chemical in neg_samples
            if chemical.smiles in matched_neg_samples
        ][:max_examples]

        fn_mol_names = [
            chemical.name
            for chemical in pos_samples
            if chemical.smiles in unmatched_pos_samples
        ][:max_examples]

        message = (
            f"The generated FOL definition did not meet the required F1 score threshold:\n"
            f"Please find below the names of molecules that were misclassified:\n"
            f"False Positives (FP): {fp_mol_names}\n"
            f"False Negatives (FN): {fn_mol_names}"
        )

        super().__init__(message)
