from typing import Any

from gavel.logic import logic
from gavel.logic.logic import QuantifiedFormula
from pydantic import BaseModel, ConfigDict, Field

from nl_2_fol.inference.preprocessing import CHEBI_ID


class DefinitionMetrics(BaseModel):
    TP: int = Field(..., description="True Positives")
    FP: int = Field(..., description="False Positives")
    FN: int = Field(..., description="False Negatives")
    TN: int = Field(..., description="True Negatives")
    F1: float = Field(..., description="F1 score of the learned definition")
    PPV: float = Field(..., description="Positive Predictive Value (Precision)")
    NPV: float = Field(..., description="Negative Predictive Value")


class FOLFormula(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    formula: QuantifiedFormula = Field(
        ..., description="TPTP FOL formula representing the definition"
    )
    pred_variables: list[logic.Variable] = Field(
        ..., description="List of predicate variables used in the formula"
    )


class LearnedDefinition(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    train_metrics: DefinitionMetrics = Field(
        ..., description="Metrics of the learned definition"
    )
    val_metrics: DefinitionMetrics | None = Field(
        default=None, description="Metrics of the learned definition"
    )
    learned_FOL: FOLFormula = Field(
        ..., description="Learned FOL formula for the chemical class"
    )
    prompts_history: dict[str, Any] = Field(
        ..., description="History of prompts used to learn this definition"
    )
    name: str = Field(..., description="rdfs:label of the class in CHEBI")
    definition: str = Field(..., description="definition of the structure from CHEBI")

    learn_success: bool = Field(
        default=True,
        description="If False, indicates definition could not be learned "
        "(e.g., due to generated FOL being too complex for model checking)",
    )


class AdditionalDefinition(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    fol_formula: FOLFormula = Field(
        ..., description="FOL formula representing the additional definition"
    )
    used_for: list[CHEBI_ID] = Field(
        ...,
        description="List of ChEBI IDs for which this additional definition is relevant",
    )

    learn_success: bool = Field(
        default=True,
        description="If False, indicates definition could not be learned with "
        "ATLEAST one of the main chemical class definition "
        "(e.g., due to generated FOL being too complex for model checking)",
    )


class DefinitionLearningResults(BaseModel):
    """Dictionary mapping ChEBI IDs to their learned definitions."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    learned_definitions: dict[CHEBI_ID, LearnedDefinition] = Field(
        ..., description="Dictionary mapping ChEBI IDs to their learned definitions"
    )
    additional_definitions: dict[str, AdditionalDefinition] = Field(
        ..., description="Additional definitions provided by the user (optional)"
    )


if __name__ == "__main__":
    # Example usage
    from nl_2_fol.inference.fol_reasoner import GavelFOLReasoner

    gavel = GavelFOLReasoner()
    pred_variables, fol_formula = gavel.get_tptp_fol_definition(
        "carbonMonoxide <=> ?[A1, A2]: (c(A1) & o(A2) & has_bond_to(A1,A2))"
    )
    pred_variables_2, fol_formula_2 = gavel.get_tptp_fol_definition(
        "twoPlusCarbonCompound <=> ?[X, Y]: (c(X) & c(Y) & has_bond_to(X, Y) & X != Y)"
    )
    pred_variables_3, fol_formula_3 = gavel.get_tptp_fol_definition(
        "oneCarbonCompound <=> ?[X]: (c(X) & ~twoPlusCarbonCompound)"
    )
    results = DefinitionLearningResults(
        learned_definitions={
            12345: LearnedDefinition(
                train_metrics=DefinitionMetrics(
                    TP=10, FP=2, FN=3, TN=85, F1=0.83, PPV=0.83, NPV=0.97
                ),
                learned_FOL=FOLFormula(
                    formula=fol_formula, pred_variables=pred_variables
                ),
                prompts_history={
                    "What is the definition of CHEBI:12345?": "A chemical class used for demonstration purposes.",
                    "List the properties of CHEBI:12345.": "Properties of CHEBI:12345.",
                },
                name="Example Chemical Class",
                definition="A chemical class used for demonstration purposes.",
            ),
            56645: LearnedDefinition(
                train_metrics=DefinitionMetrics(
                    TP=8, FP=1, FN=4, TN=87, F1=0.80, PPV=0.89, NPV=0.96
                ),
                learned_FOL=FOLFormula(
                    formula=fol_formula_2, pred_variables=pred_variables_2
                ),
                prompts_history={
                    "What is the definition of CHEBI:56645?": "A chemical class used for demonstration purposes.",
                    "List the properties of CHEBI:56645.": "Properties of CHEBI:56645.",
                },
                name="Another Example Chemical Class",
                definition="Another chemical class used for demonstration purposes.",
            ),
        },
        additional_definitions={
            "Example": AdditionalDefinition(
                fol_formula=FOLFormula(
                    formula=fol_formula_3, pred_variables=pred_variables_3
                ),
                used_for=[12345, 56645],
            )
        },
    )
    print(results)
