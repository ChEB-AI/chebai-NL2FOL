from gavel.logic import logic
from gavel.logic.logic import QuantifiedFormula
from pydantic import BaseModel, ConfigDict, Field

from nl_2_fol.inference.data_model import CHEBI_ID


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

    metrics: DefinitionMetrics = Field(
        ..., description="Metrics of the learned definition"
    )
    learned_FOL: FOLFormula = Field(
        ..., description="Learned FOL formula for the chemical class"
    )
    prompts_history: dict[str, str] = Field(
        ..., description="History of prompts used to learn this definition"
    )
    name: str = Field(..., description="rdfs:label of the class in CHEBI")
    definition: str = Field(..., description="definition of the structure from CHEBI")


class DefinitionLearningResults(BaseModel):
    """Dictionary mapping ChEBI IDs to their learned definitions."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    learned_definitions: dict[CHEBI_ID, LearnedDefinition] = Field(
        ..., description="Dictionary mapping ChEBI IDs to their learned definitions"
    )
    additional_definitions: dict[str, FOLFormula] = Field(
        ..., description="Additional definitions provided by the user (optional)"
    )


if __name__ == "__main__":
    # Example usage
    from nl_2_fol.inference.model_check_molecule import GavelFOLReasoner

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
                metrics=DefinitionMetrics(
                    TP=10, FP=2, FN=3, TN=85, F1=0.83, PPV=0.83, NPV=0.97
                ),
                learned_FOL=FOLFormula(
                    formula=fol_formula, pred_variables=pred_variables
                ),
                prompts_history=[
                    "What is the definition of CHEBI:12345?",
                    "List the properties of CHEBI:12345.",
                ],
                name="Example Chemical Class",
                definition="A chemical class used for demonstration purposes.",
            ),
            56645: LearnedDefinition(
                metrics=DefinitionMetrics(
                    TP=8, FP=1, FN=4, TN=87, F1=0.80, PPV=0.89, NPV=0.96
                ),
                learned_FOL=FOLFormula(
                    formula=fol_formula_2, pred_variables=pred_variables_2
                ),
                prompts_history=[
                    "What is the definition of CHEBI:56645?",
                    "List the properties of CHEBI:56645.",
                ],
                name="Another Example Chemical Class",
                definition="Another chemical class used for demonstration purposes.",
            ),
        },
        additional_definitions={
            "Example": FOLFormula(
                formula=fol_formula_3, pred_variables=pred_variables_3
            )
        },
    )
    print(results)
