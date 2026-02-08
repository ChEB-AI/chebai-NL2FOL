from gavel.logic.logic import QuantifiedFormula
from pydantic import BaseModel, Field, RootModel


class DefinitionMetrics(BaseModel):
    TP: int = Field(..., description="True Positives")
    FP: int = Field(..., description="False Positives")
    FN: int = Field(..., description="False Negatives")
    TN: int = Field(..., description="True Negatives")
    F1: float = Field(..., description="F1 score of the learned definition")
    PPV: float = Field(..., description="Positive Predictive Value (Precision)")
    NPV: float = Field(..., description="Negative Predictive Value")


class LearnedDefinition(BaseModel):
    metrics: DefinitionMetrics = Field(
        ..., description="Metrics of the learned definition"
    )
    definition: QuantifiedFormula = Field(
        ..., description="Learned TPTP FOL definition for the chemical class"
    )


class DefinitionLearningResults(RootModel[dict[str, LearnedDefinition]]):
    """Dictionary mapping ChEBI IDs to their learned definitions."""
    root: dict[str, LearnedDefinition]

    def __getitem__(self, item):
        return self.root[item]

    def __setitem__(self, key, value):
        self.root[key] = value

    def __contains__(self, item):
        return item in self.root
