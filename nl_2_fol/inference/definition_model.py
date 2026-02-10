from gavel.logic.logic import QuantifiedFormula
from pydantic import BaseModel, Field, RootModel

from nl_2_fol.inference.data_model import CHEBI_ID


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
    learned_FOL: QuantifiedFormula = Field(
        ..., description="Learned TPTP FOL definition for the chemical class"
    )
    prompts_history: list[str] = Field(
        ..., description="History of prompts used to learn this definition"
    )
    name: str = Field(..., description="rdfs:label of the class in CHEBI")
    definition: str = Field(..., description="definition of the structure from CHEBI")


class DefinitionLearningResults(RootModel[dict[CHEBI_ID, LearnedDefinition]]):
    """Dictionary mapping ChEBI IDs to their learned definitions."""

    root: dict[CHEBI_ID, LearnedDefinition]

    def __getitem__(self, item):
        return self.root[item]

    def __setitem__(self, key, value):
        self.root[key] = value

    def __contains__(self, item):
        return item in self.root
