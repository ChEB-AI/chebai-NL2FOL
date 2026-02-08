from pydantic import BaseModel, Field


# --- Pydantic Models ---
class IntermediateOutput(BaseModel):
    relevant_definition: str = Field(
        ..., description="Relevant part of the CHEBI definition"
    )
    superclass: str = Field(..., description="Superclass of the CHEBI class")
    explanation: str = Field(..., description="How the class is defined")


class CHEBIFOLOutput(BaseModel):
    intermediate_output: IntermediateOutput
    FOL_formula: str = Field(..., description="First-order logic formula")
