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


class OutOfBoxPredicateDefinitions(BaseModel):
    """Model for parsing predicate definitions from LLM response."""

    predicate_definitions: dict[str, str] = Field(
        ...,
        description="Dictionary mapping predicate names to their FOL formulas",
    )


if __name__ == "__main__":
    # Example usage
    intermediate_output = IntermediateOutput(
        relevant_definition="A carbon monoxide is a compound that consists of one carbon atom and one oxygen atom.",
        superclass="carbon compound",
        explanation="The definition states that a carbon monoxide is a compound made of one carbon and one oxygen, which matches the superclass 'carbon compound'.",
    )
    fol_output = CHEBIFOLOutput(
        intermediate_output=intermediate_output,
        FOL_formula="carbonMonoxide(x) <=> (oneCarbonCompound(x) & hasPart(x, y) & c(y) & hasPart(x, z) & o(z))",
    )
    print(fol_output.model_dump_json(indent=2))

    out_of_box_predicates = OutOfBoxPredicateDefinitions(
        predicate_definitions={
            "oneCarbonCompound": "oneCarbonCompound(x) <=> ...",
            "hasPart": "hasPart(x, y) <=> ...",
        }
    )
    print(out_of_box_predicates.model_dump_json(indent=2))
