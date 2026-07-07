import ast
import json

from pydantic import BaseModel, Field, field_validator


# --- Pydantic Models ---
class IntermediateOutput(BaseModel):
    relevant_definition: str = Field(
        ..., description="Relevant part of the CHEBI definition"
    )
    superclasses: str = Field(..., description="Superclass(es) of the CHEBI class")
    explanation: str = Field(..., description="How the class is defined")


class CHEBIFOLOutput(BaseModel):
    intermediate_output: IntermediateOutput
    FOL_formula: str = Field(..., description="First-order logic formula")

    @field_validator("intermediate_output", mode="before")
    @classmethod
    def _coerce_intermediate_output(cls, value):
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return value

            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(stripped)
                except (json.JSONDecodeError, ValueError, SyntaxError):
                    continue
                if isinstance(parsed, dict):
                    return parsed

        return value

    @field_validator("FOL_formula", mode="before")
    @classmethod
    def _coerce_fol_formula(cls, value):
        if isinstance(value, list):
            return " ".join(
                part.strip() for part in value if isinstance(part, str) and part.strip()
            )
        return value


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
        superclasses="carbon compound",
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
