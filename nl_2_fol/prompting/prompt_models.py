import ast
import json

from pydantic import BaseModel, Field, field_validator, model_validator


# --- Pydantic Models ---
class IntermediateOutput(BaseModel):
    relevant_definition: str = Field(
        ..., description="Relevant part of the CHEBI definition"
    )
    superclasses: str = Field(..., description="Superclass(es) of the CHEBI class")
    explanation: str = Field(..., description="How the class is defined")


class ChEBI_FOL_AT(BaseModel):
    """Model for parsing ChEBI FOL definitions from LLM response."""

    FOL_formula: str = Field(..., description="First-order logic formula")

    @field_validator("FOL_formula", mode="before")
    @classmethod
    def _coerce_fol_formula(cls, value):
        if isinstance(value, list):
            return " ".join(
                part.strip() for part in value if isinstance(part, str) and part.strip()
            )
        return value


class CHEBIFOLOutput(ChEBI_FOL_AT):
    intermediate_output: IntermediateOutput

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


class PredicateDefinition(BaseModel):
    predicate_name: str = Field(..., description="Name of the defined predicate")
    formula: str = Field(..., description="FOL formula defining the predicate")


class OutOfBoxPredicateDefinitions(BaseModel):
    """Model for parsing predicate definitions from LLM response."""

    # A list instead of a dict (predicate name -> formula): objects with arbitrary keys
    # are not supported by strict structured outputs (e.g. Anthropic's output_config.format)
    definitions: list[PredicateDefinition] = Field(
        ...,
        description="FOL formulas for each of the undefined predicates",
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_dict_format(cls, value):
        # accept the previous format {"predicate_definitions": {name: formula}}
        if isinstance(value, dict) and isinstance(
            value.get("predicate_definitions"), dict
        ):
            return {
                "definitions": [
                    {"predicate_name": name, "formula": formula}
                    for name, formula in value["predicate_definitions"].items()
                ]
            }
        return value

    @property
    def predicate_definitions(self) -> dict[str, str]:
        """Mapping from predicate names to their formulas."""
        return {d.predicate_name: d.formula for d in self.definitions}


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
