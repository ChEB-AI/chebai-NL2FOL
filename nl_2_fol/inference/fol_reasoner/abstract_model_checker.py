from typing import List

from nl_2_fol.inference.fol_reasoner.base_predicates import GAVEL_PREDICATES

import json
from pathlib import Path

class FOLDefinition:
    
    def __init__(self, predicate_name: str, variables: List, definition):
        self.predicate_name = predicate_name
        self.variables = variables
        self.definition = definition

    def __str__(self):
        return str(self.definition)

class AbstractModelCheckerWrapper:

    def __init__(self) -> None:
        self._base_predicates: dict[str, str] = GAVEL_PREDICATES
        self.background_definitions: dict[
            str, FOLDefinition
        ] = {}

    def parse_definition(self, definition: str) -> FOLDefinition:
        """Parse a definition string into a FOLDefinition object."""
        raise NotImplementedError
    
    def _extract_predicate_names(self, formula) -> set[str]:
        """Extract predicate names from a formula."""
        raise NotImplementedError
    
    def extract_unknown_predicates(
        self,
        formula,
        temp_additional_defs: dict[
            str, FOLDefinition
        ]
        | None = None,
    ) -> set[str]:
        predicates = self._extract_predicate_names(formula)
        missing_predicates = predicates - self._base_predicates.keys()
        missing_predicates = (
            missing_predicates - self.background_definitions.keys()
            if self.background_definitions
            else missing_predicates
        )
        if temp_additional_defs:
            missing_predicates = missing_predicates - temp_additional_defs.keys()

        # {Token('LOWER_WORD', 'predicate_name')} -> {'predicate_name'}
        return {str(pred) for pred in missing_predicates}
    
    def add_background_definition(
        self,
        definition: FOLDefinition,
    ):
        """Add a single background definition with extracted free variables."""
        self.background_definitions[definition.predicate_name] = definition

    def save_background_definitions_to_json(self, file_path: str | Path) -> None:
        payload = []
        for predicate_name, definition in self.background_definitions.items():
            payload.append(
                {
                    "predicate": predicate_name,
                    "definition": str(definition),
                }
            )

        Path(file_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load_background_definitions_from_json(
        self,
        file_path: str | Path,
        *,
        replace: bool = True,
    ) -> dict[str, FOLDefinition]:
        """Load background definitions from a JSON file.

        Args:
            file_path: JSON file created by `save_background_definitions_to_json`.
            replace: If True, clear the current background definitions before loading.

        Returns:
            The loaded background definitions in the internal tuple format.
        """
        payload = json.loads(Path(file_path).read_text(encoding="utf-8"))

        loaded_definitions = {}
        for item in payload:
            predicate_name = item["predicate"]
            definition = self.parse_definition(
                item["definition"]
            )
            loaded_definitions[predicate_name] = definition

        if replace:
            self.background_definitions.clear()
        self.background_definitions.update(loaded_definitions)
        return loaded_definitions

    def convert_to_background_definitions(
        self,
        predicates: dict[str, str],
    ) -> dict[str, FOLDefinition]:
        """Convert a dictionary of predicate definitions (as strings) to the internal format."""
        converted = {}
        for name, def_str in predicates.items():
            converted[name] = self.parse_definition(def_str)
        return converted