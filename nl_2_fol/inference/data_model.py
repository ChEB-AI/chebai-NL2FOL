"""
Note this file(code) is copied from the following source:
https://github.com/chemkg/c3p/blob/main/c3p/datamodel.py

This file is used to load the https://github.com/chemkg/c3p 's  dataset
which is available at https://huggingface.co/datasets/MonarchInit/C3PO

"""

from copy import copy
from typing import Optional

from pydantic import BaseModel, Field

SMILES_STRING = str


# make this hashable?
class ChemicalStructure(BaseModel):
    """Represents a chemical entity with a known specific structure/formula."""

    name: str = Field(..., description="rdfs:label of the structure in CHEBI")
    smiles: SMILES_STRING = Field(..., description="SMILES string derived from CHEBI")

    def __hash__(self):
        return hash(self.smiles)

    def __eq__(self, other):
        if not isinstance(other, ChemicalStructure):
            return NotImplemented
        return self.smiles == other.smiles


class ChemicalClass(BaseModel):
    """Represents a class/grouping of chemical entities."""

    id: str = Field(..., description="id/curie of the CHEBI class")
    name: str = Field(..., description="rdfs:label of the class in CHEBI")
    definition: Optional[str] = Field(
        None, description="definition of the structure from CHEBI"
    )
    parents: Optional[list[str]] = Field(default=None, description="parent classes")
    xrefs: Optional[list[str]] = Field(default=None, description="mappings")
    all_positive_examples: list[SMILES_STRING] = []

    def lite_copy(self) -> "ChemicalClass":
        """
        Create a copy of the chemical class without the instance fields
        Returns:
        """
        cc = copy(self)
        cc.all_positive_examples = []
        return cc


class Dataset(BaseModel):
    """
    Represents a dataset of chemical classes.
    """

    ontology_version: Optional[str] = None
    min_members: Optional[int] = None
    max_members: Optional[int] = None
    classes: list[ChemicalClass]
    structures: list[ChemicalStructure] = []
    validation_examples: Optional[list[SMILES_STRING]] = None

    @property
    def name(self):
        return f"bench-{self.ontology_version}-{self.min_members}-{self.max_members}"

    def all_smiles(self) -> set[SMILES_STRING]:
        return {s.smiles for s in self.structures}

    def smiles_to_instance(self) -> dict[SMILES_STRING, ChemicalStructure]:
        return {s.smiles: s for s in self.structures}

    def get_chemical_class_by_id(self, class_id: str) -> ChemicalClass:
        for cc in self.classes:
            if cc.id == class_id:
                return cc
        raise ValueError(f"Class {class_id} not found in dataset")

    def get_chemical_class_by_name(self, class_name: str) -> ChemicalClass:
        for cc in self.classes:
            if cc.name == class_name:
                return cc
        raise ValueError(f"Class {class_name} not found in dataset")


if __name__ == "__main__":
    dataset_path = "data/dataset.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = Dataset.model_validate_json(f.read())
        print(f"Classes: {len(dataset.classes)} Instances: {len(dataset.structures)}")

    assert len(dataset.structures) > 0, "Dataset should have at least one structure"
    assert len(dataset.classes) == len(dataset.structures), (
        "Number of classes should be equal to number of structures"
    )
