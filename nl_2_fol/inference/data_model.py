"""
Note this file(code) is copied from the following source:
https://github.com/chemkg/c3p/blob/main/c3p/datamodel.py

This file is used to load the https://github.com/chemkg/c3p 's  dataset
which is available at https://huggingface.co/datasets/MonarchInit/C3PO

"""

import os
from copy import copy
from typing import Optional

import pandas as pd
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


def load_c3po_slim_dataset(
    slim_dataset_path: str = "data/classes_slim.csv",
    structures_path: str = "data/structures.csv",
) -> tuple[
    Dataset,
    dict[str, ChemicalStructure],
    set[str],
    set[str],
]:
    if not os.path.exists(slim_dataset_path) or not os.path.exists(structures_path):
        raise FileNotFoundError(
            f"Dataset files not found. Please ensure {slim_dataset_path} "
            f"and {structures_path} exist."
        )

    slim_df = pd.read_csv(slim_dataset_path)
    structures_df = pd.read_csv(structures_path)

    validation_smiles = structures_df.loc[
        structures_df["in_validation_set"], "smiles"
    ].tolist()
    assert validation_smiles, (
        "No validation examples found in the dataset. Please check the dataset files."
    )

    # Convert to string type to avoid type errors with pandas Scalar
    def _split_if_notna(val: str) -> list[str] | None:
        return val.split("|") if pd.notna(val) else None

    classes = [
        ChemicalClass(
            id=str(row.id),
            name=str(row.name),
            definition=str(row.definition),
            parents=_split_if_notna(str(row.parents)),
            xrefs=_split_if_notna(str(row.xrefs)),
        )
        for row in slim_df.itertuples()
    ]

    structures = [
        ChemicalStructure(name=str(row.name), smiles=str(row.smiles))
        for row in structures_df.itertuples()
    ]

    dataset = Dataset(
        ontology_version="slim",
        classes=classes,
        structures=structures,
        validation_examples=validation_smiles,
    )

    print(
        f"Loaded : Classes: {len(dataset.classes)}\n"
        f"Instances: {len(dataset.structures)}\n"
        f"Validation examples: {len(dataset.validation_examples)}"  # pyright: ignore[reportArgumentType]
    )

    s2i = dataset.smiles_to_instance()
    all_validation = (
        set(dataset.validation_examples)
        if dataset.validation_examples is not None
        else set()
    )
    all_smiles = dataset.all_smiles()
    return dataset, s2i, all_validation, all_smiles


if __name__ == "__main__":
    load_c3po_slim_dataset()
