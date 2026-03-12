"""
Note this file(code) is copied from the following source and modified to fit our use case:
https://github.com/chemkg/c3p/blob/main/c3p/datamodel.py

This file is used to load the https://github.com/chemkg/c3p 's  dataset
which is available at https://huggingface.co/datasets/MonarchInit/C3PO

"""

import ast
import os
import sys
from copy import copy

import pandas as pd
import tqdm
from pydantic import BaseModel, Field
from rdkit import Chem

from nl_2_fol.inference.preprocessing import CHEBI_ID, SMILES_STRING
from nl_2_fol.inference.preprocessing.chebi_data import ChEBIDataWrapper
from nl_2_fol.inference.utils.to_camel_case import to_camel_case


# make this hashable?
class ChemicalStructure(BaseModel):
    """Represents a chemical entity with a known specific structure/formula."""

    model_config = {"arbitrary_types_allowed": True}

    name: str = Field(..., description="rdfs:label of the structure in CHEBI")
    smiles: SMILES_STRING = Field(..., description="SMILES string derived from CHEBI")
    mol: Chem.Mol = Field(
        ..., description="RDKit Mol object derived from the SMILES string"
    )

    def __hash__(self):
        return hash(self.smiles)

    def __eq__(self, other):
        if not isinstance(other, ChemicalStructure):
            return NotImplemented
        return self.smiles == other.smiles


class ChemicalClass(BaseModel):
    """Represents a class/grouping of chemical entities."""

    id: CHEBI_ID = Field(..., description="id/curie of the CHEBI class")
    name: str = Field(..., description="rdfs:label of the class in CHEBI")
    definition: str | None = Field(
        None, description="definition of the structure from CHEBI"
    )
    parents: list[str] | None = Field(default=None, description="parent classes")
    xrefs: list[str] | None = Field(default=None, description="mappings")
    all_positive_examples: set[SMILES_STRING] = Field(
        ...,
        description="list of SMILES strings of all positive examples of the class",
    )

    def lite_copy(self) -> "ChemicalClass":
        """
        Create a copy of the chemical class without the instance fields
        Returns:
        """
        cc = copy(self)
        cc.all_positive_examples = set()
        return cc

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, ChemicalStructure):
            return NotImplemented
        return self.name == other.name


class Dataset(BaseModel):
    """
    Represents a dataset of chemical classes.
    """

    ontology_version: str | None = None
    min_members: int | None = None
    max_members: int | None = None
    # class name -> ChemicalClass object, dict is needed to preserve insertion order

    classes: dict[str, ChemicalClass]
    structures: set[ChemicalStructure] = set()

    @property
    def name(self):
        return f"bench-{self.ontology_version}-{self.min_members}-{self.max_members}"

    @property
    def all_smiles(self) -> set[SMILES_STRING]:
        cache_attr = "_all_smiles_cache"
        if not hasattr(self, cache_attr):
            setattr(self, cache_attr, {s.smiles for s in self.structures})
        return getattr(self, cache_attr)

    @property
    def smiles_to_instance(self) -> dict[SMILES_STRING, ChemicalStructure]:
        cache_attr = "_smiles_to_instance_cache"
        if not hasattr(self, cache_attr):
            setattr(self, cache_attr, {s.smiles: s for s in self.structures})
        return getattr(self, cache_attr)

    @property
    def id_to_class_name(self) -> dict[str, str]:
        cache_attr = "_id_to_class_name_cache"
        if not hasattr(self, cache_attr):
            setattr(
                self,
                cache_attr,
                {str(cls.id): cls.name for cls in self.classes.values()},
            )
        return getattr(self, cache_attr)

    def get_chemical_class_by_id(self, class_id: CHEBI_ID) -> ChemicalClass:
        for cc in self.classes.values():
            if cc.id == class_id:
                return cc
        raise ValueError(f"Class {class_id} not found in dataset")

    def get_chemical_class_by_name(self, class_name: str) -> ChemicalClass:
        if class_name not in self.classes:
            raise ValueError(f"Class {class_name} not found in dataset")
        return self.classes[class_name]


def load_c3po_slim_dataset(
    *,
    slim_dataset_path: str = "data/classes_slim.csv",
    structures_path: str = "data/structures.csv",
    chebi_version: int = 244,
    split="train",
) -> tuple[Dataset, ChEBIDataWrapper]:
    print("Loading and processing C3PO slim dataset...")
    if not os.path.exists(slim_dataset_path) or not os.path.exists(structures_path):
        raise FileNotFoundError(
            f"Dataset files not found. Please ensure {slim_dataset_path} "
            f"and {structures_path} exist."
        )

    slim_df = pd.read_csv(slim_dataset_path)
    structures_df = pd.read_csv(structures_path)
    validation_smiles = set(
        structures_df.loc[structures_df["in_validation_set"], "smiles"]
    )
    assert validation_smiles, (
        "No validation examples found in the dataset. Please check the dataset files."
    )
    if split == "train":
        structures_df = structures_df[~structures_df["in_validation_set"]]
        slim_df = slim_df.loc[~slim_df["smiles"].isin(validation_smiles)]
        print(f"Found {len(slim_df)} classes for train set ")
        print(f"Found {len(structures_df)} training structures")

    elif split == "val":
        structures_df = structures_df[structures_df["in_validation_set"]]
        slim_df = slim_df.loc[slim_df["smiles"].isin(validation_smiles)]
        print(f"Found {len(slim_df)} classes for validation set")
        print(f"Found {len(structures_df)} validation structures ")
    else:
        raise ValueError()

    assert not structures_df.empty and not slim_df.empty

    chemlog_chebi_class = ChEBIDataWrapper(
        chebi_version=chebi_version, validation_smiles=validation_smiles
    )
    slim_df["id"] = slim_df["id"].apply(chemlog_chebi_class.chebi_to_int)
    slim_df["name"] = slim_df["name"].apply(to_camel_case)

    # Sorting abstract classes first, specific classes later,
    # This to ensure when FOL definition for specific class is generated, all
    # the predicates in the definition are known
    topological_ordering = chemlog_chebi_class.get_topological_ordering()
    order_index = {v: i for i, v in enumerate(topological_ordering)}
    slim_df = slim_df.sort_values("id", key=lambda x: x.map(order_index))

    assert sys.version_info >= (
        3,
        7,
    ), "This code requires Python 3.7 or higher."
    # For python 3.7+, the standard dict type preserves insertion order, and is iterated over in same order
    # https://docs.python.org/3/whatsnew/3.7.html#summary-release-highlights
    # https://mail.python.org/pipermail/python-dev/2017-December/151283.html

    def parse_smiles_to_mol(smiles: str) -> Chem.Mol | None:
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol
        except Exception:
            return None

    structures_df["name"] = structures_df["name"].apply(to_camel_case)
    structures = {
        ChemicalStructure(
            name=str(row.name),
            smiles=str(row.smiles),
            mol=mol,
        )
        for row in tqdm.tqdm(
            structures_df.itertuples(),
            total=len(structures_df),
            desc="Loading structures",
        )
        if (mol := parse_smiles_to_mol(str(row.smiles))) is not None
    }

    def parse_positive_examples(examples: str) -> set[SMILES_STRING]:
        p_examples: set[SMILES_STRING] = set(ast.literal_eval(str(examples)))
        return p_examples - validation_smiles

    classes = {
        row.name: ChemicalClass(
            id=row.id,  # pyright: ignore[reportArgumentType]
            name=str(row.name),
            definition=str(row.definition),
            all_positive_examples=parse_positive_examples(
                str(row.all_positive_examples)
            ),
        )
        for row in tqdm.tqdm(
            slim_df.itertuples(), total=len(slim_df), desc="Loading classes"
        )
    }

    dataset = Dataset(
        ontology_version="slim",
        classes=classes,  # pyright: ignore[reportArgumentType]
        structures=structures,
    )

    print(
        f"For split : {split}"
        f"Loaded : Classes: {len(dataset.classes)}\n"
        f"Instances: {len(dataset.structures)}\n"
    )

    return dataset, chemlog_chebi_class


if __name__ == "__main__":
    load_c3po_slim_dataset()
