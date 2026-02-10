import os
from typing import Hashable

import pandas as pd
from chemlog.preprocessing.chebi_data import ChEBIData


class ChEBIDataWrapper(ChEBIData):
    def __init__(self, chebi_version: int):
        self.chebi_version = chebi_version

        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(
            os.path.join(self.base_dir, f"chebi_v{self.chebi_version}"), exist_ok=True
        )
        # chebi: dict with entries from chebi

    def get_name_to_data_mapping(self) -> dict[Hashable, dict]:
        df = self._preprocess_data()
        # --- One Duplicate entry in Chebi 244 -----
        # 1. Name: l-lysine zwitterion
        #    Count: 2
        #    - ID: 133538, SMILES: [O-]C([C@H](CCCCN)[NH3+])=O...
        #    - ID: 194466, SMILES: [NH3+]CCCC[C@@H](C([O-])=O)N...

        # Latest chebi version has only 194466, so delete the duplicate entry with 133538
        df = df[df.index != 133538]

        return df.set_index("name").to_dict(orient="index")

    def _preprocess_data(self) -> pd.DataFrame:
        data_dict = self.process_chebi()
        df = pd.DataFrame.from_dict(data_dict, orient="index")
        df = df[["smiles", "definition", "name"]]
        df["name"] = df["name"].str.lower().str.strip()
        df = df.dropna(subset=["smiles", "definition", "name"])
        return df


if __name__ == "__main__":
    chebi_data_wrapper = ChEBIDataWrapper(chebi_version=244)
    # Get mappings
    name_to_data_mapping = chebi_data_wrapper.get_name_to_data_mapping()
    print(f"\n\nTotal unique names (mapping keys): {len(name_to_data_mapping)}")
    print(f"Sample entries: {list(name_to_data_mapping)[:5]}")
