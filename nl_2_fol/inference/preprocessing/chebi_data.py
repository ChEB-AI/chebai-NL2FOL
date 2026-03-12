import os
import pickle

import networkx as nx
import pandas as pd
from chemlog.preprocessing.chebi_data import ChEBIData

from nl_2_fol.inference.preprocessing.c3po_slim_data import SMILES_STRING
from nl_2_fol.inference.utils.to_camel_case import to_camel_case


class ChEBIDataWrapper(ChEBIData):
    def __init__(self, chebi_version: int, validation_smiles: set[SMILES_STRING]):
        self.chebi_version = chebi_version
        self.validation_smiles = validation_smiles

        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(
            os.path.join(self.base_dir, f"chebi_v{self.chebi_version}"), exist_ok=True
        )
        # ---- Dont want both chebi and processed data to be loaded into memory
        # ---- processed data will be loaded when required
        # # chebi: dict with entries from chebi
        # self.chebi = self.process_chebi()
        # # processed: dataframe that combines chebi data with mols from sdf file
        # self.processed = self.process_data()

    def get_name_to_data_mapping_train(self) -> dict[str, dict]:
        df = self._get_name_to_data_mapping()
        df = df[~df["smiles"].isin(self.validation_smiles)]
        return df.set_index("name").to_dict(orient="index")  # pyright: ignore[reportReturnType]

    def get_name_to_data_mapping_all(self) -> dict[str, dict]:
        df = self._get_name_to_data_mapping()
        return df.set_index("name").to_dict(orient="index")  # pyright: ignore[reportReturnType]

    def _get_name_to_data_mapping(self) -> pd.DataFrame:
        df = self._preprocess_data()
        # --- One Duplicate entry in Chebi 244 -----
        # 1. Name: l-lysine zwitterion
        #    Count: 2
        #    - ID: 133538, SMILES: [O-]C([C@H](CCCCN)[NH3+])=O...
        #    - ID: 194466, SMILES: [NH3+]CCCC[C@@H](C([O-])=O)N...

        # Latest chebi version has only 194466, so delete the duplicate entry with 133538
        df = df[df.index != 133538]

        # CHEBI:91305  lysophosphatidylcholine(16:1/0:0)
        # CHEBI:134604 lysophosphatidylcholine (16:1/0:0)
        # Naming same for both entity with difference of single space
        # Hence, deleting CHEBI:134604 to avoid duplicate entry after camel case conversion
        df = df[df.index != 134604]

        # CHEBI:91309  lysophosphatidylcholine(18:2/0:0)
        # CHEBI:136082 lysophosphatidylcholine (18:2/0:0)
        # Naming same for both entity with difference of single space
        # Hence, deleting CHEBI:136082 to avoid duplicate entry after camel case conversion
        df = df[df.index != 136082]

        df["name"] = df["name"].apply(to_camel_case)

        duplicate_rows = df.loc[df["name"].duplicated(keep=False), ["name"]]
        if not duplicate_rows.empty:
            duplicate_rows["chebi_id"] = duplicate_rows.index
            duplicate_rows = duplicate_rows.sort_values(["name", "chebi_id"])
            print("Found non-unique names after normalization:")
            for name, group in duplicate_rows.groupby("name", sort=False):
                ids = ", ".join(str(chebi_id) for chebi_id in group["chebi_id"])
                print(f"- {name}: CHEBI IDs [{ids}]")

            raise Exception(
                "Duplicate names found after normalization. "
                "Please resolve duplicates before proceeding."
                "Adjust rules in camel case conversion if needed."
            )

        return df

    def get_chebi_id_to_data_mapping_train(self) -> dict[str, dict]:
        df = self._get_chebi_id_to_data_mapping()
        df = df[~df["smiles"].isin(self.validation_smiles)]
        return df.to_dict(orient="index")  # pyright: ignore[reportReturnType]

    def get_chebi_id_to_data_mapping_all(self) -> dict[str, dict]:
        df = self._get_chebi_id_to_data_mapping()
        return df.to_dict(orient="index")  # pyright: ignore[reportReturnType]

    def _get_chebi_id_to_data_mapping(self) -> pd.DataFrame:
        df = self._preprocess_data()
        df["name"] = df["name"].apply(to_camel_case)
        return df

    def _preprocess_data(self) -> pd.DataFrame:
        data_dict = self.process_chebi()
        df = pd.DataFrame.from_dict(data_dict, orient="index")
        df = df[["smiles", "definition", "name"]]
        df["name"] = df["name"].str.lower().str.strip()
        df = df.dropna(subset=["smiles", "definition", "name"])
        return df

    def build_hierarchy_graph(self) -> nx.DiGraph:
        print("Building hierarchy graph from ChEBI data...")
        g = nx.DiGraph()
        chebi = self.process_chebi()
        g.add_nodes_from(chebi.keys())
        for chebi_id, row in chebi.items():
            if "parents" in row:
                for parent in row["parents"]:
                    g.add_edge(parent, chebi_id)
        return g

    def get_topological_ordering(self) -> list[int]:
        """Sort nodes by number of children (out-degree) in descending order.

        Nodes with the most children come first.

        Returns:
            List of node IDs sorted by out-degree (number of children) descending.
        """
        g = self.get_trans_hierarchy()
        print("Sorting nodes by out-degree (number of children) in descending order...")
        # Sort nodes by out-degree (number of children) in descending order
        # The most abstract classes will be first, specific classes will be last
        sorted_nodes = sorted(
            g.nodes(), key=lambda node: g.out_degree(node), reverse=True
        )
        return sorted_nodes

    def get_trans_hierarchy(self):
        if not os.path.exists(self.trans_hierarchy_path):
            g = self.build_hierarchy_graph()
            if not os.path.exists(self.undirected_hierarchy_path):
                with open(self.undirected_hierarchy_path, "wb") as f:
                    pickle.dump(g.to_undirected(), f)
            with open(self.trans_hierarchy_path, "wb") as f:
                pickle.dump(nx.transitive_closure(g), f)
            return g
        with open(self.trans_hierarchy_path, "rb") as f:
            return pickle.load(f)

    def get_undirected_hierarchy_graph(self) -> nx.Graph:
        if not os.path.exists(self.undirected_hierarchy_path):
            g = self.build_hierarchy_graph()
            undirected_g = g.to_undirected()
            with open(self.undirected_hierarchy_path, "wb") as f:
                pickle.dump(undirected_g, f)
            return undirected_g
        with open(self.undirected_hierarchy_path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def chebi_to_int(s: str) -> int:
        """
        Converts a ChEBI term string representation to an integer ID.

        Args:
        - s (str): A ChEBI term string, e.g., "CHEBI:12345".

        Returns:
        - int: The integer ID extracted from the ChEBI term string.
        """
        return int(s[s.index(":") + 1 :])

    @property
    def undirected_hierarchy_path(self):
        return os.path.join(
            self.base_dir, f"chebi_v{self.chebi_version}", "undirected_hierarchy.pkl"
        )


if __name__ == "__main__":
    chebi_data_wrapper = ChEBIDataWrapper(chebi_version=244, validation_smiles=set())
    # Get mappings
    name_to_data_mapping = chebi_data_wrapper.get_name_to_data_mapping_train()
    print(f"\n\nTotal unique names (mapping keys): {len(name_to_data_mapping)}")
    print(f"Sample entries: {list(name_to_data_mapping)[:5]}")
    chebi_id_to_data_mapping = chebi_data_wrapper.get_chebi_id_to_data_mapping_train()
    print(f"Sample entries: {list(chebi_id_to_data_mapping)[:5]}")
    # Get hierarchy graph and topological ordering
    topological_ordering = chebi_data_wrapper.get_topological_ordering()
    print(
        f"Topological ordering of classes (by out-degree) first 10: {topological_ordering[:10]}"
    )
    print(
        f"Topological ordering of classes (by out-degree) last 10: {topological_ordering[-10:]}"
    )
