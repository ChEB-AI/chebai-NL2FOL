import os
import pickle

import fastobo
import networkx as nx
import pandas as pd
from chebi_utils.downloader import download_chebi_obo
from chebi_utils.obo_extractor import _term_data

from nl_2_fol.inference.preprocessing import CHEBI_ID, SMILES_STRING
from nl_2_fol.inference.utils.to_camel_case import to_camel_case


class ChEBIDataWrapper:
    def __init__(self, chebi_version: int, validation_smiles: set[SMILES_STRING]):
        self.chebi_version = chebi_version
        self.validation_smiles = validation_smiles

        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(
            os.path.join(self.base_dir, f"chebi_v{self.chebi_version}"), exist_ok=True
        )

    def get_name_to_data_mapping_train(self) -> dict[str, dict]:
        df = self._get_name_to_data_mapping()
        df = df[~df["smiles"].isin(self.validation_smiles)]
        return df.set_index("name").to_dict(orient="index")  # pyright: ignore[reportReturnType]

    def get_name_to_data_mapping_all(self) -> dict[str, dict]:
        df = self._get_name_to_data_mapping()
        df["chebi_id"] = df.index
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

    def get_chebi_id_to_data_mapping_train(self) -> dict[CHEBI_ID, dict]:
        df = self._get_chebi_id_to_data_mapping()
        df = df[~df["smiles"].isin(self.validation_smiles)]
        return df.to_dict(orient="index")  # pyright: ignore[reportReturnType]

    def get_chebi_id_to_data_mapping_all(self) -> dict[CHEBI_ID, dict]:
        df = self._get_chebi_id_to_data_mapping()
        return df.to_dict(orient="index")  # pyright: ignore[reportReturnType]

    def _get_chebi_id_to_data_mapping(self) -> pd.DataFrame:
        df = self._preprocess_data()
        df["predicate_name"] = df["name"].apply(to_camel_case)
        return df

    def _preprocess_data(self) -> pd.DataFrame:
        data_dict = self.process_chebi()
        df = pd.DataFrame.from_dict(data_dict, orient="index")
        df = df[["smiles", "definition", "name", "parents"]]
        df["name"] = df["name"].str.lower().str.strip()
        df = df.dropna(subset=["definition", "name"])
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
                g = nx.transitive_closure(g)
                pickle.dump(g, f)
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

    @property
    def base_dir(self):
        return "data"

    @property
    def chebi_path(self):
        return os.path.join(self.base_dir, f"chebi_v{self.chebi_version}", "chebi.obo")

    @property
    def chebi_dict_path(self):
        return os.path.join(
            self.base_dir, f"chebi_v{self.chebi_version}", "chebi_dict.pkl"
        )

    @property
    def trans_hierarchy_path(self):
        return os.path.join(
            self.base_dir, f"chebi_v{self.chebi_version}", "trans_hierarchy.pkl"
        )

    @property
    def processed_path(self):
        return os.path.join(
            self.base_dir, f"chebi_v{self.chebi_version}", "processed.pkl"
        )

    def download_chebi(self) -> None:
        if not os.path.exists(self.chebi_path):
            download_chebi_obo(self.chebi_version, os.path.dirname(self.chebi_path))

    def process_chebi(self) -> dict:
        self.download_chebi()
        if not os.path.exists(self.chebi_dict_path):
            with open(self.chebi_path, encoding="utf-8") as chebi_raw:
                chebi = "\n".join(
                    line for line in chebi_raw if not line.startswith("xref:")
                )
            res = {}
            for term in fastobo.loads(chebi):  # type: ignore
                if (
                    term
                    and ":" in str(term.id)
                    and not any(
                        [
                            clause.raw_tag() == "is_obsolete"
                            and clause.raw_value() == "true"
                            for clause in term
                        ]
                    )
                ):
                    term = _term_data(term)
                    if term is None:
                        continue
                    chebi_id = int(term.pop("id"))
                    res[chebi_id] = term
            with open(self.chebi_dict_path, "wb") as f:
                pickle.dump(res, f)
            return res
        else:
            with open(self.chebi_dict_path, "rb") as f:
                return pickle.load(f)


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
