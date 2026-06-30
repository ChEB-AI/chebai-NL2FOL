import numpy as np
from sentence_transformers import SentenceTransformer


class IncrementalSemanticRetriever:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is not None:
            raise Exception(f"{cls.__name__} instance already exists")

        cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.predicates = []
        self.embeddings = None

    def add_predicate(self, predicate: str):
        if predicate in self.predicates:
            raise ValueError(
                f"Predicate '{predicate}' already exists in the retriever."
            )
        new_embedding = self.model.encode([predicate], normalize_embeddings=True)

        self.predicates.append(predicate)

        if self.embeddings is None:
            self.embeddings = new_embedding
        else:
            self.embeddings = np.vstack([self.embeddings, new_embedding])

    def add_predicates(self, predicates: list[str]):
        if len(set(predicates)) != len(predicates):
            raise ValueError("Duplicate predicates found.")

        if any(predicate in self.predicates for predicate in predicates):
            raise ValueError("One or more predicates already exist in the retriever.")

        new_embeddings = self.model.encode(predicates, normalize_embeddings=True)

        self.predicates.extend(predicates)

        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

    def retrieve_relevant_predicates(
        self, text: str, top_k: int | None = None, threshold=0.35
    ) -> list[str]:
        if self.embeddings is None or len(self.predicates) == 0:
            return []

        if top_k is not None and (top_k <= 0):
            raise ValueError(
                f"top_k must be a positive integer less than or equal to the number of predicates ({len(self.predicates)})."
            )
        if top_k is not None and top_k > len(self.predicates):
            return [predicate for predicate in self.predicates]

        query_embedding = self.model.encode(text, normalize_embeddings=True)

        scores = np.dot(self.embeddings, query_embedding)
        ranked_indices = np.argsort(scores)[::-1]

        results = []

        if top_k is not None:
            ranked_indices = ranked_indices[:top_k]

        for idx in ranked_indices:
            score = float(scores[idx])

            if score >= threshold:
                results.append(self.predicates[idx])
        return results


if __name__ == "__main__":
    retriever = IncrementalSemanticRetriever()
    retriever.add_predicates(
        ["refund policy", "shipping delay", "account cancellation"]
    )
    retriever.add_predicate("delete user account")
    matches = retriever.retrieve_relevant_predicates(
        "The user wants to permanently close their account.", top_k=3, threshold=0.35
    )
    print(matches)
