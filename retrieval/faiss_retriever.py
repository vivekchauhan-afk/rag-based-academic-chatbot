import faiss
import numpy as np
import pickle


class FAISSRetriever:
    """
    Handles FAISS vector search.

    Responsibilities:
    - Load the FAISS index
    - Load document metadata
    - Convert query into an embedding
    - Perform similarity search
    - Return matching document indices
    """

    def __init__(
        self,
        index_path: str,
        metadata_path: str,
        embedding_service
    ):
        print("Loading FAISS vector database...")

        self.index = faiss.read_index(index_path)

        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

        self.embedding_service = embedding_service

        print("FAISS vector database loaded.")
        print(f"Total vectors: {self.index.ntotal}")
        print(f"Total metadata records: {len(self.metadata)}")

    def search(self, query: str, top_k: int = 10):
        """
        Search FAISS for the most similar document chunks.

        Args:
            query:
                User query.

            top_k:
                Number of results to retrieve.

        Returns:
            list[int]:
                Indices of matching document chunks.
        """

        query_vector = self.embedding_service.encode_normalized(
            query
        )

        distances, indices = self.index.search(
            query_vector,
            top_k
        )

        results = []

        for idx in indices[0]:

            if idx == -1:
                continue

            if idx >= len(self.metadata):
                continue

            results.append(int(idx))

        return results