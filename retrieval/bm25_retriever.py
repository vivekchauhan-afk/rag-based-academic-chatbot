from rank_bm25 import BM25Okapi


class BM25Retriever:
    """
    Handles keyword-based BM25 retrieval.

    Responsibilities:
    - Build the BM25 index from document metadata
    - Tokenize the corpus
    - Search for documents relevant to a query
    """

    def __init__(self, metadata):
        """
        Args:
            metadata:
                List of document chunks.
                Each item must contain a 'text' field.
        """

        print("Building BM25 index...")

        self.metadata = metadata

        self.tokenized_corpus = [
            chunk["text"].lower().split()
            for chunk in metadata
        ]

        self.bm25 = BM25Okapi(
            self.tokenized_corpus
        )

        print("BM25 index ready.")

    def search(self, query: str, top_k: int = 10):
        """
        Search the BM25 index.

        Args:
            query:
                User query.

            top_k:
                Number of results to return.

        Returns:
            list[int]:
                Indices of matching document chunks.
        """

        tokenized_query = query.lower().split()

        scores = self.bm25.get_scores(
            tokenized_query
        )

        ranked_indices = scores.argsort()[::-1][:top_k]

        return [
            int(idx)
            for idx in ranked_indices
        ]