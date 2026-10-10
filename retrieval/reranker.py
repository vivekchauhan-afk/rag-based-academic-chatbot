from sentence_transformers import CrossEncoder


class DocumentReranker:
    """
    Reranks retrieved document chunks using a CrossEncoder.

    The retriever finds candidate documents using
    FAISS + BM25.

    The reranker then compares:

        query ↔ document chunk

    and assigns a relevance score.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        print("Loading reranker model...")

        self.model = CrossEncoder(
            model_name
        )

        print("Reranker model loaded.")

    def rerank(
        self,
        query: str,
        indices: list[int],
        metadata: list[dict]
    ):
        """
        Rerank retrieved document indices.

        Args:
            query:
                Original user question.

            indices:
                Candidate document indices returned
                by the hybrid retriever.

            metadata:
                Document metadata.

        Returns:
            list[dict]:
                Reranked document metadata with scores.
        """

        candidates = []

        for idx in indices:

            if idx < 0:
                continue

            if idx >= len(metadata):
                continue

            candidates.append(
                metadata[idx]
            )

        if not candidates:
            return []

        pairs = [
            [query, document["text"]]
            for document in candidates
        ]

        scores = self.model.predict(
            pairs
        )

        reranked_documents = []

        for document, score in zip(
            candidates,
            scores
        ):

            # Make a copy so we don't modify
            # the original metadata.
            document_copy = document.copy()

            document_copy["score"] = float(
                score
            )

            reranked_documents.append(
                document_copy
            )

        reranked_documents.sort(
            key=lambda document: document["score"],
            reverse=True
        )

        return reranked_documents