from collections import defaultdict


class ContextBuilder:
    """
    Builds the final context that will be sent to the LLM.

    Responsibilities:
    - Remove duplicate chunks
    - Limit chunks from the same source
    - Limit total context chunks
    - Prepare source information
    """

    def __init__(
        self,
        max_source_per_file: int = 3,
        max_content_chunks: int = 8,
        min_text_length: int = 80
    ):
        self.max_source_per_file = max_source_per_file
        self.max_content_chunks = max_content_chunks
        self.min_text_length = min_text_length

    def build(self, documents):
        """
        Build context and source information.

        Args:
            documents:
                Reranked document chunks.

        Returns:
            tuple:
                context, sources, max_score
        """

        context_parts = []
        sources = []

        used_text = set()
        source_count = defaultdict(int)

        for document in documents:

            text = document["text"].strip()

            # ---------------------------------------------
            # Skip very short chunks
            # ---------------------------------------------

            if len(text) < self.min_text_length:
                continue

            # ---------------------------------------------
            # Remove duplicate chunks
            # ---------------------------------------------

            text_key = text[:300].lower()

            if text_key in used_text:
                continue

            used_text.add(text_key)

            # ---------------------------------------------
            # Limit chunks from the same source
            # ---------------------------------------------

            source = document.get(
                "source",
                "Unknown"
            )

            if (
                source_count[source]
                >= self.max_source_per_file
            ):
                continue

            source_count[source] += 1

            # ---------------------------------------------
            # Add context
            # ---------------------------------------------

            context_parts.append(text)

            # ---------------------------------------------
            # Add source information
            # ---------------------------------------------

            sources.append({
                "source": source,
                "page": document.get("page", ""),
                "score": round(
                    document.get("score", 0),
                    3
                ),
                "url": document.get(
                    "url",
                    ""
                )
            })

            # ---------------------------------------------
            # Maximum number of chunks
            # ---------------------------------------------

            if (
                len(context_parts)
                >= self.max_content_chunks
            ):
                break

        # ---------------------------------------------
        # Combine chunks
        # ---------------------------------------------

        context = "\n\n".join(
            context_parts
        )

        # ---------------------------------------------
        # Highest reranker score
        # ---------------------------------------------

        max_score = 0.0

        if documents:
            max_score = float(
                documents[0].get(
                    "score",
                    0
                )
            )

        return (
            context,
            sources,
            max_score
        )