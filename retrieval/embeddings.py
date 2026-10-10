from sentence_transformers import SentenceTransformer
import numpy as np
import faiss


class EmbeddingService:
    """
    Handles text embedding generation.

    This class is responsible only for:
    - Loading the embedding model
    - Converting text into embeddings
    - Normalizing embeddings for cosine similarity search
    """

    def __init__(
        self,
        model_name: str = "all-mpnet-base-v2"
    ):
        print("Loading embedding model...")

        self.model = SentenceTransformer(model_name)

        print("Embedding model loaded.")

    def encode(self, texts):
        """
        Convert one or more texts into embeddings.

        Args:
            texts:
                A string or a list of strings.

        Returns:
            np.ndarray:
                Float32 embedding matrix.
        """

        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True
        )

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32
        )

        return embeddings

    def encode_normalized(self, texts):
        """
        Generate normalized embeddings.

        Normalization allows inner-product search in FAISS
        to behave like cosine similarity search.
        """

        embeddings = self.encode(texts)

        faiss.normalize_L2(embeddings)

        return embeddings