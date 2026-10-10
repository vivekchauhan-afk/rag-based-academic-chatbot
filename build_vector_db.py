import json
import os
import pickle

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


# =====================================================
# CONFIGURATION
# =====================================================

CHUNK_FILE = "processed/all_chunks.json"

VECTOR_DB_FOLDER = "vector_db"

FAISS_INDEX_FILE = os.path.join(
    VECTOR_DB_FOLDER,
    "faiss_index.bin"
)

METADATA_FILE = os.path.join(
    VECTOR_DB_FOLDER,
    "metadata.pkl"
)

EMBEDDING_MODEL = "all-mpnet-base-v2"

BATCH_SIZE = 64


# =====================================================
# LOAD CHUNKS
# =====================================================

if not os.path.exists(CHUNK_FILE):

    raise FileNotFoundError(
        f"Chunk file not found: {CHUNK_FILE}\n"
        "Run ingestion/chunk_data.py first."
    )


with open(
    CHUNK_FILE,
    "r",
    encoding="utf-8"
) as f:

    chunks = json.load(f)


if not chunks:

    raise ValueError(
        "No chunks found. "
        "Run ingestion/chunk_data.py first."
    )


texts = [
    chunk["text"]
    for chunk in chunks
]


print("=" * 50)
print(f"Loaded {len(texts)} chunks")
print("=" * 50)


# =====================================================
# DEVICE
# =====================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Using device: {device}")


# =====================================================
# LOAD EMBEDDING MODEL
# =====================================================

print(
    f"Loading embedding model: {EMBEDDING_MODEL}"
)

model = SentenceTransformer(
    EMBEDDING_MODEL,
    device=device
)

print("Embedding model loaded.")


# =====================================================
# CREATE EMBEDDINGS
# =====================================================

print("Generating embeddings...")

embeddings = model.encode(

    texts,

    batch_size=BATCH_SIZE,

    show_progress_bar=True,

    convert_to_numpy=True,

    normalize_embeddings=True
)


# =====================================================
# CONVERT TO FLOAT32
# =====================================================

embeddings = np.asarray(
    embeddings,
    dtype=np.float32
)


print(
    f"Embeddings shape: {embeddings.shape}"
)


# =====================================================
# CREATE FAISS INDEX
# =====================================================

dimension = embeddings.shape[1]

print(
    f"Embedding dimension: {dimension}"
)


# IndexFlatIP is appropriate because
# embeddings are normalized.
index = faiss.IndexFlatIP(
    dimension
)


index.add(
    embeddings
) # pyright: ignore[reportCallIssue]


print(
    f"Vectors stored: {index.ntotal}"
)


# =====================================================
# CREATE VECTOR DB DIRECTORY
# =====================================================

os.makedirs(
    VECTOR_DB_FOLDER,
    exist_ok=True
)


# =====================================================
# SAVE FAISS INDEX
# =====================================================

faiss.write_index(
    index,
    FAISS_INDEX_FILE
)


# =====================================================
# SAVE METADATA
# =====================================================

with open(
    METADATA_FILE,
    "wb"
) as f:

    pickle.dump(
        chunks,
        f
    )


# =====================================================
# VERIFY DATABASE
# =====================================================

print("\n========================================")
print("VECTOR DATABASE CREATED")
print("========================================")

print(
    f"Chunks       : {len(chunks)}"
)

print(
    f"Embeddings   : {index.ntotal}"
)

print(
    f"Dimension    : {dimension}"
)

print(
    f"Index File   : {FAISS_INDEX_FILE}"
)

print(
    f"Metadata     : {METADATA_FILE}"
)

print("========================================")