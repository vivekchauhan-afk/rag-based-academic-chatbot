import json
import faiss
import pickle
import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# =====================================================
# LOAD CHUNKS
# =====================================================

CHUNK_FILE = "processed/all_chunks.json"

with open(CHUNK_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

if not chunks:
    raise ValueError(
        "No chunks found. Run chunking.py first."
    )

texts = [chunk["text"] for chunk in chunks]

print("=" * 50)
print(f"Loaded {len(texts)} chunks")
print("=" * 50)

# =====================================================
# DEVICE
# =====================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device : {device}")

# =====================================================
# LOAD EMBEDDING MODEL
# =====================================================

print("Loading embedding model...")

model = SentenceTransformer(
    "all-mpnet-base-v2",
    device=device
)

# =====================================================
# CREATE EMBEDDINGS
# =====================================================

print("Generating embeddings...")

embeddings = model.encode(

    texts,

    batch_size=64,

    show_progress_bar=True,

    convert_to_numpy=True,

    normalize_embeddings=True

)

embeddings = embeddings.astype(np.float32)

print("Embeddings Shape :", embeddings.shape)

# =====================================================
# CREATE FAISS INDEX
# =====================================================
eembeddings = np.asarray(embeddings, dtype=np.float32)

dimension = embeddings.shape[1]

print(f"Embedding Dimension: {dimension}")

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

print(f"Vectors Stored: {index.ntotal}")
print(f"Vectors Stored: {index.ntotal}")
# =====================================================
# SAVE DATABASE
# =====================================================

os.makedirs("../vector_db", exist_ok=True)

faiss.write_index(
    index,
    "vector_db/faiss_index.bin"
)

with open(
        "../vector_db/metadata.pkl",
    "wb"
) as f:

    pickle.dump(chunks, f)

print("\n========================================")
print("VECTOR DATABASE CREATED")
print("========================================")
print(f"Chunks      : {len(chunks)}")
print(f"Embeddings  : {index.ntotal}")
print("Index File  : vector_db/faiss_index.bin")
print("Metadata    : vector_db/metadata.pkl")
print("========================================")