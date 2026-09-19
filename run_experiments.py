import json
import ollama
import time
# NEW IMPORTS FOR RAG
import faiss
import pickle
from sentence_transformers import SentenceTransformer

INPUT_FILE = "processed/qa_pairs_gold.json"
OUTPUT_FILE = "processed/model_outputs.json"

MODELS = [
    "mistral",
    "gemma3:4b"
]
# LOAD EMBEDDING MODEL
embed_model = SentenceTransformer("all-mpnet-base-v2")

#  LOAD FAISS INDEX
index = faiss.read_index("vector_db/faiss_index.bin")

#  LOAD METADATA (CHUNKS)
with open("vector_db/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)


#  RETRIEVE TOP-K CONTEXT
def retrieve_context(question, k=3):
    q_embedding = embed_model.encode([question])
    u,v = index.search(q_embedding, k)

    contexts = []
    for idx in v[0]:
        try:
            # if metadata stored as dict
            contexts.append(metadata[idx]["text"])

        except:

            # fallback if plain text
            contexts.append(metadata[idx])

    return "\n".join(contexts)


#  UPDATED GENERATION FUNCTION (WITH CONTEXT)
def generate_answer(model, question, context):
    prompt = f"""
Answer the question using ONLY the given context.

Context:
{context}

Question:
{question}

Answer:
"""

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"].strip()


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []

    for i, item in enumerate(data):
        print(f"Processing {i+1}/{len(data)}")

        #  SAFE DATA HANDLING
        question = item.get("question", "")
        gold = item.get("gold_answer") or item.get("answer", "")

        #  RETRIEVE CONTEXT FROM FAISS
        context = retrieve_context(question)

        row = {
            "question": question,
            "gold": gold
        }

        for model in MODELS:
            try:
                answer = generate_answer(model, question, context)
                row[model] = answer
                time.sleep(1)

            except Exception as e:
                print(f"{model} error: {e}")
                row[model] = ""

        results.append(row)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print("✅ Model outputs saved!")


if __name__ == "__main__":
    main()