import faiss
import pickle
import numpy as np
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
from collections import defaultdict
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

# =====================================================
# LOAD FILE URLS
# =====================================================

try:
    with open(
        "processed/file_urls.json",
        "r",
        encoding="utf-8"
    ) as f:

        file_urls = json.load(f)

except FileNotFoundError:

    print("Warning: file_urls.json not found.")

    file_urls = []
# -----------------------------
# CHAT MEMORY
# -----------------------------

chat_history = []

# -----------------------------
# LOAD MODELS
# -----------------------------

print("Loading embedding model...")
embedding_model = SentenceTransformer("all-mpnet-base-v2")

print("Loading reranker model...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# -----------------------------
# LOAD VECTOR DATABASE
# -----------------------------

print("Loading vector database...")

index = faiss.read_index("vector_db/faiss_index.bin")

with open("vector_db/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)


# -----------------------------
# BUILD BM25 INDEX
# -----------------------------

tokenized_corpus = [
    chunk["text"].lower().split()
    for chunk in metadata
]

bm25 = BM25Okapi(tokenized_corpus)

print("AI Assistant Ready!\n")

# -----------------------------
# INTENT KEYWORDS
# -----------------------------

GREETINGS = {
    "hi",
    "hello",
    "hii",
    "hey"
}



FESTIVALS = {
    "holi",
    "diwali",
    "ram navami",
    "eid",
    "bakrid",
    "christmas",
    "independence day",
    "republic day"
}

def expand_query(query):
    query = query.lower()

    expansions = {

        "calendar":
            "academic calendar semester schedule holiday dates",

        "syllabus":
            "course syllabus curriculum subjects units",

        "attendance":
            "attendance policy minimum attendance rules",

        "exam":
            "mid semester end semester examination schedule",

        "placement":
            "training placement recruitment internship company",

        "registration":
            "registration admission apply form",

        "hostel":
            "hostel accommodation mess rooms",

        "fees":
            "fee structure tuition hostel fee payment",

        "holiday":
            "festival holiday academic calendar",

        "festival":
            "festival holiday academic calendar celebration",

        "holi":
            "holi holiday academic calendar",

        "diwali":
            "diwali deepawali holiday academic calendar",

        "ram navami":
            "ram navami holiday academic calendar",

        "eid":
            "eid holiday academic calendar",

        "bakrid":
            "bakrid holiday academic calendar",

        "library":
            "central library books journals digital library",

        "faculty":
            "staff professor department faculty",

        "result":
            "exam result grades marks"

    }

    expanded = query

    for key, value in expansions.items():

        if key in query:

            expanded += " " + value

    return expanded


# -----------------------------
# INTENT DETECTION
# -----------------------------

def detect_intent(question):

    q = question.lower().strip()

    # Greetings
    if any(word in q for word in GREETINGS):
        return "greeting"

    # Download requests
    if any(word in q for word in [
        "download",
        "pdf",
        "document",
        "file",
        "link",
        "brochure"
    ]):
        return "download"

    # Faculty related
    if any(word in q for word in [
        "director",
        "hod",
        "head of department",
        "dean",
        "registrar",
        "faculty",
        "professor",
        "assistant professor",
        "associate professor"
    ]):
        return "faculty"

    # Academic Calendar
    if any(word in q for word in [
        "academic calendar",
        "calendar",
        "holiday",
        "semester schedule",
        "odd semester",
        "even semester"
    ]):
        return "calendar"

    # Syllabus
    if any(word in q for word in [
        "syllabus",
        "curriculum",
        "scheme",
        "course structure",
        "subject"
    ]):
        return "syllabus"

    # Placement
    if any(word in q for word in [
        "placement",
        "package",
        "recruiter",
        "company",
        "internship",
        "training and placement"
    ]):
        return "placement"

    # Hostel
    if any(word in q for word in [
        "hostel",
        "mess",
        "room",
        "accommodation",
        "warden"
    ]):
        return "hostel"

    # Admission
    if any(word in q for word in [
        "admission",
        "apply",
        "eligibility",
        "jee",
        "josaa"
    ]):
        return "admission"

    return "information"


def get_top_k(question):

    q = question.lower()

    if is_festival_query(q):
        return 20

    elif "calendar" in q:
        return 20

    elif "syllabus" in q:
        return 15

    elif "placement" in q:
        return 15

    elif "attendance" in q:
        return 12

    elif len(q.split()) <= 4:
        return 8

    return 12

CONFIDENCE_THRESHOLD = 0.15

# -----------------------------
# HYBRID SEARCH
# -----------------------------

# -----------------------------
# HYBRID SEARCH (FAISS + BM25 + RRF)
# -----------------------------

def hybrid_search(question):

    expanded_question = expand_query(question)

    top_k = get_top_k(question)

    # ------------------------------------
    # FAISS SEARCH
    # ------------------------------------

    query_vector = embedding_model.encode(
        [expanded_question]
    )

    query_vector = np.array(
        query_vector,
        dtype=np.float32
    )

    faiss.normalize_L2(query_vector)

    _, faiss_indices = index.search(
        query_vector,
        top_k
    )

    faiss_results = [
        idx for idx in faiss_indices[0]
        if idx != -1
    ]

    # ------------------------------------
    # BM25 SEARCH
    # ------------------------------------

    tokenized_query = expanded_question.lower().split()

    bm25_scores = bm25.get_scores(
        tokenized_query
    )

    bm25_results = list(
        np.argsort(bm25_scores)[::-1][:top_k]
    )

    # ------------------------------------
    # RECIPROCAL RANK FUSION
    # ------------------------------------

    fusion_scores = defaultdict(float)

    k = 60

    for rank, idx in enumerate(faiss_results):

        fusion_scores[idx] += 1 / (k + rank)

    for rank, idx in enumerate(bm25_results):

        fusion_scores[idx] += 1 / (k + rank)

    # ------------------------------------
    # DOCUMENT BOOSTING
    # ------------------------------------

    important_keywords = {

        "calendar": 0.30,
        "academic": 0.30,
        "syllabus": 0.30,
        "placement": 0.30,
        "hostel": 0.25,
        "admission": 0.25,
        "faculty": 0.20,
        "library": 0.20,
        "fee": 0.20,
        "attendance": 0.20
    }

    question_lower = question.lower()

    for idx in fusion_scores:

        text = metadata[idx]["text"].lower()

        for keyword, boost in important_keywords.items():

            if keyword in question_lower and keyword in text:

                fusion_scores[idx] += boost

    # ------------------------------------
    # SORT
    # ------------------------------------

    ranked = sorted(

        fusion_scores.items(),

        key=lambda x: x[1],

        reverse=True

    )

    final_results = [

        idx

        for idx, score in ranked

    ]

    return final_results
def is_festival_query(question):

    q = question.lower()

    return any(
        festival in q
        for festival in FESTIVALS
    )

# -----------------------------
# RERANK DOCUMENTS
# -----------------------------

def rerank_documents(question, indices):

    candidates = []

    for idx in indices:

        if idx >= len(metadata):
            continue

        candidates.append(metadata[idx])

    pairs = [
        [question, doc["text"]]
        for doc in candidates
    ]

    scores = reranker.predict(pairs)

    for i in range(len(candidates)):
        candidates[i]["score"] = float(scores[i])

    candidates = sorted(
        candidates,
        key=lambda x: x["score"],
        reverse=True
    )

    return candidates

# -----------------------------
# FILTER LOW CONFIDENCE
# -----------------------------

def filter_documents(candidates):

    if not candidates:
        return []

    # Always keep Top-5 reranked documents
    return candidates[:5]

# -----------------------------
# BUILD CONTEXT
# -----------------------------

# -----------------------------
# BUILD CONTEXT
# -----------------------------

def build_context(documents):

    context_parts = []

    sources = []

    used_text = set()

    source_count = defaultdict(int)

    max_source_per_file = 3
    max_content_chunks = 8

    for doc in documents:

        text = doc["text"].strip()

        # Skip very short chunks
        if len(text) < 80:
            continue

        # Remove duplicate chunks
        text_key = text[:300].lower()

        if text_key in used_text:
            continue

        used_text.add(text_key)

        # Maximum chunks from one PDF
        if source_count[doc["source"]] >= max_source_per_file:
            continue

        source_count[doc["source"]] += 1

        context_parts.append(text)

        sources.append({

            "source": doc["source"],

            "page": doc["page"],

            "score": round(doc["score"], 3),

            "url": doc.get("url", "")

        })

        if len(context_parts) >= max_content_chunks:
            break

    # Merge context
    context = "\n\n".join(context_parts)

    return context, sources
# -----------------------------
# COMPLETE RETRIEVAL PIPELINE
# -----------------------------

def retrieve_documents(question):

    indices = hybrid_search(question)

    reranked = rerank_documents(
        question,
        indices
    )

    if is_festival_query(question):

        # Retrieve more documents for festivals
        reranked = reranked[:10]


    filtered = filter_documents(reranked)

    if not filtered:
        return "", [], 0

    context, sources = build_context(filtered)

    max_score = 0

    if len(filtered):
        max_score = filtered[0]["score"]

    return context, sources, max_score
# ==============================
# LOAD FINETUNED GEMMA MODEL
# ==============================

BASE_MODEL = "google/gemma-2-2b-it"
MODEL_PATH = "best_models/gemma/fold_5"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)



bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto"
)

model = PeftModel.from_pretrained(
    base_model,
    MODEL_PATH
)
# -----------------------------
# MAIN QUESTION ANSWERING
# -----------------------------

def ask_question(question):

    question = question.strip()

    intent = detect_intent(question)

    # =====================================================
    # GREETING
    # =====================================================

    if intent == "greeting":

        return {

            "answer":
                "Hello! 👋\n\n"
                "I can help you with:\n"
                "• Academic Calendar\n"
                "• Syllabus\n"
                "• Placement\n"
                "• Hostel\n"
                "• Admissions\n"
                "• Faculty\n"
                "• Downloads\n"
                ,

            "sources": [],

            "downloads": []



        }

    # =====================================================
    # DOWNLOAD SEARCH
    # =====================================================

    downloads = []

    if intent == "download":

        query = question.lower()

        # Remove common words
        stop_words = {
            "give", "me", "the", "a", "an", "of",
            "for", "to", "please", "download",
            "pdf", "document", "file", "link"
        }

        keywords = [
            word
            for word in query.split()
            if word not in stop_words
        ]

        matches = []

        for file in file_urls:

            filename = file["filename"].lower()

            score = 0

            for keyword in keywords:

                if keyword.replace("-", " ") in filename.replace("-", " "):
                    score += 1

            if score > 0:
                matches.append((score, file))

        # Sort by best match
        matches.sort(key=lambda x: x[0], reverse=True)

        seen = set()

        for _, file in matches:

            if file["url"] not in seen:
                downloads.append(file["url"])

                seen.add(file["url"])

    # =====================================================
    # RETURN DOWNLOAD DIRECTLY
    # =====================================================

    if downloads:
        return {

            "answer": "I found the requested document.",

            "sources": [],

            "downloads": downloads

        }
    # =====================================================
    # RETRIEVAL
    # =====================================================

    context, sources, max_score = retrieve_documents(question)
    print("\n" + "=" * 80)
    print("QUESTION:", question)
    print("INTENT:", intent)
    print("SCORE:", round(max_score, 3))

    print("\nRETRIEVED CONTEXT:")
    print(context)

    print("\nSOURCES:")
    for source in sources:
        print(source)

    print("=" * 80 + "\n")

    if max_score < CONFIDENCE_THRESHOLD:
        return {
            "answer":
                "I couldn't find reliable information related to your question in the available documents.",
            "sources": [],
            "downloads": downloads
        }

    # =====================================================
    # PROMPT
    # =====================================================

    prompt = f"""
    You are an AI Assistant for National Institute of Technology Kurukshetra.

    Your task is to answer the question ONLY from the Retrieved Context.

    STRICT RULES:
    1. Do not use pretrained knowledge.
    2. Do not use information from outside the Retrieved Context.
    3. Do not guess or hallucinate.
    4. For names, dates, numbers, designations, and official positions, use the exact information stated in the Retrieved Context.
    5. If the Retrieved Context contains a person explicitly associated with the requested position, return that person's name.
    6. Do not replace the retrieved person's name with another person.
    7. If the answer cannot be found in the Retrieved Context, say exactly:
    "I could not find this information in the available documents."

    Retrieved Context:
    {context}

    Question:
    {question}

    Answer:
    """

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()


    # =====================================================
    # RETURN
    # =====================================================

    return {

        "answer": answer,

        "sources": sources,

        "downloads": downloads

    }
