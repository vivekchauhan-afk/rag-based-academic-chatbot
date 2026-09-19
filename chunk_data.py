import json
import os
import re

# =====================================================
# CONFIGURATION
# =====================================================

MAX_WORDS = 300
MIN_WORDS = 120
OVERLAP_SENTENCES = 1

INPUT_JSON = "processed/extracted_raw.json"
OUTPUT_JSON = "processed/all_chunks.json"
# =====================================================
# LOAD DOCUMENTS
# =====================================================

with open(INPUT_JSON, "r", encoding="utf-8") as f:
    documents = json.load(f)

print(f"Loaded {len(documents)} pages.")

# =====================================================
# CLEAN TEXT
# =====================================================

def clean_text(text):

    if not text:
        return ""

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# =====================================================
# SPLIT PARAGRAPHS
# =====================================================

def split_paragraphs(text):

    paragraphs = re.split(r"\n\s*\n", text)

    paragraphs = [
        clean_text(p)
        for p in paragraphs
        if clean_text(p)
    ]

    return paragraphs


# =====================================================
# SPLIT SENTENCES
# =====================================================

def split_sentences(text):

    sentences = re.split(
        r'(?<=[.!?])\s+',
        text
    )

    return [
        s.strip()
        for s in sentences
        if s.strip()
    ]


# =====================================================
# REMOVE PAGE HEADINGS
# =====================================================

def remove_page_markers(text):

    text = re.sub(
        r"=+\s*PAGE\s*\d+\s*=+",
        "",
        text,
        flags=re.IGNORECASE
    )

    return text.strip()


# =====================================================
# CREATE SEMANTIC CHUNKS
# =====================================================

chunks = []

chunk_id = 1

for doc in documents:

    source = doc["source"]
    page = doc["page"]
    url = doc.get("url", "")

    text = remove_page_markers(doc["text"])

    paragraphs = split_paragraphs(text)

    current_chunk = ""
    current_sentences = []

    for paragraph in paragraphs:

        # ------------------------------------------------
        # TABLE CHUNK
        # ------------------------------------------------

        if "===== TABLE DATA =====" in paragraph:

            if len(current_chunk.split()) >= MIN_WORDS:

                chunks.append({

                    "id": chunk_id,

                    "text": current_chunk.strip(),

                    "source": source,

                    "page": page,

                    "url": url

                })

                chunk_id += 1

            chunks.append({

                "id": chunk_id,

                "text": paragraph.strip(),

                "source": source,

                "page": page,

                "url": url

            })

            chunk_id += 1

            current_chunk = ""
            current_sentences = []

            continue

        # ------------------------------------------------
        # NORMAL PARAGRAPH
        # ------------------------------------------------

        sentences = split_sentences(paragraph)

        for sentence in sentences:

            temp_chunk = current_chunk + " " + sentence

            if len(temp_chunk.split()) <= MAX_WORDS:

                current_chunk = temp_chunk.strip()

                current_sentences.append(sentence)

            else:

                if len(current_chunk.split()) >= MIN_WORDS:

                    chunks.append({

                        "id": chunk_id,

                        "text": current_chunk.strip(),

                        "source": source,

                        "page": page,

                        "url": url

                    })

                    chunk_id += 1

                overlap = current_sentences[-OVERLAP_SENTENCES:]

                current_chunk = " ".join(overlap)

                current_chunk += " " + sentence

                current_sentences = overlap + [sentence]

    # ------------------------------------------------
    # LAST CHUNK
    # ------------------------------------------------

    if len(current_chunk.split()) >= MIN_WORDS:

        chunks.append({

            "id": chunk_id,

            "text": current_chunk.strip(),

            "source": source,

            "page": page,

            "url": url

        })

        chunk_id += 1


# =====================================================
# REMOVE DUPLICATE CHUNKS
# =====================================================

print("Removing duplicate chunks...")

unique_chunks = []

seen = set()

for chunk in chunks:

    key = " ".join(
        chunk["text"].lower().split()[:60]
    )

    if key in seen:
        continue

    seen.add(key)

    unique_chunks.append(chunk)

chunks = unique_chunks

print(f"Chunks after duplicate removal : {len(chunks)}")


# =====================================================
# SAVE CHUNKS
# =====================================================

os.makedirs("../processed", exist_ok=True)

with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        chunks,
        f,
        indent=4,
        ensure_ascii=False
    )

print("\n========================================")
print("SMART CHUNKING COMPLETED")
print("========================================")
print(f"Total Chunks : {len(chunks)}")
print(f"Output File  : {OUTPUT_JSON}")
print("========================================") 