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

if not os.path.exists(INPUT_JSON):

    raise FileNotFoundError(
        f"Input file not found: {INPUT_JSON}\n"
        "Run ingestion/extract_data.py first."
    )


with open(
    INPUT_JSON,
    "r",
    encoding="utf-8"
) as f:

    documents = json.load(f)


if not documents:

    raise ValueError(
        "No documents found in extracted_raw.json."
    )


print(
    f"Loaded {len(documents)} pages."
)


# =====================================================
# CLEAN TEXT
# =====================================================

def clean_text(text):

    if not text:
        return ""

    # Remove null characters
    text = text.replace("\x00", " ")

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =====================================================
# SPLIT PARAGRAPHS
# =====================================================

def split_paragraphs(text):

    if not text:
        return []

    # Split on blank lines
    paragraphs = re.split(
        r"\n\s*\n",
        text
    )

    cleaned_paragraphs = []

    for paragraph in paragraphs:

        paragraph = clean_text(
            paragraph
        )

        if paragraph:

            cleaned_paragraphs.append(
                paragraph
            )

    return cleaned_paragraphs


# =====================================================
# SPLIT SENTENCES
# =====================================================

def split_sentences(text):

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


# =====================================================
# REMOVE PAGE MARKERS
# =====================================================

def remove_page_markers(text):

    if not text:
        return ""

    # Handles common page-marker formats
    text = re.sub(
        r"=+\s*PAGE\s*\d+\s*=+",
        "",
        text,
        flags=re.IGNORECASE
    )

    return text.strip()


# =====================================================
# CREATE CHUNK OBJECT
# =====================================================

def create_chunk(
    chunk_id,
    text,
    source,
    page,
    url
):

    return {
        "id": chunk_id,
        "text": text.strip(),
        "source": source,
        "page": page,
        "url": url
    }


# =====================================================
# CREATE SEMANTIC CHUNKS
# =====================================================

chunks = []

chunk_id = 1


for doc_number, doc in enumerate(
    documents,
    start=1
):

    source = doc.get(
        "source",
        "Unknown"
    )

    page = doc.get(
        "page",
        ""
    )

    url = doc.get(
        "url",
        ""
    )

    text = remove_page_markers(
        doc.get(
            "text",
            ""
        )
    )

    if not text:
        continue


    paragraphs = split_paragraphs(
        text
    )

    current_chunk = ""

    current_sentences = []


    for paragraph in paragraphs:

        # =================================================
        # TABLE CHUNK
        # =================================================

        # extract_data.py creates:
        #
        # TABLE DATA
        #
        # Therefore we detect that exact marker.

        if paragraph.startswith(
            "TABLE DATA"
        ):

            # Save the current normal chunk first
            if current_chunk:

                word_count = len(
                    current_chunk.split()
                )

                if word_count >= MIN_WORDS:

                    chunks.append(
                        create_chunk(
                            chunk_id,
                            current_chunk,
                            source,
                            page,
                            url
                        )
                    )

                    chunk_id += 1

                elif word_count > 0:

                    # Preserve short content instead
                    # of silently discarding it.
                    chunks.append(
                        create_chunk(
                            chunk_id,
                            current_chunk,
                            source,
                            page,
                            url
                        )
                    )

                    chunk_id += 1


            # Save table as its own chunk
            table_text = paragraph.strip()

            if len(table_text.split()) > 0:

                chunks.append(
                    create_chunk(
                        chunk_id,
                        table_text,
                        source,
                        page,
                        url
                    )
                )

                chunk_id += 1


            # Reset normal chunk state
            current_chunk = ""
            current_sentences = []

            continue


        # =================================================
        # NORMAL PARAGRAPH
        # =================================================

        sentences = split_sentences(
            paragraph
        )


        for sentence in sentences:

            sentence = sentence.strip()

            if not sentence:
                continue


            # -------------------------------------------------
            # EMPTY CURRENT CHUNK
            # -------------------------------------------------

            if not current_chunk:

                current_chunk = sentence

                current_sentences = [
                    sentence
                ]

                continue


            # -------------------------------------------------
            # TEST NEW SENTENCE
            # -------------------------------------------------

            candidate = (
                current_chunk
                + " "
                + sentence
            )

            candidate_word_count = len(
                candidate.split()
            )


            # -------------------------------------------------
            # SENTENCE FITS
            # -------------------------------------------------

            if candidate_word_count <= MAX_WORDS:

                current_chunk = candidate

                current_sentences.append(
                    sentence
                )

                continue


            # -------------------------------------------------
            # CURRENT CHUNK IS FULL
            # -------------------------------------------------

            current_word_count = len(
                current_chunk.split()
            )


            # Save current chunk
            if current_word_count > 0:

                chunks.append(
                    create_chunk(
                        chunk_id,
                        current_chunk,
                        source,
                        page,
                        url
                    )
                )

                chunk_id += 1


            # -------------------------------------------------
            # CREATE OVERLAP
            # -------------------------------------------------

            overlap = (
                current_sentences[
                    -OVERLAP_SENTENCES:
                ]
                if OVERLAP_SENTENCES > 0
                else []
            )


            current_chunk = " ".join(
                overlap
            )


            # -------------------------------------------------
            # HANDLE VERY LONG SENTENCE
            # -------------------------------------------------

            if len(sentence.split()) > MAX_WORDS:

                # If a single sentence is larger than
                # MAX_WORDS, split it by words so that
                # the maximum size remains controlled.

                words = sentence.split()

                while len(words) > MAX_WORDS:

                    part = " ".join(
                        words[:MAX_WORDS]
                    )

                    chunks.append(
                        create_chunk(
                            chunk_id,
                            part,
                            source,
                            page,
                            url
                        )
                    )

                    chunk_id += 1

                    words = words[
                        MAX_WORDS:
                    ]

                current_chunk = " ".join(
                    words
                )

                current_sentences = [
                    current_chunk
                ]

            else:

                if current_chunk:

                    current_chunk += " "

                current_chunk += sentence

                current_sentences = (
                    overlap
                    + [sentence]
                )


    # =================================================
    # SAVE LAST CHUNK OF PAGE
    # =================================================

    if current_chunk:

        final_word_count = len(
            current_chunk.split()
        )

        if final_word_count > 0:

            chunks.append(
                create_chunk(
                    chunk_id,
                    current_chunk,
                    source,
                    page,
                    url
                )
            )

            chunk_id += 1

# =====================================================
# REMOVE DUPLICATE CHUNKS
# =====================================================

print(
    "\nRemoving duplicate chunks..."
)

unique_chunks = []

seen = {}


for chunk in chunks:

    # Normalize the entire chunk for
    # duplicate comparison.
    normalized_text = " ".join(
        chunk["text"]
        .lower()
        .split()
    )

    if not normalized_text:
        continue

    key = normalized_text


    # -------------------------------------------------
    # FIRST OCCURRENCE
    # -------------------------------------------------

    if key not in seen:

        seen[key] = len(unique_chunks)

        unique_chunks.append(
            chunk
        )

        continue


    # -------------------------------------------------
    # DUPLICATE FOUND
    # -------------------------------------------------

    existing_index = seen[key]

    existing_chunk = unique_chunks[
        existing_index
    ]


    # -------------------------------------------------
    # PREFER CHUNK WITH URL
    # -------------------------------------------------

    existing_url = existing_chunk.get(
        "url",
        ""
    )

    current_url = chunk.get(
        "url",
        ""
    )


    if not existing_url and current_url:

        unique_chunks[
            existing_index
        ] = chunk


chunks = unique_chunks

# =====================================================
# SAVE CHUNKS
# =====================================================

os.makedirs(
    "processed",
    exist_ok=True
)


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


# =====================================================
# STATISTICS
# =====================================================

total_words = sum(
    len(chunk["text"].split())
    for chunk in chunks
)

average_words = (
    total_words / len(chunks)
    if chunks
    else 0
)


# =====================================================
# FINAL OUTPUT
# =====================================================

print(
    "\n========================================"
)

print(
    "SMART CHUNKING COMPLETED"
)

print(
    "========================================"
)

print(
    f"Input Pages      : {len(documents)}"
)

print(
    f"Total Chunks     : {len(chunks)}"
)

print(
    f"Average Words    : {average_words:.2f}"
)

print(
    f"Output File      : {OUTPUT_JSON}"
)

print(
    "========================================"
)