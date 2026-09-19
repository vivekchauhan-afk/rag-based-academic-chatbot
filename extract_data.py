import os
import re
import json
import fitz                     # PyMuPDF
import pdfplumber

# =====================================================
# CONFIGURATION
# =====================================================

PDF_FOLDER = "data/pdfs"
OUTPUT_FOLDER = "data/extracted"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs("../processed", exist_ok=True)

# =====================================================
# LOAD URL MAPPING
# =====================================================

url_mapping = {}

mapping_file = "../processed/file_urls.json"

if os.path.exists(mapping_file):

    with open(
        mapping_file,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

        for item in data:

            url_mapping[item["filename"]] = item["url"]

# =====================================================
# CLEAN TEXT
# =====================================================

def clean_text(text):

    if not text:
        return ""

    text = text.replace("\x00", " ")

    text = re.sub(r"[ \t]+", " ", text)

    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

# =====================================================
# EXTRACT PAGE TEXT
# =====================================================

def extract_page_text(page):

    blocks = page.get_text("blocks")

    blocks = sorted(
        blocks,
        key=lambda b: (b[1], b[0])
    )

    page_text = ""

    for block in blocks:

        text = block[4].strip()

        if len(text) < 2:
            continue

        page_text += text + "\n"

    return clean_text(page_text)

# =====================================================
# EXTRACT TABLES
# =====================================================

def extract_tables(pdf_path):

    page_tables = {}

    try:

        with pdfplumber.open(pdf_path) as pdf:

            for page_number, page in enumerate(pdf.pages):

                tables = page.extract_tables()

                text = ""

                if not tables:

                    page_tables[page_number + 1] = ""

                    continue

                for table in tables:

                    if not table:
                        continue

                    for row in table:

                        row = [

                            "" if c is None else str(c).strip()

                            for c in row

                        ]

                        row = [

                            c for c in row

                            if c != ""

                        ]

                        if len(row) == 0:
                            continue

                        text += " | ".join(row)

                        text += "\n"

                    text += "\n"

                page_tables[page_number + 1] = clean_text(text)

    except Exception as e:

        print(f"Table Extraction Error : {e}")

    return page_tables
# =====================================================
# PROCESS SINGLE PDF
# =====================================================

def process_pdf(pdf_path):

    filename = os.path.basename(pdf_path)

    print(f"\nProcessing: {filename}")

    documents = []

    # Extract all tables first
    table_data = extract_tables(pdf_path)

    try:

        doc = fitz.open(pdf_path)

        total_pages = len(doc)

        print(f"Pages: {total_pages}")

        for page_number in range(total_pages):

            page = doc.load_page(page_number)

            # -----------------------------
            # NORMAL TEXT
            # -----------------------------

            page_text = extract_page_text(page)

            # -----------------------------
            # TABLE TEXT
            # -----------------------------

            table_text = table_data.get(
                page_number + 1,
                ""
            )

            # -----------------------------
            # MERGE PAGE CONTENT
            # -----------------------------

            merged_text = ""

            if page_text:

                merged_text += page_text

            if table_text:

                merged_text += "\n\n"

                merged_text += "TABLE DATA\n"

                merged_text += table_text

            merged_text = clean_text(
                merged_text
            )

            # Skip blank pages
            if len(merged_text) < 20:
                continue

            documents.append({

                "text": merged_text,

                "source": filename,

                "page": page_number + 1,

                "url": url_mapping.get(
                    filename,
                    ""
                )

            })

        doc.close()

    except Exception as e:

        print(e)

    return documents


# =====================================================
# PROCESS ALL PDFS
# =====================================================

all_documents = []

total_files = 0

for filename in sorted(os.listdir(PDF_FOLDER)):

    if not filename.lower().endswith(".pdf"):
        continue

    total_files += 1

    pdf_path = os.path.join(
        PDF_FOLDER,
        filename
    )

    pages = process_pdf(
        pdf_path
    )

    all_documents.extend(
        pages
    )

print()

print(f"PDF Files : {total_files}")

print(f"Pages Extracted : {len(all_documents)}")

# =====================================================
# SAVE EXTRACTED PAGE TEXT (OPTIONAL)
# =====================================================

print("\nSaving extracted text files...")

for page in all_documents:

    filename = page["source"].replace(".pdf", "")

    page_number = page["page"]

    output_name = f"{filename}_page_{page_number}.txt"

    output_path = os.path.join(
        OUTPUT_FOLDER,
        output_name
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(page["text"])

# =====================================================
# SAVE JSON FOR CHUNKING
# =====================================================

output_json = "processed/extracted_raw.json"

with open(
    output_json,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        all_documents,
        f,
        indent=4,
        ensure_ascii=False
    )

print("\n========================================")
print("EXTRACTION COMPLETED")
print("========================================")
print(f"PDF Files        : {total_files}")
print(f"Pages Extracted  : {len(all_documents)}")
print(f"Output JSON      : {output_json}")
print(f"Text Files       : {OUTPUT_FOLDER}")
print("========================================")