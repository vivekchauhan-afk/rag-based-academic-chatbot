import json
import os
import re
from urllib.parse import urlparse, unquote

import fitz          # PyMuPDF
import pdfplumber


# =====================================================
# CONFIGURATION
# =====================================================

PDF_FOLDER = "data/pdfs"
OUTPUT_FOLDER = "data/extracted"
PROCESSED_FOLDER = "processed"

OUTPUT_JSON = os.path.join(
    PROCESSED_FOLDER,
    "extracted_raw.json"
)

FILE_URLS_FILE = os.path.join(
    PROCESSED_FOLDER,
    "file_urls.json"
)

NAVIGATION_FILE = os.path.join(
    PROCESSED_FOLDER,
    "navigation.json"
)


# =====================================================
# CREATE DIRECTORIES
# =====================================================

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)

os.makedirs(
    PROCESSED_FOLDER,
    exist_ok=True
)


# =====================================================
# SKIP IF EXTRACTION ALREADY EXISTS
# =====================================================

if os.path.exists(OUTPUT_JSON):

    print("=" * 50)
    print("EXTRACTED DATA ALREADY EXISTS")
    print("=" * 50)

    print(
        f"Existing file: {OUTPUT_JSON}"
    )

    print(
        "Skipping PDF extraction."
    )

    print("=" * 50)

    raise SystemExit


# =====================================================
# URL MAPPING
# =====================================================

url_mapping = {}


# =====================================================
# LOAD file_urls.json
# =====================================================

if os.path.exists(FILE_URLS_FILE):

    try:

        with open(
            FILE_URLS_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        if isinstance(data, list):

            for item in data:

                filename = item.get(
                    "filename",
                    ""
                )

                url = item.get(
                    "url",
                    ""
                )

                if filename and url:

                    url_mapping[filename] = url

        elif isinstance(data, dict):

            for filename, url in data.items():

                if filename and url:

                    url_mapping[filename] = url

        print(
            f"URLs loaded from file_urls.json: "
            f"{len(url_mapping)}"
        )

    except Exception as e:

        print(
            f"Warning: Could not load file_urls.json: {e}"
        )

else:

    print(
        "Warning: file_urls.json not found."
    )


# =====================================================
# LOAD navigation.json
# =====================================================

if os.path.exists(NAVIGATION_FILE):

    try:

        with open(
            NAVIGATION_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            navigation_data = json.load(f)

        navigation_pdf_count = 0

        for item in navigation_data:

            url = item.get(
                "url",
                ""
            )

            if not url:
                continue

            parsed_url = urlparse(url)

            # Only use direct PDF URLs
            if not parsed_url.path.lower().endswith(
                ".pdf"
            ):
                continue

            filename = os.path.basename(
                parsed_url.path
            )

            filename = unquote(
                filename
            )

            if filename:

                # navigation.json is used as an
                # additional URL source.
                #
                # If the same filename already exists
                # in file_urls.json, keep the existing URL.

                if filename not in url_mapping:

                    url_mapping[filename] = url

                navigation_pdf_count += 1

        print(
            f"PDF URLs found in navigation.json: "
            f"{navigation_pdf_count}"
        )

    except Exception as e:

        print(
            f"Warning: Could not load navigation.json: {e}"
        )

else:

    print(
        "Warning: navigation.json not found."
    )


print(
    f"Total URL mappings available: "
    f"{len(url_mapping)}"
)


# =====================================================
# CLEAN TEXT
# =====================================================

def clean_text(text):

    if not text:

        return ""

    # Remove null characters

    text = text.replace(
        "\x00",
        " "
    )

    # Normalize spaces and tabs

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    # Reduce excessive blank lines

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# =====================================================
# EXTRACT PAGE TEXT
# =====================================================

def extract_page_text(page):

    blocks = page.get_text(
        "blocks"
    )

    # Sort approximately
    # top-to-bottom and left-to-right

    blocks = sorted(
        blocks,
        key=lambda block: (
            block[1],
            block[0]
        )
    )

    page_text_parts = []

    for block in blocks:

        text = block[4].strip()

        # Skip tiny/empty blocks

        if len(text) < 2:

            continue

        page_text_parts.append(
            text
        )

    page_text = "\n".join(
        page_text_parts
    )

    return clean_text(
        page_text
    )


# =====================================================
# EXTRACT TABLES
# =====================================================

def extract_tables(pdf_path):

    page_tables = {}

    try:

        with pdfplumber.open(
            pdf_path
        ) as pdf:

            for page_number, page in enumerate(
                pdf.pages,
                start=1
            ):

                tables = page.extract_tables()

                if not tables:

                    page_tables[page_number] = ""

                    continue

                table_text_parts = []

                for table in tables:

                    if not table:

                        continue

                    for row in table:

                        if not row:

                            continue

                        cleaned_row = []

                        for cell in row:

                            if cell is None:

                                cell = ""

                            cell = str(
                                cell
                            ).strip()

                            if cell:

                                cleaned_row.append(
                                    cell
                                )

                        if not cleaned_row:

                            continue

                        table_text_parts.append(
                            " | ".join(
                                cleaned_row
                            )
                        )

                    # Separate multiple tables

                    table_text_parts.append(
                        ""
                    )

                table_text = "\n".join(
                    table_text_parts
                )

                page_tables[page_number] = clean_text(
                    table_text
                )

    except Exception as e:

        print(
            f"Table Extraction Error: {e}"
        )

    return page_tables


# =====================================================
# PROCESS SINGLE PDF
# =====================================================

def process_pdf(pdf_path):

    filename = os.path.basename(
        pdf_path
    )

    print(
        f"\nProcessing: {filename}"
    )

    documents = []

    # -------------------------------------------------
    # GET URL
    # -------------------------------------------------

    document_url = url_mapping.get(
        filename,
        ""
    )

    if document_url:

        print(
            f"URL found: {document_url}"
        )

    else:

        print(
            "Warning: URL not found for this PDF."
        )

    # -------------------------------------------------
    # EXTRACT TABLES FIRST
    # -------------------------------------------------

    table_data = extract_tables(
        pdf_path
    )

    try:

        with fitz.open(
            pdf_path
        ) as doc:

            total_pages = len(
                doc
            )

            print(
                f"Pages: {total_pages}"
            )

            for page_number in range(
                total_pages
            ):

                page = doc.load_page(
                    page_number
                )

                # -----------------------------------------
                # NORMAL TEXT
                # -----------------------------------------

                page_text = extract_page_text(
                    page
                )

                # -----------------------------------------
                # TABLE TEXT
                # -----------------------------------------

                table_text = table_data.get(
                    page_number + 1,
                    ""
                )

                # -----------------------------------------
                # MERGE PAGE CONTENT
                # -----------------------------------------

                merged_parts = []

                if page_text:

                    merged_parts.append(
                        page_text
                    )

                if table_text:

                    merged_parts.append(
                        "TABLE DATA"
                    )

                    merged_parts.append(
                        table_text
                    )

                merged_text = "\n\n".join(
                    merged_parts
                )

                merged_text = clean_text(
                    merged_text
                )

                # -----------------------------------------
                # SKIP BLANK PAGES
                # -----------------------------------------

                if len(merged_text) < 20:

                    continue

                # -----------------------------------------
                # CREATE PAGE DOCUMENT
                # -----------------------------------------

                documents.append({

                    "text": merged_text,

                    "source": filename,

                    "page": page_number + 1,

                    "url": document_url

                })

    except Exception as e:

        print(
            f"PDF Processing Error "
            f"({filename}): {e}"
        )

    return documents


# =====================================================
# CHECK PDF FOLDER
# =====================================================

if not os.path.exists(
    PDF_FOLDER
):

    raise FileNotFoundError(
        f"PDF folder not found: {PDF_FOLDER}"
    )


# =====================================================
# PROCESS ALL PDFs
# =====================================================

all_documents = []

total_files = 0


for filename in sorted(
    os.listdir(PDF_FOLDER)
):

    if not filename.lower().endswith(
        ".pdf"
    ):

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


# =====================================================
# EXTRACTION STATISTICS
# =====================================================

print()

print(
    f"PDF Files      : {total_files}"
)

print(
    f"Pages Extracted: {len(all_documents)}"
)


# =====================================================
# SAVE EXTRACTED PAGE TEXT
# =====================================================

print(
    "\nSaving extracted text files..."
)


for page in all_documents:

    source_filename = page[
        "source"
    ]

    filename_without_extension = os.path.splitext(
        source_filename
    )[0]

    page_number = page[
        "page"
    ]

    output_name = (
        f"{filename_without_extension}"
        f"_page_{page_number}.txt"
    )

    output_path = os.path.join(
        OUTPUT_FOLDER,
        output_name
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            page["text"]
        )


# =====================================================
# SAVE JSON FOR CHUNKING
# =====================================================

with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        all_documents,
        f,
        indent=4,
        ensure_ascii=False
    )


# =====================================================
# FINAL OUTPUT
# =====================================================

print(
    "\n========================================"
)

print(
    "EXTRACTION COMPLETED"
)

print(
    "========================================"
)

print(
    f"PDF Files       : {total_files}"
)

print(
    f"Pages Extracted : {len(all_documents)}"
)

print(
    f"URL Mappings    : {len(url_mapping)}"
)

print(
    f"Output JSON     : {OUTPUT_JSON}"
)

print(
    f"Text Files      : {OUTPUT_FOLDER}"
)

print(
    "========================================"
)