# RAG-Based Academic Chatbot

An AI-powered academic assistant designed to answer institute-related
questions using **Retrieval-Augmented Generation (RAG)**. The project
organizes document ingestion, retrieval, context construction, and
response generation into separate Python modules.

> **Project status:** This repository contains the modular RAG
> implementation. Exact data paths, model settings, and runtime
> dependencies may depend on the local configuration. Review `config.py`
> and the ingestion scripts before running the pipeline for the first
> time.

## Contents

-   [Overview](#overview)
-   [How it works](#how-it-works)
-   [Project structure](#project-structure)
-   [Requirements](#requirements)
-   [Setup](#setup)
-   [Prepare your knowledge base](#prepare-your-knowledge-base)
-   [Run the application](#run-the-application)
-   [Troubleshooting](#troubleshooting)
-   [Notes for contributors](#notes-for-contributors)

## Overview

The chatbot is intended to answer questions from a curated collection of
academic or institute documents. Instead of relying only on a language
model's built-in knowledge, the RAG workflow retrieves relevant document
passages and uses them as context for generating a response.

Core components include:

-   **Document ingestion:** extract text from supported source
    documents.
-   **Chunking:** split extracted text into smaller passages suitable
    for retrieval.
-   **Embeddings:** represent passages and user questions as vectors
    using Sentence Transformers.
-   **Vector search:** retrieve semantically relevant passages using
    FAISS.
-   **Hybrid retrieval:** combine retrieval approaches where configured,
    including BM25 keyword retrieval.
-   **Reranking:** reorder candidate passages where configured.
-   **Context construction:** assemble retrieved passages for the
    generation step.
-   **Answer generation:** generate a response using the model
    configured by the project.

The embedding model referenced in the project notes is
`all-mpnet-base-v2`. Confirm the active model and settings in the code
before changing them.

## How it works

``` text
Academic documents
       |
       v
Text extraction and cleaning
       |
       v
Chunking
       |
       v
Embedding generation ----------------------+
       |                                   |
       v                                   |
FAISS vector index                         |
                                           |
User question                              |
       |                                   |
       v                                   |
Question embedding ------------------------+
       |
       v
Retrieval (vector / keyword / hybrid, as configured)
       |
       v
Optional reranking
       |
       v
Context construction
       |
       v
Configured language model
       |
       v
Answer to the user
```

The exact sequence depends on the settings and implementation in the
current code.

## Project structure

``` text
.
├── app.py                     # Application entry point
├── build_vector_db.py         # Builds the vector index
├── config.py                  # Project configuration
├── component.py               # Application components
├── query_system.py            # Query-system module (if used by the current app)
├── run_experiments.py         # Experiment runner
├── test_output.py             # Output/testing utility
├── generation/
│   ├── llm.py                 # Language-model integration
│   └── prompts.py             # Prompt templates
├── ingestion/
│   ├── extract_data.py         # Document text extraction
│   ├── chunk_data.py           # Text chunking
│   └── website_crawler.py      # Website crawling
├── pipeline/
│   ├── rag_pipeline.py         # RAG pipeline orchestration
│   └── context_builder.py      # Retrieved-context preparation
├── retrieval/
│   ├── embeddings.py           # Embedding model utilities
│   ├── faiss_retriever.py      # FAISS retrieval
│   ├── bm25_retriever.py       # BM25 keyword retrieval
│   ├── hybrid_retriever.py     # Hybrid retrieval
│   └── reranker.py             # Candidate reranking
├── requirements.txt
└── README.md
```

Some files and generated folders may differ between local environments.
Check the repository if a listed optional file is not present.

## Requirements

-   Python version compatible with the packages pinned in
    `requirements.txt`
-   Git
-   Internet access for downloading Python packages and any models not
    already cached
-   Local source documents for your academic knowledge base
-   Any model runtime required by the configuration in
    `generation/llm.py` (for example, a locally hosted model service if
    configured)

> **Dependency note:** `requirements.txt` pins many packages, including
> a CUDA-specific PyTorch build. Installation can depend on your
> operating system, Python version, CUDA support, and package index
> configuration. If installation fails, use the official PyTorch
> installation instructions to select a compatible build, then install
> the remaining project dependencies. Do not blindly upgrade every
> package, because version changes may affect compatibility.

## Setup

The commands below use Windows PowerShell. Run them from the repository
root---the folder containing `app.py` and `requirements.txt`.

### 1. Clone the repository

``` powershell
git clone https://github.com/vivekchauhan-afk/rag-based-academic-chatbot.git
cd rag-based-academic-chatbot
```

If you already have the repository locally, open that folder instead of
cloning it again.

### 2. Create and activate a virtual environment

``` powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, consult the Python documentation for
your system's execution-policy settings. Avoid changing machine-wide
security settings unnecessarily.

### 3. Install dependencies

``` powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If installation fails, read the first package-specific error carefully.
In particular, verify that the pinned PyTorch build is available for
your platform and that the Python version is supported by the pinned
packages.

## Prepare your knowledge base

1.  Review `config.py` and the ingestion scripts to find the expected
    input paths and configuration values.

2.  Put only documents you are permitted to process into the configured
    input location.

3.  Make sure the source files are in a format supported by the
    extraction code.

4.  Build or refresh the vector database using the project's indexing
    script:

    ``` powershell
    python build_vector_db.py
    ```

5.  Confirm that the script completes successfully and creates the index
    files in the configured output location.

**Important:** The repository's `.gitignore` excludes local data and
generated vector-database artifacts. These files may need to be created
on your machine; they are not necessarily included in a fresh clone. Do
not commit private institute documents, credentials, model weights, or
large generated files unless you have permission and a clear reason to
do so.

If the indexing script expects specific filenames or folders, follow the
paths defined in the code rather than assuming a default path.

## Run the application

From the repository root, with the virtual environment activated, run:

``` powershell
streamlit run app.py
```

Streamlit normally prints a local address in the terminal. Open that
address in your browser.

If the application uses a local language-model service, start that
service and download/configure the required model first. Check
`generation/llm.py` and `config.py` for the model provider, model name,
and required environment variables. Do not assume a model name that is
not configured in your copy of the project.

## Troubleshooting

### `ModuleNotFoundError`

Make sure the virtual environment is activated and install the
dependencies:

``` powershell
python -m pip install -r requirements.txt
```

Run commands from the repository root so Python can find the project
packages.

### Vector index or data files are missing

Run the indexing script after confirming the expected source-data path:

``` powershell
python build_vector_db.py
```

Check `config.py` and the ingestion scripts for the configured paths. A
fresh clone may not include local source data or generated indexes.

### PyTorch or CUDA installation fails

The pinned PyTorch build may not match your computer. Select a
compatible installation for your operating system, Python version, and
hardware using the official PyTorch installation selector. Keep in mind
that changing PyTorch versions may require matching versions of related
packages.

### The language model does not respond

Check the settings in `generation/llm.py` and `config.py`. Confirm that
the configured model is installed and that any required local service is
running. Review the terminal output for the actual error.

### Answers are incorrect or not grounded in documents

-   Check that the correct documents were indexed.
-   Rebuild the index after changing source documents.
-   Review chunking, retrieval, reranking, and context settings.
-   Make sure the generation prompt instructs the model to use retrieved
    context and to acknowledge missing information.

## Notes for contributors

-   Keep secrets and credentials in environment variables or a local
    `.env` file; never commit them.
-   Keep `.venv/`, model checkpoints, generated indexes, logs, and
    private source documents out of version control unless there is a
    deliberate reason to include them.
-   Update this README whenever setup commands, required models, data
    paths, or application entry points change.
-   Test the ingestion, indexing, retrieval, and application flow after
    changing dependencies or pipeline modules.

## License

No license is specified here. Add a `LICENSE` file before inviting
others to reuse, modify, or redistribute the project. Without an
explicit license, standard copyright restrictions generally apply.
