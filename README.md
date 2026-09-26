

````markdown
# 📚 Bookly - RAG Based Educational Chatbot

Bookly is an open-source **Retrieval-Augmented Generation (RAG) based educational chatbot** designed to answer questions using information collected from educational websites and documents.

The system combines **web crawling, data extraction, text chunking, embeddings, FAISS vector search, hybrid retrieval, Cross-Encoder reranking, query processing, and an LLM** to generate context-aware answers.

The project also contains a separate **fine-tuning pipeline** for experimenting with language-model adaptation and evaluation.

---

## 🚀 Project Overview

Educational information is often distributed across websites, PDFs, HTML pages, and Excel files. Finding the correct information manually can be time-consuming.

Bookly provides a complete pipeline that can:

1. Crawl an educational website.
2. Extract useful information from supported data sources.
3. Clean and divide the extracted text into meaningful chunks.
4. Convert chunks into vector embeddings.
5. Store embeddings in a FAISS vector database.
6. Process the user's question.
7. Retrieve relevant information using semantic and keyword-based retrieval.
8. Rerank retrieved results using a Cross-Encoder.
9. Filter the retrieved context.
10. Pass the relevant context to an LLM.
11. Generate an answer through the chatbot interface.

The project is designed so that users can provide their **own educational website/data** rather than depending on the original dataset used during development.

---

# 🧠 System Architecture

```text
                    ┌─────────────────────┐
                    │   Educational URL   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Website Crawler   │
                    │ website_crawler.py  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Data Extraction  │
                    │   extract_data.py   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      Chunking       │
                    │    chunk_data.py    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Embeddings       │
                    │ Sentence Transformers│
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    FAISS Index      │
                    │ build_vector_db.py  │
                    └──────────┬──────────┘
                               │
                               │
                    ┌──────────▼──────────┐
                    │    User Question    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Query Processing  │
                    │   query_system.py   │
                    └──────────┬──────────┘
                               │
                               ▼
                ┌──────────────────────────────┐
                │     Hybrid Retrieval        │
                │ Semantic + BM25 Retrieval   │
                └──────────────┬───────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Cross Encoder    │
                    │      Reranking      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Context Filtering  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │        LLM          │
                    │   Response Generate │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      Chatbot        │
                    │       app.py        │
                    └─────────────────────┘
````

---

# ✨ Main Features

* 🌐 Educational website crawling
* 📄 PDF data extraction
* 🌐 HTML data extraction
* 📊 Excel data processing
* ✂️ Automatic text chunking
* 🔢 Sentence Transformer embeddings
* ⚡ FAISS vector database
* 🔍 Semantic search
* 🔤 BM25 keyword retrieval
* 🔀 Hybrid retrieval
* 🎯 Cross-Encoder reranking
* 🧹 Context filtering
* 🤖 LLM-based answer generation
* 🧪 Experiment and evaluation pipeline
* 🎓 Separate fine-tuning pipeline
* 💻 Chatbot application interface

---

# 📂 Project Structure

```text
Bookly/
│
├── README.md
├── .gitignore
├── requirements.txt
│
├── website_crawler.py
├── extract_data.py
├── chunk_data.py
├── build_vector_db.py
├── query_system.py
│
├── app.py
├── component.py
├── config.py
├── run_experiments.py
├── test_output.py
├── style.css
│
├── assets/
│
└── finetuning/
    ├── __init__.py
    ├── callbacks.py
    ├── compare_models.py
    ├── config.py
    ├── dataset_loader.py
    ├── evaluate.py
    ├── metrics.py
    ├── predict.py
    ├── prepare_dataset.py
    ├── train.py
    └── trainer.py
```

---

# 🔧 Project Components

## 1. Website Crawler

### `website_crawler.py`

This module is responsible for collecting web pages from an educational website.

The crawler:

* Starts from a provided URL.
* Maintains a queue of URLs to visit.
* Keeps track of visited URLs.
* Downloads accessible web pages.
* Extracts links from pages.
* Adds relevant links to the crawling queue.
* Limits the number of pages crawled.

The crawler allows the project to work with a user's own educational website instead of requiring the original dataset.

### Important consideration

When crawling a website, users should respect:

* `robots.txt`
* Website terms of service
* Copyright restrictions
* Access restrictions
* Appropriate request rates

---

# 📄 2. Data Extraction

### `extract_data.py`

This module converts different data sources into usable text.

The project supports processing sources such as:

* PDF
* HTML
* Excel

The extraction stage removes the dependency on the original document format and produces text that can be processed by the RAG pipeline.

Libraries used include:

* PyMuPDF
* pdfplumber
* BeautifulSoup
* pandas

---

# ✂️ 3. Text Chunking

### `chunk_data.py`

This module divides large extracted documents into smaller text chunks.

Large documents cannot efficiently be passed directly to the language model. Therefore, the text is divided into smaller sections.

The chunking process:

1. Reads extracted text.
2. Identifies meaningful text sections.
3. Creates chunks within the configured size.
4. Maintains a small overlap between chunks.
5. Stores the resulting chunks for embedding generation.

Chunking improves retrieval because the vector database can search smaller and more focused pieces of information.

---

# 🔢 4. Embedding Generation

Bookly converts every text chunk into a numerical vector using a Sentence Transformer model.

The project uses:

```text
all-mpnet-base-v2
```

The process is:

```text
Text Chunk
    ↓
Sentence Transformer
    ↓
Embedding Vector
```

These vectors allow the system to perform semantic similarity search.

This means that the system can retrieve information based on **meaning**, rather than only matching exact words.

---

# ⚡ 5. FAISS Vector Database

### `build_vector_db.py`

FAISS is used to store and search embedding vectors efficiently.

The general process is:

```text
Text Chunks
     ↓
Embeddings
     ↓
FAISS Index
```

The project uses an Inner Product based FAISS index:

```python
index = faiss.IndexFlatIP(dimension)
```

The embedding dimension is obtained from the embedding matrix:

```python
dimension = embeddings.shape[1]
```

The vectors are then added to the FAISS index:

```python
index.add(embeddings)
```

FAISS allows the system to efficiently find vectors that are similar to the query embedding.

---

# 🔍 6. Query Processing

### `query_system.py`

This module handles the retrieval stage after the user asks a question.

The query pipeline includes:

```text
User Query
    ↓
Query Processing
    ↓
Query Expansion
    ↓
Semantic Retrieval
    +
BM25 Retrieval
    ↓
Hybrid Results
    ↓
Cross-Encoder Reranking
    ↓
Context Filtering
    ↓
Relevant Context
```

The goal is to provide the language model with the most relevant information.

---

# 🔀 7. Hybrid Retrieval

Bookly uses more than one retrieval method.

## Semantic Retrieval

The user's query is converted into an embedding and compared with document embeddings.

This helps retrieve information based on semantic meaning.

## BM25 Retrieval

BM25 provides keyword-based retrieval.

It is useful when exact terms, names, keywords, or phrases are important.

## Hybrid Retrieval

The two retrieval approaches are combined:

```text
Semantic Retrieval
        +
    BM25 Retrieval
        ↓
 Hybrid Candidates
```

This provides both semantic and lexical retrieval capabilities.

---

# 🎯 8. Cross-Encoder Reranking

After retrieving candidate chunks, Bookly uses a Cross-Encoder to rerank them.

The project uses:

```text
ms-marco-MiniLM-L-6-v2
```

The Cross-Encoder receives:

```text
Query + Candidate Document
```

and calculates a relevance score.

The candidates are then sorted according to their relevance.

This provides an additional relevance-ranking stage after the initial retrieval.

---

# 🧹 9. Context Filtering

Not every retrieved document is necessarily useful.

The query system filters the retrieved results so that the final context contains the most relevant information.

This helps reduce unnecessary information being sent to the LLM.

The final context is then used for response generation.

---

# 🤖 10. Large Language Model

The retrieved context is provided to the language model together with the user's question.

The general process is:

```text
User Question
      +
Retrieved Context
      ↓
     LLM
      ↓
Generated Answer
```

The LLM generates the final natural-language response using the retrieved information.

The project can be configured to work with a locally hosted model through the project's configured LLM setup.

---

# 💬 11. Chatbot Application

### `app.py`

`app.py` provides the application interface for interacting with the Bookly chatbot.

The application connects the user interface with the retrieval and generation pipeline.

The overall interaction is:

```text
User
 ↓
Question
 ↓
Query System
 ↓
Relevant Documents
 ↓
LLM
 ↓
Answer
```

---

# 🧩 12. Supporting Files

## `component.py`

Contains reusable components used by the application/project.

## `config.py`

Contains project configuration and configurable parameters.

## `style.css`

Contains styling for the application interface.

## `test_output.py`

Used for testing the system output.

## `run_experiments.py`

Used to run experiments and compare different configurations/results.

---

# 🎓 Fine-Tuning Pipeline

The `finetuning/` directory contains the source code used for fine-tuning experiments.

```text
finetuning/
├── __init__.py
├── callbacks.py
├── compare_models.py
├── config.py
├── dataset_loader.py
├── evaluate.py
├── metrics.py
├── predict.py
├── prepare_dataset.py
├── train.py
└── trainer.py
```

---

## `prepare_dataset.py`

Responsible for preparing the dataset before training.

The preparation stage converts raw data into a format suitable for model training.

---

## `dataset_loader.py`

Responsible for loading the prepared training and validation data.

---

## `train.py`

Contains the training workflow.

The general process is:

```text
Dataset
   ↓
Tokenizer
   ↓
Model
   ↓
Training Configuration
   ↓
Trainer
   ↓
Fine-Tuned Model
```

---

## `trainer.py`

Contains the training logic and configuration used during model training.

It handles the training process and integrates the required training components.

---

## `config.py`

Contains fine-tuning configuration parameters such as training-related settings.

Keeping configuration separate makes experiments easier to reproduce and modify.

---

## `callbacks.py`

Contains callbacks used during the training process.

Callbacks can be used to perform actions at different stages of training.

---

## `metrics.py`

Contains evaluation metric calculations used to measure model performance.

---

## `evaluate.py`

Responsible for evaluating the trained model using the project's evaluation process.

---

## `predict.py`

Used for generating predictions from the trained or fine-tuned model.

---

## `compare_models.py`

Used to compare different model configurations or fine-tuning results.

---

# 🧪 Experiments

### `run_experiments.py`

The project includes an experiment pipeline to evaluate different configurations of the retrieval and response-generation system.

The experiment stage can be used to investigate the effect of different retrieval and model configurations.

The fine-tuning directory also provides tools for:

* Training
* Evaluation
* Metrics
* Prediction
* Model comparison

---

# 🛠️ Technologies Used

| Technology            | Purpose                       |
| --------------------- | ----------------------------- |
| Python                | Main programming language     |
| FAISS                 | Vector similarity search      |
| Sentence Transformers | Text embeddings               |
| Cross-Encoder         | Document reranking            |
| BM25                  | Keyword-based retrieval       |
| PyMuPDF               | PDF processing                |
| pdfplumber            | PDF text extraction           |
| BeautifulSoup         | HTML parsing                  |
| pandas                | Excel/data processing         |
| Ollama                | Local LLM serving             |
| Streamlit             | Chatbot interface             |
| PyTorch               | Deep learning/fine-tuning     |
| Hugging Face          | Model training and evaluation |

---

# 📦 Installation

## 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Move into the project directory:

```bash
cd Bookly
```

---

## 2. Create a virtual environment

### Windows

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🌐 Using Your Own Educational Website

Bookly is designed so users can provide their own educational website.

The general workflow is:

```text
Educational Website URL
        ↓
Website Crawler
        ↓
HTML/Data Extraction
        ↓
Text Cleaning
        ↓
Chunking
        ↓
Embeddings
        ↓
FAISS
        ↓
Question Answering
```

For example, a user can provide the URL of an educational institution or educational resource website and crawl the permitted pages.

> Make sure you have permission to crawl and process the website's content and follow its robots.txt and terms of service.

---

# ▶️ Running the Pipeline

## Step 1 - Crawl

Run:

```bash
python website_crawler.py
```

The crawler collects accessible pages from the configured starting URL.

---

## Step 2 - Extract

Run:

```bash
python extract_data.py
```

This extracts usable text from the collected data.

---

## Step 3 - Chunk

Run:

```bash
python chunk_data.py
```

This divides the extracted content into smaller chunks suitable for retrieval.

---

## Step 4 - Build Vector Database

Run:

```bash
python build_vector_db.py
```

This:

1. Loads the text chunks.
2. Generates embeddings.
3. Creates the FAISS index.
4. Stores the vectors.

---

## Step 5 - Query

The query system performs:

```text
Question
 ↓
Query Processing
 ↓
Semantic Retrieval
 +
BM25
 ↓
Reranking
 ↓
Context Filtering
 ↓
LLM
 ↓
Answer
```

---

## Step 6 - Start the Application

Run:

```bash
streamlit run app.py
```

The application will start the Bookly chatbot interface.

---

# ⚙️ Configuration

Project configuration is maintained through configuration files.

Important settings can include:

* Model configuration
* Embedding model
* Retrieval parameters
* Chunking parameters
* Reranking parameters
* Application settings
* Fine-tuning parameters

Users should review:

```text
config.py
```

and the configuration files inside:

```text
finetuning/
```

before running the system.

---

# 📊 Fine-Tuning Workflow

The fine-tuning pipeline follows this general process:

```text
Raw Dataset
     ↓
Dataset Preparation
     ↓
Dataset Loading
     ↓
Model + Tokenizer
     ↓
Training
     ↓
Evaluation
     ↓
Metrics
     ↓
Prediction
     ↓
Model Comparison
```

The source code for each stage is provided inside the `finetuning/` directory.

Large generated datasets, caches, training outputs, and model weights are intentionally not included in this repository.

---

# 📁 Data and Model Files

The original development dataset and generated files are not included in the GitHub repository.

These local/generated resources include:

```text
data/
processed/
vector_db/
best_models/
finetuning/dataset/
finetuning/folds/
finetuning/outputs/
```

These files can be generated locally by following the project pipeline.

This keeps the repository smaller and allows users to build the system using their own permitted educational data.

---

# 🔒 Privacy and Security

Do not commit:

* API keys
* Passwords
* Access tokens
* `.env` files
* Private datasets
* Private documents
* Restricted model weights

The `.gitignore` file is configured to prevent many generated and sensitive files from being committed accidentally.

---

# ⚠️ Limitations

The performance of the chatbot depends on:

* Quality of the source website/data
* Quality of extracted text
* Chunking strategy
* Embedding quality
* Retrieval parameters
* Reranking quality
* LLM capabilities
* Fine-tuning data quality

The crawler also depends on the structure and accessibility of the target website.

---

# 🔮 Future Improvements

Possible future improvements include:

* Graph-based retrieval
* Better query rewriting
* Improved document metadata handling
* Multilingual retrieval
* Better citation/source tracking
* Conversation memory
* Improved evaluation datasets
* More efficient vector search
* Improved reranking models
* Support for additional document formats
* Cloud deployment
* Authentication and user management

---

# 🎯 Project Goals

The main goals of Bookly are:

* Build a practical educational RAG system.
* Retrieve relevant information from educational resources.
* Reduce the need for manually searching large collections of documents.
* Combine semantic and keyword retrieval.
* Improve retrieval using reranking.
* Experiment with model fine-tuning.
* Provide a reusable pipeline that can work with different educational websites and datasets.

---

# 🤝 Contributing

Contributions are welcome.

To contribute:

1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Test the changes.
5. Create a pull request.

---

# 📜 License

Add the license that you choose for this project.

If the project contains content collected from third-party websites or documents, make sure that your use and redistribution comply with the applicable licenses, copyright rules, and website terms.

---

# 👨‍💻 Project

## Bookly - RAG Based Educational Chatbot

An educational project demonstrating:

```text
Web Crawling
     +
Data Processing
     +
NLP
     +
Embeddings
     +
Vector Search
     +
Hybrid Retrieval
     +
Reranking
     +
LLM
     +
Fine-Tuning
```

---

````

### After pasting

Save the file, then run:

```powershell
git add README.md
````

Then:

```powershell
git status
```

Send me the output. **Don't commit yet**—we'll verify everything one final time before the first commit.
