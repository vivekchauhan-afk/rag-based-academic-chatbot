import json
import os
from urllib.parse import urlparse, unquote

from retrieval.embeddings import EmbeddingService
from retrieval.faiss_retriever import FAISSRetriever
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import DocumentReranker

from pipeline.context_builder import ContextBuilder

from generation.prompts import build_rag_prompt
from generation.llm import GemmaLLM


# =====================================================
# CONFIGURATION
# =====================================================

METADATA_PATH = "vector_db/metadata.pkl"
FAISS_INDEX_PATH = "vector_db/faiss_index.bin"

FILE_URLS_PATH = "processed/file_urls.json"
NAVIGATION_PATH = "processed/navigation.json"

CONFIDENCE_THRESHOLD = 0.15


# =====================================================
# RAG PIPELINE
# =====================================================

class RAGPipeline:

    def __init__(self):

        print("\n========================================")
        print("Initializing RAG Pipeline")
        print("========================================")

        # -------------------------------------------------
        # LOAD METADATA
        # -------------------------------------------------

        print("\nLoading metadata...")

        import pickle

        with open(
            METADATA_PATH,
            "rb"
        ) as f:

            self.metadata = pickle.load(f)

        print(
            f"Metadata records loaded: {len(self.metadata)}"
        )

        # -------------------------------------------------
        # LOAD DOCUMENT URLS
        # -------------------------------------------------

        self.document_urls = []

        self.load_document_urls()

        # -------------------------------------------------
        # EMBEDDING SERVICE
        # -------------------------------------------------

        self.embedding_service = EmbeddingService(
            model_name="all-mpnet-base-v2"
        )

        # -------------------------------------------------
        # FAISS RETRIEVER
        # -------------------------------------------------

        self.faiss_retriever = FAISSRetriever(
            index_path=FAISS_INDEX_PATH,
            metadata_path=METADATA_PATH,
            embedding_service=self.embedding_service
        )

        # -------------------------------------------------
        # BM25 RETRIEVER
        # -------------------------------------------------

        self.bm25_retriever = BM25Retriever(
            metadata=self.metadata
        )

        # -------------------------------------------------
        # HYBRID RETRIEVER
        # -------------------------------------------------

        self.hybrid_retriever = HybridRetriever(
            faiss_retriever=self.faiss_retriever,
            bm25_retriever=self.bm25_retriever,
            metadata=self.metadata
        )

        # -------------------------------------------------
        # RERANKER
        # -------------------------------------------------

        self.reranker = DocumentReranker(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

        # -------------------------------------------------
        # CONTEXT BUILDER
        # -------------------------------------------------

        self.context_builder = ContextBuilder(
            max_source_per_file=3,
            max_content_chunks=8,
            min_text_length=80
        )

        # -------------------------------------------------
        # LLM
        # -------------------------------------------------

        self.llm = GemmaLLM(
            base_model="google/gemma-2-2b-it",
            adapter_path="best_models/gemma/fold_5",
            max_new_tokens=100
        )

        print("\n========================================")
        print("RAG Pipeline Ready")
        print("========================================\n")


    # =====================================================
    # LOAD DOCUMENT URLS
    # =====================================================

    def load_document_urls(self):

        # -------------------------------------------------
        # LOAD DOWNLOADED FILE URLS
        # -------------------------------------------------

        if os.path.exists(FILE_URLS_PATH):

            try:

                with open(
                    FILE_URLS_PATH,
                    "r",
                    encoding="utf-8"
                ) as f:

                    file_urls = json.load(f)

                if isinstance(file_urls, list):

                    for item in file_urls:

                        filename = item.get(
                            "filename",
                            ""
                        )

                        url = item.get(
                            "url",
                            ""
                        )

                        if filename and url:

                            self.add_document_url(
                                filename=filename,
                                url=url,
                                title=filename
                            )

                elif isinstance(file_urls, dict):

                    for filename, url in file_urls.items():

                        if filename and url:

                            self.add_document_url(
                                filename=filename,
                                url=url,
                                title=filename
                            )

                print(
                    f"File URL mappings loaded: "
                    f"{len(file_urls)}"
                )

            except Exception as e:

                print(
                    f"Warning: Could not load file URLs: {e}"
                )

        else:

            print(
                "Warning: file_urls.json not found."
            )


        # -------------------------------------------------
        # LOAD NAVIGATION URLS
        # -------------------------------------------------

        if os.path.exists(NAVIGATION_PATH):

            try:

                with open(
                    NAVIGATION_PATH,
                    "r",
                    encoding="utf-8"
                ) as f:

                    navigation = json.load(f)

                pdf_count = 0

                for item in navigation:

                    url = item.get(
                        "url",
                        ""
                    )

                    title = item.get(
                        "title",
                        ""
                    )

                    if not url:
                        continue

                    parsed_url = urlparse(url)

                    # Only direct PDF URLs
                    if not parsed_url.path.lower().endswith(
                        ".pdf"
                    ):
                        continue

                    filename = os.path.basename(
                        parsed_url.path
                    )

                    filename = unquote(filename)

                    self.add_document_url(
                        filename=filename,
                        url=url,
                        title=title
                    )

                    pdf_count += 1

                print(
                    f"PDF URLs loaded from navigation.json: "
                    f"{pdf_count}"
                )

            except Exception as e:

                print(
                    f"Warning: Could not load navigation URLs: {e}"
                )

        else:

            print(
                "Warning: navigation.json not found."
            )


        print(
            f"Total document URLs available: "
            f"{len(self.document_urls)}"
        )


    # =====================================================
    # ADD DOCUMENT URL
    # =====================================================

    def add_document_url(
        self,
        filename: str,
        url: str,
        title: str = ""
    ):

        # Avoid duplicate URLs
        for document in self.document_urls:

            if document["url"] == url:

                # Update title if useful
                if (
                    title
                    and document.get("title") == document["filename"]
                ):

                    document["title"] = title

                return

        self.document_urls.append(
            {
                "filename": filename,
                "url": url,
                "title": title
            }
        )
    # =====================================================
    # INTENT DETECTION
    # =====================================================

    def detect_intent(self, question: str):

        q = question.lower().strip()

        # -------------------------------------------------
        # GREETING
        # -------------------------------------------------

        greetings = [
            "hi",
            "hello",
            "hey",
            "hii",
            "helo",
            "good morning",
            "good afternoon",
            "good evening"
        ]

        if q in greetings:
            return "greeting"


        # -------------------------------------------------
        # DOWNLOAD
        # -------------------------------------------------

        download_keywords = [
            "download",
            "pdf",
            "document",
            "file",
            "brochure",
            "prospectus"
        ]

        if any(
            keyword in q
            for keyword in download_keywords
        ):
            return "download"


        # -------------------------------------------------
        # INFORMATION
        # -------------------------------------------------

        return "information"


    # =====================================================
    # DOWNLOAD SEARCH
    # =====================================================

    def find_downloads(self, question: str):

        downloads = []

        q = question.lower()

        for document in self.document_urls:

            filename = document["filename"]
            url = document["url"]
            title = document.get("title", "")

            filename_lower = filename.lower()
            title_lower = title.lower()

            search_text = (
                filename_lower
                .replace("_", " ")
                .replace("-", " ")
            )

            filename_words = search_text.split()

            title_words = (
                title_lower
                .replace("_", " ")
                .replace("-", " ")
                .split()
            )

            matched = False

            # -------------------------------------------------
            # MATCH FILENAME
            # -------------------------------------------------

            for word in filename_words:

                if len(word) <= 2:
                    continue

                if word in q:
                    matched = True
                    break


            # -------------------------------------------------
            # MATCH NAVIGATION TITLE
            # -------------------------------------------------

            if not matched:

                for word in title_words:

                    if len(word) <= 2:
                        continue

                    if word in q:
                        matched = True
                        break


            # -------------------------------------------------
            # GENERIC DOWNLOAD REQUEST
            # -------------------------------------------------

            if not matched and "download" in q:
                matched = True


            # -------------------------------------------------
            # ADD URL
            # -------------------------------------------------

            if matched and url not in downloads:
                downloads.append(url)


        return downloads


    # =====================================================
    # RETRIEVE DOCUMENTS
    # =====================================================

    def retrieve_documents(self, question: str):

        candidate_indices = (
            self.hybrid_retriever.retrieve(
                question
            )
        )

        if not candidate_indices:
            return [], "", [], 0.0


        # -------------------------------------------------
        # RERANK
        # -------------------------------------------------

        reranked_documents = (
            self.reranker.rerank(
                question,
                candidate_indices,
                self.metadata
            )
        )

        if not reranked_documents:
            return [], "", [], 0.0


        # -------------------------------------------------
        # FESTIVAL SPECIAL CASE
        # -------------------------------------------------

        if self.hybrid_retriever.is_festival_query(
            question
        ):

            filtered_documents = (
                reranked_documents[:10]
            )

        else:

            filtered_documents = (
                reranked_documents[:5]
            )


        # -------------------------------------------------
        # BUILD CONTEXT
        # -------------------------------------------------

        (
            context,
            sources,
            max_score
        ) = self.context_builder.build(
            filtered_documents
        )


        return (
            filtered_documents,
            context,
            sources,
            max_score
        )


    # =====================================================
    # ASK QUESTION
    # =====================================================

    def ask(self, question: str):

        question = question.strip()

        # -------------------------------------------------
        # EMPTY QUESTION
        # -------------------------------------------------

        if not question:

            return {
                "answer": "Please enter a question.",
                "sources": [],
                "downloads": []
            }


        # -------------------------------------------------
        # INTENT
        # -------------------------------------------------

        intent = self.detect_intent(
            question
        )


        # -------------------------------------------------
        # GREETING
        # -------------------------------------------------

        if intent == "greeting":

            return {
                "answer": (
                    "Hello! I am the NIT Kurukshetra "
                    "AI Assistant. How can I help you?"
                ),
                "sources": [],
                "downloads": []
            }


        # -------------------------------------------------
        # DOWNLOAD INTENT
        # -------------------------------------------------

        if intent == "download":

            downloads = self.find_downloads(
                question
            )

            if downloads:

                return {
                    "answer": (
                        "I found the following "
                        "document(s) for you:"
                    ),
                    "sources": [],
                    "downloads": downloads
                }


        # -------------------------------------------------
        # RETRIEVAL
        # -------------------------------------------------

        (
            documents,
            context,
            sources,
            max_score
        ) = self.retrieve_documents(
            question
        )


        # -------------------------------------------------
        # NO RELEVANT DOCUMENTS
        # -------------------------------------------------

        if not context:

            return {
                "answer": (
                    "I could not find this information "
                    "in the available documents."
                ),
                "sources": [],
                "downloads": []
            }


        # -------------------------------------------------
        # CONFIDENCE CHECK
        # -------------------------------------------------

        if max_score < CONFIDENCE_THRESHOLD:

            return {
                "answer": (
                    "I could not find this information "
                    "in the available documents."
                ),
                "sources": sources,
                "downloads": []
            }


        # -------------------------------------------------
        # BUILD RAG PROMPT
        # -------------------------------------------------

        prompt = build_rag_prompt(
            context=context,
            question=question
        )


        # -------------------------------------------------
        # GENERATE ANSWER
        # -------------------------------------------------

        answer = self.llm.generate(
            prompt
        )


        # -------------------------------------------------
        # FALLBACK
        # -------------------------------------------------

        if not answer:

            answer = (
                "I could not find this information "
                "in the available documents."
            )


        # -------------------------------------------------
        # RETURN RESPONSE
        # -------------------------------------------------

        return {
            "answer": answer,
            "sources": sources,
            "downloads": []
        }