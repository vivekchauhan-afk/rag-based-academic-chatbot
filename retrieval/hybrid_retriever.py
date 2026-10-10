from collections import defaultdict


class HybridRetriever:
    """
    Combines FAISS semantic retrieval and BM25 keyword retrieval.

    Pipeline:

        Query
          ↓
      Query Expansion
          ↓
       ┌───────┐
       ↓       ↓
     FAISS    BM25
       ↓       ↓
       └───┬───┘
           ↓
          RRF
           ↓
    Keyword Boosting
           ↓
      Ranked Results
    """

    def __init__(
        self,
        faiss_retriever,
        bm25_retriever,
        metadata
    ):
        self.faiss_retriever = faiss_retriever
        self.bm25_retriever = bm25_retriever
        self.metadata = metadata

    # =====================================================
    # QUERY EXPANSION
    # =====================================================

    def expand_query(self, query: str) -> str:

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

    # =====================================================
    # TOP-K SELECTION
    # =====================================================

    def get_top_k(self, question: str) -> int:

        q = question.lower()

        if self.is_festival_query(q):
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

    # =====================================================
    # FESTIVAL DETECTION
    # =====================================================

    def is_festival_query(self, question: str) -> bool:

        festivals = {
            "holi",
            "diwali",
            "ram navami",
            "eid",
            "bakrid",
            "christmas",
            "independence day",
            "republic day"
        }

        question = question.lower()

        return any(
            festival in question
            for festival in festivals
        )

    # =====================================================
    # HYBRID RETRIEVAL
    # =====================================================

    def retrieve(self, question: str):

        expanded_question = self.expand_query(
            question
        )

        top_k = self.get_top_k(
            question
        )

        # -------------------------------------------------
        # FAISS RETRIEVAL
        # -------------------------------------------------

        faiss_results = self.faiss_retriever.search(
            expanded_question,
            top_k
        )

        # -------------------------------------------------
        # BM25 RETRIEVAL
        # -------------------------------------------------

        bm25_results = self.bm25_retriever.search(
            expanded_question,
            top_k
        )

        # -------------------------------------------------
        # RECIPROCAL RANK FUSION
        # -------------------------------------------------

        fusion_scores = defaultdict(float)

        k = 60

        for rank, idx in enumerate(faiss_results):

            fusion_scores[idx] += (
                1 / (k + rank)
            )

        for rank, idx in enumerate(bm25_results):

            fusion_scores[idx] += (
                1 / (k + rank)
            )

        # -------------------------------------------------
        # DOCUMENT KEYWORD BOOSTING
        # -------------------------------------------------

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

            if idx >= len(self.metadata):
                continue

            text = self.metadata[idx][
                "text"
            ].lower()

            for keyword, boost in (
                important_keywords.items()
            ):

                if (
                    keyword in question_lower
                    and keyword in text
                ):
                    fusion_scores[idx] += boost

        # -------------------------------------------------
        # FINAL SORTING
        # -------------------------------------------------

        ranked = sorted(
            fusion_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            idx
            for idx, score in ranked
        ]