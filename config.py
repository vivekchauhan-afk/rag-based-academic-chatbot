"""
=========================================================
NIT Kurukshetra AI Assistant
Configuration File
=========================================================
"""

import os

# =========================================================
# APPLICATION SETTINGS
# =========================================================

APP_TITLE = "🎓 NIT Kurukshetra AI Assistant"

APP_SUBTITLE = "Hybrid RAG Powered College Information System"

PAGE_ICON = "🎓"

LAYOUT = "wide"

SIDEBAR_STATE = "expanded"

# =========================================================
# COLLEGE INFORMATION
# =========================================================

COLLEGE_NAME = "National Institute of Technology Kurukshetra"

COLLEGE_WEBSITE = "https://nitkkr.ac.in"

DEPARTMENT = "Department of Computer Engineering"

# =========================================================
# AI MODELS
# =========================================================

EMBEDDING_MODEL = "all-mpnet-base-v2"

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

LLM_MODEL = "mistral"

SEARCH_METHOD = "Hybrid RAG"

# =========================================================
# COLORS
# =========================================================

PRIMARY_COLOR = "#800000"      # Maroon

SECONDARY_COLOR = "#D4AF37"    # Gold

BACKGROUND_COLOR = "#F8F9FA"

TEXT_COLOR = "#222222"

CARD_COLOR = "#FFFFFF"

# =========================================================
# FILE PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VECTOR_DB = os.path.join(BASE_DIR, "vector_db")

PROCESSED = os.path.join(BASE_DIR, "processed")

DATA = os.path.join(BASE_DIR, "data")

ASSETS = os.path.join(BASE_DIR, "assets")

# =========================================================
# LOGO
# =========================================================
#
# LOGO = os.path.join(ASSETS, "nit_logo.png")
#
# BOT_IMAGE = os.path.join(ASSETS, "bot.png")
#
# BANNER = os.path.join(ASSETS, "banner.png")

# =========================================================
# CHAT SETTINGS
# =========================================================

MAX_HISTORY = 5

MAX_SOURCES = 3

CONFIDENCE_THRESHOLD = 0.25

# =========================================================
# FOOTER
# =========================================================

FOOTER = """
Developed for National Institute of Technology Kurukshetra

Hybrid Retrieval-Augmented Generation
FAISS • BM25 • CrossEncoder • Mistral
"""

# =========================================================
# QUICK QUESTIONS
# =========================================================

QUICK_QUESTIONS = [

    "Academic Calendar",

    "Placement Policy",

    "Hostel Fees",

    "Attendance Policy",

    "Download Academic Calendar",

    "Faculty Information",

    "Admission Process",

    "Library Timings"

]

# =========================================================
# SIDEBAR MENU
# =========================================================

MENU = [

    "🏠 Home",

    "💬 AI Assistant",

    "📅 Academic Calendar",

    "📚 Syllabus",

    "🎓 Admissions",

    "💼 Placements",

    "🏠 Hostel",

    "📥 Downloads",

    "📍 Navigation",

    "ℹ About"

]