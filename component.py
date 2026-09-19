import streamlit as st

from rag.config import *

# ==========================================================
# HEADER
# ==========================================================

def show_header():

    st.markdown(
        f"""
        <div class="header">

            <h1>{APP_TITLE}</h1>

            <p>{APP_SUBTITLE}</p>

        </div>
        """,
        unsafe_allow_html=True
    )


# ==========================================================
# FOOTER
# ==========================================================

def show_footer():

    st.markdown("---")

    st.markdown(
        f"""
        <div class="footer">

        {FOOTER}

        </div>
        """,
        unsafe_allow_html=True
    )


# ==========================================================
# SOURCE CARD
# ==========================================================

def source_card(source):

    st.markdown(
        f"""
        <div class="source-card">

        <h4>📄 {source['source']}</h4>

        <p><b>Page :</b> {source['page']}</p>

        <p><b>Score :</b> {source['score']:.3f}</p>

        </div>
        """,
        unsafe_allow_html=True
    )


# ==========================================================
# DOWNLOAD CARD
# ==========================================================

def download_card(path):

    st.markdown(
        f"""
        <div class="download-card">

        📥 Download Document

        </div>
        """,
        unsafe_allow_html=True
    )

    st.link_button(
        "Open File",
        path
    )


# ==========================================================
# NAVIGATION CARD
# ==========================================================

def navigation_card(page):

    st.markdown(
        f"""
        <div class="navigation-card">

        <h3>📍 Website Navigation</h3>

        <b>Section</b><br>

        {page['parent_title']}<br><br>

        <b>Page</b><br>

        {page['title']}

        </div>
        """,
        unsafe_allow_html=True
    )

    st.link_button(
        "Open Website",
        page["url"]
    )


# ==========================================================
# USER MESSAGE
# ==========================================================

def user_message(message):

    with st.chat_message("user"):

        st.markdown(message)


# ==========================================================
# AI MESSAGE
# ==========================================================

def assistant_message(message):

    with st.chat_message("assistant"):

        st.markdown(message)


# ==========================================================
# STATISTICS
# ==========================================================

def show_statistics():

    st.sidebar.markdown("## 📊 System Information")

    st.sidebar.info(

f"""
**Embedding**

{EMBEDDING_MODEL}

---

**Retriever**

FAISS + BM25

---

**Reranker**

{RERANKER_MODEL}

---

**LLM**

{LLM_MODEL}

---

**Search**

{SEARCH_METHOD}

"""
)


# ==========================================================
# QUICK QUESTIONS
# ==========================================================

def quick_questions():

    st.markdown("### 💡 Suggested Questions")

    cols = st.columns(4)

    buttons = []

    for i, question in enumerate(QUICK_QUESTIONS):

        with cols[i % 4]:

            clicked = st.button(
                question,
                use_container_width=True,
                key=f"quick_{i}"
            )

            buttons.append(clicked)

    for i, clicked in enumerate(buttons):

        if clicked:

            st.session_state["question"] = QUICK_QUESTIONS[i]


# ==========================================================
# SIDEBAR
# ==========================================================

def sidebar():

    # st.sidebar.image(
    #     LOGO,
    #     width=150
    # )

    st.sidebar.title("Navigation")

    page = st.sidebar.radio(

        "",

        MENU

    )

    show_statistics()

    return page