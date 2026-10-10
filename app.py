import streamlit as st
from pipeline.rag_pipeline import RAGPipeline


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="NIT Kurukshetra AI Assistant",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# =====================================================
# SESSION STATE
# =====================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# =====================================================
# LOAD RAG PIPELINE
# =====================================================

@st.cache_resource
def load_rag_pipeline():
    """
    Load the complete RAG pipeline once.

    Streamlit reruns the script whenever the user
    interacts with the UI, so cache_resource prevents
    the embedding model, reranker, and LLM from being
    loaded again on every interaction.
    """
    return RAGPipeline()


rag_pipeline = load_rag_pipeline()


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.title("🎓 NIT Kurukshetra")

    st.caption("AI Assistant")

    if st.button("🗑️ New Chat", use_container_width=True):

        st.session_state.messages.clear()

        st.rerun()


# =====================================================
# LOAD CSS
# =====================================================

with open("style.css", "r", encoding="utf-8") as f:

    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )


# =====================================================
# HEADER
# =====================================================

st.markdown(
    """
<div class="main-header">

<h1>🎓 College AI Assistant</h1>

<p>
Hybrid Retrieval-Augmented Generation
</p>

</div>
""",
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)


# =====================================================
# CHAT HISTORY
# =====================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        # ---------------------------------------------
        # ASSISTANT CONTENT
        # ---------------------------------------------

        if message["role"] == "assistant":

            # -----------------------------------------
            # SOURCES
            # -----------------------------------------

            sources = message.get(
                "sources",
                []
            )

            if sources:

                st.markdown("#### 📄 Sources")

                for source in sources:

                    st.markdown(
                        f"""
- **{source.get("source", "Unknown")}**
  - Page: {source.get("page", "")}
"""
                    )

                    # ---------------------------------
                    # CLICKABLE SOURCE URL
                    # ---------------------------------

                    source_url = source.get(
                        "url",
                        ""
                    )

                    if source_url:

                        st.link_button(
                            "🔗 Open Source Document",
                            source_url
                        )

            # -----------------------------------------
            # DOWNLOADS
            # -----------------------------------------

            downloads = message.get(
                "downloads",
                []
            )

            if downloads:

                st.markdown("#### 📥 Downloads")

                for link in downloads:

                    st.link_button(
                        "Open Document",
                        link
                    )


# =====================================================
# CHAT INPUT
# =====================================================

question = st.chat_input(
    "Ask anything about NIT Kurukshetra..."
)


# =====================================================
# PROCESS QUESTION
# =====================================================

if question:

    # =================================================
    # SHOW USER MESSAGE
    # =================================================

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)


    # =================================================
    # CALL RAG PIPELINE
    # =================================================

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching official documents..."
        ):

            try:

                response = rag_pipeline.ask(
                    question
                )

            except Exception as e:

                st.error(
                    f"Error while processing the question: {e}"
                )

                st.stop()


        # =================================================
        # ANSWER
        # =================================================

        answer = response.get(
            "answer",
            "No response generated."
        )

        st.markdown(answer)


        # =================================================
        # SOURCES
        # =================================================

        sources = response.get(
            "sources",
            []
        )

        if sources:

            st.markdown("### 📄 Sources")

            for source in sources:

                st.markdown(
                    f"""
**Document**

{source.get("source", "Unknown")}

**Page**

{source.get("page", "")}
"""
                )

                # -----------------------------------------
                # CLICKABLE SOURCE URL
                # -----------------------------------------

                source_url = source.get(
                    "url",
                    ""
                )

                if source_url:

                    st.link_button(
                        "🔗 Open Source Document",
                        source_url
                    )

                st.markdown("---")


        # =================================================
        # DOWNLOAD LINKS
        # =================================================

        downloads = response.get(
            "downloads",
            []
        )

        if downloads:

            st.markdown("### 📥 Download")

            for link in downloads:

                st.link_button(
                    "Open Document",
                    link
                )


    # =================================================
    # SAVE ASSISTANT RESPONSE
    # =================================================

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "downloads": downloads
        }
    )