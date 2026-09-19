import streamlit as st
from query_system import ask_question

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
# SESSION STATE
# =====================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

# =====================================================
# HEADER
# =====================================================

st.markdown(
    """
<div class="main-header">

<h1>🎓 College  AI Assistant</h1>

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

        # Sources

        if message["role"] == "assistant":

            if "sources" in message:

                if message["sources"]:

                    st.markdown("#### 📄 Sources")

                    for s in message["sources"]:

                        st.markdown(
                            f"""
- **{s['source']}**
  - Page: {s['page']}
"""
                        )

            # Downloads

            if "downloads" in message:

                if message["downloads"]:

                    st.markdown("#### 📥 Downloads")

                    for link in message["downloads"]:

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

if question:

    # -----------------------------
    # SHOW USER MESSAGE
    # -----------------------------

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # -----------------------------
    # CALL AI
    # -----------------------------

    with st.chat_message("assistant"):

        with st.spinner("Searching official documents..."):

            try:

                response = ask_question(question)

            except Exception as e:

                st.error(f"Error: {e}")

                st.stop()

        answer = response.get(
            "answer",
            "No response generated."
        )

        st.markdown(answer)

        # -------------------------
        # SOURCES
        # -------------------------

        sources = response.get(
            "sources",
            []
        )

        if sources:

            st.markdown("### 📄 Sources")

            for s in sources:

                st.markdown(
                    f"""
**Document**

{s["source"]}

**Page**

{s["page"]}

---
"""
                )

        # -------------------------
        # DOWNLOAD LINKS
        # -------------------------

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


    # -----------------------------
    # SAVE CHAT HISTORY
    # -----------------------------

    st.session_state.messages.append({

        "role": "assistant",

        "content": answer,

        "sources": sources,

        "downloads": downloads

    })
