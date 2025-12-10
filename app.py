import streamlit as st
import time
import os

from RAG_Engine.Loaders import load_uploaded_files
from RAG_Engine.Preprocess import sanitize_documents
from RAG_Engine.Vectorstore import (
    initialize_vectorstore,
    add_documents,
    clear_vectorstore,
    check_retention_expiry,
)
from RAG_Engine.qa import build_qa_chain

#Stramlit Page Config
st.set_page_config(
    page_title="Gemini RAG Brain",
    page_icon="🧠",
    layout="wide"
)
st.title("🧠 Gemini RAG: Local Knowledge Engine")

#Session State Init
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "kb_created_at" not in st.session_state:
    st.session_state.kb_created_at = None

if "kb_retention_hours" not in st.session_state:
    st.session_state.kb_retention_hours = 24

#Sidebar Config

api_key_ok = False
debug = False

with st.sidebar:
    st.header("Configuration")

    # Loading API key
    if "GEMINI_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]
        st.success("Gemini API Key Loaded")
        api_key_ok = True
    else:
        st.error("Missing `GEMINI_API_KEY` in .streamlit/secrets.toml")

    #Optional debug toggle
    debug = st.checkbox("Show debug info", value=False)

    st.divider()
    st.header("Document Upload")
    uploaded_files = st.file_uploader(
        "Upload PDF/TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )
    process_btn = st.button("Process Documents")

    st.divider()
    st.header("Retention")
    retention = st.number_input(
        "KB retention (hours)",
        min_value=1,
        max_value=24 * 365,
        value=st.session_state.kb_retention_hours,
    )
    st.session_state.kb_retention_hours = retention

    if st.session_state.vectorstore:
        if st.button("Delete Knowledge Base"):
            clear_vectorstore(st.session_state)
            st.success("Knowledge base deleted.")

if not api_key_ok:
    st.info("Add your Gemini API key in .streamlit/secrets.toml to start using the app.")
    st.stop()

#Auto Retention Expiry
check_retention_expiry(st.session_state)

if process_btn and uploaded_files:

    with st.spinner("Loading and processing documents..."):

        raw_docs = load_uploaded_files(uploaded_files)

        if debug:
            st.write("DEBUG: loaded docs:", raw_docs)

        if not raw_docs:
            st.warning("No readable text found in uploaded files.")
        else:
            sanitized = sanitize_documents(raw_docs)

            if st.session_state.vectorstore is None:
                st.session_state.vectorstore = initialize_vectorstore(sanitized)
                st.session_state.kb_created_at = time.time()
            else:
                add_documents(st.session_state.vectorstore, sanitized)
                st.session_state.kb_created_at = time.time()

            st.success(f"Indexed {len(sanitized)} sanitized knowledge chunks.")
            st.info("Knowledge Base Updated!")

#Chat Interface
if not st.session_state.vectorstore:
    st.info("Upload documents to start chatting!")
    st.stop()

qa_chain = build_qa_chain(st.session_state.vectorstore)

#Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

#Chat input
if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = qa_chain.invoke({"query": prompt})
            answer = result["result"]
            st.markdown(answer)

            with st.expander("Sources"):
                for doc in result["source_documents"]:
                    st.caption(
                        f"{doc.metadata.get('source','unknown')} | page {doc.metadata.get('page','?')} | id: {doc.metadata.get('doc_id', 'unknown')}"
                    )
                    st.text(doc.page_content[:200] + "...")


    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )