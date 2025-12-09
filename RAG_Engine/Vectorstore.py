import shutil
import time
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def initialize_vectorstore(docs):
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma.from_documents(docs, embedding)


def add_documents(vs, docs):
    vs.add_documents(docs)


def clear_vectorstore(state):
    vs = state.get("vectorstore")

    if vs and hasattr(vs, "persist_directory"):
        persist = vs.persist_directory
        if persist and os.path.exists(persist):
            shutil.rmtree(persist, ignore_errors=True)

    state["vectorstore"] = None
    state["kb_created_at"] = None


def check_retention_expiry(state):
    vs = state.get("vectorstore")
    created = state.get("kb_created_at")

    if not vs or not created:
        return

    elapsed = (time.time() - created) / 3600
    if elapsed > state.get("kb_retention_hours", 24):
        clear_vectorstore(state)
