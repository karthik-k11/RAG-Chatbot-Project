import time
import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .Privacy import redact, make_doc_id

#Document Splitter
def sanitize_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    sanitized = []
    now = time.time()

    for i, doc in enumerate(chunks):
        clean_text = redact(doc.page_content)
        seed = f"{doc.metadata.get('source','')}_{now}_{i}"
        doc_id = make_doc_id(seed)

        #Keep minimal, non-sensitive metadata
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", None)

        sanitized.append(
            Document(
                page_content=clean_text,
                metadata={
                    "doc_id": doc_id,
                    "source": os.path.basename(src) if src != "unknown" else "unknown",
                    "page": page,
                },
            )
        )

    return sanitized

