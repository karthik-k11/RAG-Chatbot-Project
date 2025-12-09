import time
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .Privacy import redact, make_doc_id


def sanitize_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    sanitized = []
    now = time.time()

    for i, doc in enumerate(chunks):
        clean_text = redact(doc.page_content)
        seed = f"{doc.metadata.get('source','')}_{now}_{i}"
        doc_id = make_doc_id(seed)

        sanitized.append(
            Document(
                page_content=clean_text,
                metadata={"doc_id": doc_id},
            )
        )

    return sanitized
