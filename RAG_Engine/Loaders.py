import tempfile
import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader

def safe_text_load(path):
    """Try multiple encodings for TXT files."""
    encodings = ["utf-8", "utf-16", "latin-1", "cp1252"]

    for enc in encodings:
        try:
            loader = TextLoader(path, encoding=enc)
            docs = loader.load()
            if docs:
                return docs
        except Exception:
            continue
    return []


def load_uploaded_files(files):
    """Load PDF/TXT documents reliably with fallback methods."""
    all_docs = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for uf in files:
            path = os.path.join(tmpdir, uf.name)

            # Write uploaded file to disk
            with open(path, "wb") as f:
                f.write(uf.getvalue())

            # Detect file type
            is_pdf = uf.name.lower().endswith(".pdf")
            is_txt = uf.name.lower().endswith(".txt")

            try:
                if is_pdf:
                    loader = PyPDFLoader(path)
                    docs = loader.load()

                    # PDF had no text → scanned or empty
                    if not docs or all(d.page_content.strip() == "" for d in docs):
                        print("PDF contains no extractable text.")
                        continue

                    all_docs.extend(docs)

                elif is_txt:
                    docs = safe_text_load(path)
                    if docs:
                        all_docs.extend(docs)
                    else:
                        print("TXT file unreadable via all encodings.")
                        continue

            except Exception as e:
                print("Error while loading:", e)
                continue

    return all_docs