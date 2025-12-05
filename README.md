# 🧠 Gemini RAG Brain: The Conversational Knowledge Engine

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![Gemini](https://img.shields.io/badge/AI-Google%20Gemini-blueviolet)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B)

**Gemini RAG Brain** is a Retrieval-Augmented Generation (RAG) application built with Streamlit and Google Gemini. It allows users to upload PDF or TXT documents and chat with them using natural language. The system retrieves relevant context from your documents to provide accurate, source-backed answers.

## ✨ Key Features

* **📄 Multi-Format Support:** Upload and process multiple `.pdf` or `.txt` files simultaneously.
* **🤖 Advanced LLM Integration:** Powered by Google's **Gemini 1.5 Flash** (with fallback support for newer models).
* **🔍 Vector Search:** Uses `HuggingFaceEmbeddings` (all-MiniLM-L6-v2) and `ChromaDB` for efficient semantic search.
* **📚 Source Citation:** Every answer includes an expandable "View Sources" section, showing exactly which document and page text was used.
* **🔒 Secure Processing:** Uses temporary directories for file processing to ensure no user data is permanently stored on the server.
* **💬 Interactive Chat:** Familiar chat interface with session history.

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/gemini-rag-brain.git](https://github.com/your-username/gemini-rag-brain.git)
cd gemini-rag-brain
```

### 2. Create a Virtual Environment
python -m venv venv
#### Windows
```bash
venv\Scripts\activate
```
#### Mac/Linux
```bash
source venv/bin/activate
```
### 3. Install Dependencies
```bash
pip install streamlit langchain-google-genai langchain-community langchain-huggingface chromadb pypdf python-dotenv
```
## 🔑 Configuration

You need a Google API Key to access the Gemini models.

### Option A: Local Development (Secrets File)
Create a file named `.streamlit/secrets.toml` in your project root:

```toml
# .streamlit/secrets.toml
GEMINI_API_KEY = "your_google_api_key_here"
```