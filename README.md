# 🧠 Gemini RAG Brain: The Conversational Knowledge Engine

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![Gemini](https://img.shields.io/badge/AI-Google%20Gemini-blueviolet)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B)

**Gemini RAG Brain** is a Retrieval-Augmented Generation (RAG) application built with Streamlit and Google Gemini. It allows users to upload PDF or TXT documents and chat with them using natural language. The system retrieves relevant context from your documents to provide accurate, source-backed answers.

## 📸 Screenshots

| 1. Application UI | 2. Document Processing |
| ----------------- | ---------------------- |
| ![Opening UI](screenshots/RAG-UI1.png) | ![Processing](screenshots/RAG-UI2.png) |

| 3. AI Response | 4. Source Citation |
| -------------- | ------------------ |
| ![Q&A](screenshots/RAG-UI3.png) | ![Sources](screenshots/RAG-UI4.png) |

## ✨ Key Features

* **📄 Multi-Format Support:** Upload and process multiple `.pdf` or `.txt` files simultaneously.
* **🤖 Advanced LLM Integration:** Powered by Google's **Gemini 2.5 Flash** (with fallback support for newer models).
* **🔍 Vector Search:** Uses `HuggingFaceEmbeddings` (all-MiniLM-L6-v2) and `ChromaDB` for efficient semantic search.
* **📚 Source Citation:** Every answer includes an expandable "View Sources" section, showing exactly which document and page text was used.
* **🔒 Privacy-Friendly:** Documents are processed in temporary storage and automatically forgotten after a short period.
* **💬 Interactive Chat:** Familiar chat interface with session history.

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/karthik-k11/RAG-Chatbot-Project.git](https://github.com/karthik-k11/RAG-Chatbot-Project.git)
cd RAG-Chatbot-Project
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
```
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

Create a file named `.streamlit/secrets.toml` in your project root:

```toml
# .streamlit/secrets.toml
GEMINI_API_KEY = "your_google_api_key_here"
```
## 🚀 Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

1. Enter API Key: If not set in secrets, the app will prompt for configuration.

2. Upload Documents: Use the sidebar to upload PDFs or Text files.

3. Process: Click "Process Documents" to generate embeddings.

4. Chat: Ask questions about the content of your documents!

### 🧠 How It Works
1. Ingestion: Documents are loaded and split into smaller text chunks (1000 characters).

2. Embedding: Each chunk is converted into a numerical vector using HuggingFace embeddings (all-MiniLM-L6-v2).

3. Storage: Vectors are stored in an in-memory Chroma vector database.

4. Retrieval: When you ask a question, the system finds the most similar chunks in the database.

5. Generation: The relevant chunks + your question are sent to the Gemini model, which generates a natural language response based only on the provided context.

### Dependencies

```streamlit```

```langchain```

```langchain-google-genai```

```langchain-huggingface```

```chromadb```

```pypdf```

### 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements.