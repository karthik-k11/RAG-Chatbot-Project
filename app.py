import streamlit as st
import os
import tempfile

# Standard imports
from langchain_classic.chains import RetrievalQA 
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# Page Confi
st.set_page_config(page_title="Gemini RAG Brain", page_icon="🧠", layout="wide")
st.title("🧠 Gemini RAG: Local Knowledge Engine")

#Initialize Session State
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )

# --- Sidebar: API Key & Upload ---
with st.sidebar:
    st.header("Configuration")
    
    # Load API Key from secrets.toml
    try:
        if "GEMINI_API_KEY" in st.secrets:
            os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]
            st.success("API Key Loaded")
        else:
            st.error("Key missing in secrets.toml")
    except FileNotFoundError:
        st.error(" .streamlit/secrets.toml not found!")

    st.divider()
    st.header(" Document Management")
    uploaded_files = st.file_uploader("Upload New Documents", type=["pdf", "txt"], accept_multiple_files=True)
    process_btn = st.button("Save & Process Documents")

# --- Processing Logic ---
if process_btn and uploaded_files:
    with st.spinner("Processing documents..."):
        all_documents = []
        DATA_FOLDER = './data'
        os.makedirs(DATA_FOLDER, exist_ok=True)

        for uploaded_file in uploaded_files:
            file_path = os.path.join(DATA_FOLDER, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            try:
                if uploaded_file.name.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                else:
                    loader = TextLoader(file_path)
                all_documents.extend(loader.load())
            except Exception as e:
                st.error(f"Error loading {uploaded_file.name}: {e}")

        if all_documents:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(all_documents)
            
            # Create Embeddings & Vector Store
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            embedding_model = HuggingFaceEmbeddings(model_name=model_name)
            
            # --- IN-MEMORY MODE (No persist_directory) ---
            if st.session_state.vectorstore is None:
                st.session_state.vectorstore = Chroma.from_documents(
                    chunks, 
                    embedding_model
                    # No directory = RAM only
                )
            else:
                st.session_state.vectorstore.add_documents(chunks)
                
            st.success(f" Learned {len(chunks)} new knowledge chunks!")
        else:
            st.warning("No valid documents found.")

# --- Chat Logic ---
if st.session_state.vectorstore:
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    # Define Prompt
    template = """
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    Context: {context}
    Question: {question}
    Helpful Answer:
    """
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    # Standard RetrievalQA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        return_source_documents=True
    )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Standard invoke
                response = qa_chain.invoke({"query": prompt})
                answer = response['result']
                st.markdown(answer)
                
                with st.expander(" View Sources"):
                    for doc in response['source_documents']:
                        source = os.path.basename(doc.metadata.get('source', 'Unknown'))
                        st.caption(f" **Source:** {source}")
                        st.text(doc.page_content[:200] + "...")
                
                st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info(" Upload a document to start chatting!")