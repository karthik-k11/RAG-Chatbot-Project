import streamlit as st
import os
import tempfile
from google.colab import userdata

# --- Imports ---
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Page Config ---
st.set_page_config(page_title="Gemini RAG Brain", page_icon="🧠", layout="wide")
st.title("🧠 Gemini RAG: The Conversational Knowledge Engine")

# --- Initialize Session State ---
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
    st.header("⚙️ Configuration")
    
    # 1. Try to get key from Environment (Passed from Colab)
    if "GEMINI_API_KEY" in os.environ:
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
        st.success("✅ API Key Loaded (from Env)")
    
    # 2. Try to get key from Secrets (Direct Access)
    elif 'GEMINI_API_KEY' in userdata.keys():
        os.environ["GOOGLE_API_KEY"] = userdata.get('GEMINI_API_KEY')
        st.success("✅ API Key Loaded (from Secrets)")
        
    else:
        st.error("⚠️ API Key missing! Add it to Colab Secrets.")

    st.divider()
    st.header("📂 Document Management")
    uploaded_files = st.file_uploader("Upload New Documents", type=["pdf", "txt"], accept_multiple_files=True)
    process_btn = st.button("Save & Process Documents")

# --- Processing Logic ---
if process_btn and uploaded_files:
    with st.spinner("Saving to Drive and Processing..."):
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
            
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            embedding_model = HuggingFaceEmbeddings(model_name=model_name)
            
            persist_directory = './chroma_db_app'
            
            if st.session_state.vectorstore is None:
                st.session_state.vectorstore = Chroma.from_documents(
                    chunks, 
                    embedding_model, 
                    persist_directory=persist_directory
                )
            else:
                st.session_state.vectorstore.add_documents(chunks)
                
            st.success(f"✅ Successfully processed {len(chunks)} new chunks!")
        else:
            st.warning("No valid documents found.")

# --- Chat Logic ---
if st.session_state.vectorstore:
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=st.session_state.memory,
            return_source_documents=True,
            verbose=False
        )

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask a question..."):
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = qa_chain.invoke({"question": prompt})
                    answer = response['answer']
                    st.markdown(answer)
                    
                    with st.expander("📚 View Sources"):
                        for doc in response['source_documents']:
                            source = doc.metadata.get('source', 'Unknown')
                            st.caption(f"📄 **Source:** {os.path.basename(source)}")
                            st.text(doc.page_content[:200] + "...")
                    
                    st.session_state.messages.append({"role": "assistant", "content": answer})
    except Exception as e:
        st.error(f"Error: {e}. Check your API Key.")

else:
    st.info("👈 Upload a document to start chatting!")
