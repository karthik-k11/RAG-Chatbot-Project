from langchain_classic.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate


def build_qa_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    template = """
Use the following context to answer the question.
If unsure, say "I don’t know."

Context:
{context}

Question:
{question}

Answer:
"""
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

    return qa
