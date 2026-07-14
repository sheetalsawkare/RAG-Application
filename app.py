import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_mistralai import (
    ChatMistralAI,
    MistralAIEmbeddings
)

from langchain_chroma import Chroma

from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

st.set_page_config(
    page_title="PDF RAG Chat",
    page_icon="📄",
    layout="wide"
)

st.title("📄 PDF RAG Chatbot")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp_file:

        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    with st.spinner("Processing PDF..."):

        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )

        chunks = splitter.split_documents(docs)

        embedding_model = MistralAIEmbeddings()

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory="chroma-db"
        )

        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 4,
                "fetch_k": 10,
                "lambda_mult": 0.5
            }
        )

    st.success("PDF Processed Successfully")

    query = st.text_input(
        "Ask a question about the PDF"
    )

    if query:

        docs = retriever.invoke(query)

        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )

        prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant."
                ),
                (
                    "human",
                    "Context:\n{context}\n\nQuestion:\n{question}"
                )
            ]
        )

        final_prompt = prompt_template.invoke(
            {
                "context": context,
                "question": query
            }
        )

        llm = ChatMistralAI(
            model="mistral-small-2603",
            temperature=0
        )

        with st.spinner("Generating answer..."):
            response = llm.invoke(final_prompt)

        st.subheader("Answer")
        st.write(response.content)