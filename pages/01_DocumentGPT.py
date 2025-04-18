import streamlit as st
import os
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS

st.set_page_config(
    page_title="DocumentGPT",
    page_icon=":memo:",
)


def embed_file(file):
    file_content = file.read()
    file_path = os.path.join(".cache", "files", file.name)
    # Save the file
    with open(file_path, "wb") as f:
        f.write(file_content)
    st.success(f"File [{file.name}] uploaded successfully!")
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader("./files/chapter_one.txt")
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings, cache_dir)
    vectorstore = FAISS.from_documents(
        documents=docs,
        embedding=cached_embeddings
    )
    retriever = vectorstore.as_retriever()
    return retriever


st.title("DocumentGPT")

st.markdown("""
Welcome!

Use this chatbot to ask questions to an AI about your file!
""")

# Create necessary directories
os.makedirs("./.cache/files", exist_ok=True)
os.makedirs("./.cache/embeddings", exist_ok=True)

file = st.file_uploader("Upload a .txt .pdf or .docx file", type=[
                        "pdf", "docx", "txt"])

if file:
    retriever = embed_file(file)
    docs = retriever.invoke("whale")
    st.write(docs)
