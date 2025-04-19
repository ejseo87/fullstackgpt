import streamlit as st
import os
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.chat_models import ChatOpenAI

st.set_page_config(
    page_title="DocumentGPT",
    page_icon=":memo:",
)

llm = ChatOpenAI(temperature=0.1)


@st.cache_resource(show_spinner="Embedding file...")
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
    loader = UnstructuredFileLoader(file_path)
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


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
        if save:
            st.session_state["messages"].append(
                {"role": role, "message": message})


def paint_history():
    for message in st.session_state.messages:
        send_message(message["message"], message["role"], save=False)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


prompt = ChatPromptTemplate.from_messages([
    ("system",
     """
     Answer the question using ONLY the following context.
     If you don't know the answer, say "I don't know".
     Don't make anything up.
     ---
     Context: {context}
     """,
     ),
    ("human", "{question}"),
])

st.title("DocumentGPT")

st.markdown("""
Welcome!

Use this chatbot to ask questions to an AI about your file!

Upload your files on the sidebar.
""")

# Create necessary directories
os.makedirs("./.cache/files", exist_ok=True)
os.makedirs("./.cache/embeddings", exist_ok=True)

with st.sidebar:
    file = st.file_uploader("Upload a .txt .pdf or .docx file", type=[
        "pdf", "docx", "txt"])

if file:
    retriever = embed_file(file)
    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about the document")
    if message:
        send_message(message, "human")
        chain = {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        } | prompt | llm
        response = chain.invoke(message)
        send_message(response.content, "ai")
else:
    st.session_state["messages"] = []
