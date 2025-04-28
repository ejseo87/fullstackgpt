import streamlit as st
import os
from langchain.retrievers import WikipediaRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.chat_models import ChatOpenAI

st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)
st.title("QuizGPT")

# Initialize docs as None
docs = None
# 니꼬샘은 1106 사용, 쬐끔 더 비쌈
llm = ChatOpenAI(
    model="gpt-3.5-turbo-0125",
    temperature=0.1,
)


@st.cache_resource(show_spinner="Splitting file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    # Save the file
    with open(file_path, "wb") as f:
        f.write(file_content)
    st.success(f"File [{file.name}] uploaded successfully!")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


# Create necessary directories
os.makedirs("./.cache/quiz_files", exist_ok=True)

with st.sidebar:
    choice = st.selectbox(
        "Choose what you want to use.",
        (
            "File",
            "Wikepedia Article",
        )
    )
    if choice == "File":
        file = st.file_uploader(
            "Upload a .docx, .txt, or .pdf file",
            type=["txt", "pdf", "docx"]
        )
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("Search Wikipedia for...")
        if topic:
            retriever = WikipediaRetriever(top_k_results=5)
            with st.status("Searching Wikipedia..."):
                docs = retriever.get_relevant_documents(topic)

if docs is None:
    st.markdown(
        """
      Welcome to QuizGPT!
      
      I will make a quiz from Wikipedia articles
      or files you upload to test your knowledge and help you study.
      
      Get started by uploading a file or searching Wikipedia in the sidebar.
      """
    )
else:
    st.write(docs)
