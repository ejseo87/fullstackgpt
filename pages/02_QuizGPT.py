import streamlit as st
import os
from langchain.retrievers import WikipediaRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler

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
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
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


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


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

if not docs:
    st.markdown(
        """
      Welcome to QuizGPT!

      I will make a quiz from Wikipedia articles
      or files you upload to test your knowledge and help you study.

      Get started by uploading a file or searching Wikipedia in the sidebar.
      """
    )
else:

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
    You are a helpful assistant that is role playing as a teacher.

    Based ONLY on the following context make 10 questions
    to test the user's knowledge about the text.

    Each question should have 4 answers, three of them must be incorrect
    and one should be correct.

    Use (o) to signal the correct answer.

    Question examples:

    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)

    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut

    Question: When was Avatar released?
    Answers: 2007|2001|2009(o)|1998

    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(o)|Painter|Actor|Model

    Your turn!

    Context: {context}
""",
            )
        ]
    )

    chain = {
        "context": format_docs
    } | prompt | llm
    start = st.button("Generate Quiz")
    if start:
        with st.spinner("Generating quiz..."):
            response = chain.invoke(docs)
            st.write(response)
