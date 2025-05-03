from langchain_community.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import streamlit as st
from bs4 import BeautifulSoup


llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.1,
)

answers_prompt = ChatPromptTemplate.from_template(
    """
  Using ONLY the following context answer the user's question.
  If you don't just say you don't know, don't make anything up.
  
  Then, give a score to the answer between 0 and 5.
  If the answer answers the user question the score should be high,
  else it should be low.
  Make sure to always include the answer's score even if it's 0.
  
  Context: {context}
  
  Exmaple:
  Question: How far away is the moon?
  Answer: the moon is 384,400 km away from the earth.
  Score: 5

  Question: How far away is the sun?
  Answer: I don't know
  Score: 0
  Your turn!
  Question: {question}
  """
)


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke({
                    "context": doc.page_content,
                    "question": question,
                }).content,
                "source": doc.metadata["source"],
                "date": doc.metadata.get("lastmod", "N/A"),
            } for doc in docs
        ]}


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Use ONLY the following pre-existing answers to answer the user's question.
Use the answers that have the highest score (more helpful)
and favor the most recent ones.

Return the sources of the answers as they are, do not change them.

Answer:{answers}
      """,
        ),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    question = inputs["question"]
    answers = inputs["answers"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\n"
        f"Source:{answer['source']}\n"
        f"Date:{answer['date']}\n"
        for answer in answers
    )
    return choose_chain.invoke({
        "question": question,
        "answers": condensed,
    })


def parse_page(soup: BeautifulSoup):
    header = soup.find("header")
    footer = soup.find("footer")
    nav = soup.find_all("nav")

    if header:
        header.decompose()
    if footer:
        footer.decompose()
    if nav:
        for n in nav:
            n.decompose()

    # Clean up the text content
    text = str(soup.get_text())
    text = text.replace("\n", " ")

    # Remove navigation and forum text patterns
    text = text.replace(r'Previous:.*?Next:.*?forum', '')
    text = text.replace(r'Still have questions\?.*?Streamlit experts\.', '')

    return text


@st.cache_resource(show_spinner="Loading website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    loader = SitemapLoader(
        url,
        filter_urls=[r"^(.*\/get-started\/).*",],
        parsing_function=parse_page,
    )
    loader.requests_per_second = 1
    docs = loader.load_and_split(text_splitter=splitter)
    vector_store = FAISS.from_documents(docs, embedding=OpenAIEmbeddings())
    return vector_store.as_retriever()


st.set_page_config(
    page_title="SiteGPT",
    page_icon=":material/language:",
)


st.markdown(
    """
  # SiteGPT
  
  Ask questions about the content of a website.
  
  Start by writing the URL of the website on the sidebar.
  """
)

with st.sidebar:
    url = st.text_input(
        "Write down a URL",
        placeholder="https://example.com",
    )

if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a Sitemap URL")
    else:
        retriever = load_website(url)
        query = st.text_input("Ask a question to the website")
        if query:
            chain = {
                "docs": retriever,
                "question": RunnablePassthrough(),
            } | RunnableLambda(get_answers) | RunnableLambda(choose_answer)
            result = chain.invoke(query)
            st.write(result.content)
