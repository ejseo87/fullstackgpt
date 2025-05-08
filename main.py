from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from pinecone import Pinecone as PineconeClient
from dotenv import load_dotenv
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
    
load_dotenv()
# Initialize Pinecone client
pc = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))


embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = PineconeVectorStore.from_existing_index(
    index_name="recipes",
    embedding=embeddings,
)


app = FastAPI(
    title="ChefGPT. The best provider of recipes",
    description="Give ChefGPT a couple of ingredients and it will give you recipes in return.",
    servers=[
        {
            "url": "https://hb-spokesman-implement-rx.trycloudflare.com",
        }
    ]
)


class Document(BaseModel):
    page_content: str = Field(description="The page content of the recipe.")

# /recipes?ingredient=tofu


@app.get(
    "/recipes",
    summary="Return a list of recipes",
    description="Upon receiving an ingredient, this endpoint will return a list of recipes that contain that ingredient.",
    response_description="A Document object that contains the recipes and reparation instructions.",
    response_model=list[Document],
    openapi_extra={
        "x-openai-isConsequential": False,
    }
)
def get_recipe(ingredient: str):
    docs = vectorstore.similarity_search(ingredient)
    return docs


# fake user token db
user_token_db = {
    "ABCDEF": "nico"
}


@app.get(
    "/authorize",
    response_class=HTMLResponse,
    include_in_schema=False,
)
def handle_authorize(client_id: str, redirect_uri: str, state: str):
    return f"""
    <html>
        <head>
            <title>Nicolacus Maximus Log In</title>
        </head>
        <body>
            <h1>Log In Nicolacus Maximus</h1>
            <a href="{redirect_uri}?code=ABCDEF&state={state}">Authorize Nicolacus Maximus GPT</a>
        </body>
    </html>
    """


@app.post(
    "/token",
    include_in_schema=False,
)
def handle_token(code=Form(...)):
    print(f"code: {code}")
    return {
        "access_token": user_token_db[code],
    }
