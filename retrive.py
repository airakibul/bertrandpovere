import os
import tempfile
import re
from typing import TypedDict, List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PyPDF2 import PdfReader
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone

from langgraph.graph import StateGraph, START, END

# ------------------------------
# ENV + Clients
# ------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

app = FastAPI()

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "battro"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY)


# ------------------------------
# State schema
# ------------------------------
class State(TypedDict):
    pdf_path: str
    text: str
    chunks: List[str]
    retrieved_docs: List[str]
    recommendations: str


# ------------------------------
# Node functions
# ------------------------------
def extract_text(state: State) -> State:
    reader = PdfReader(state["pdf_path"])
    text = "".join([page.extract_text() or "" for page in reader.pages])
    state["text"] = text
    return state

def split_text(state: State) -> State:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    state["chunks"] = splitter.split_text(state["text"])
    return state

def retrieve(state: State) -> State:
    query = "\n".join(state["chunks"])
    results = docsearch.similarity_search(query, k=5)
    state["retrieved_docs"] = [r.page_content for r in results]
    return state

def generate(state: State) -> State:
    if not state["retrieved_docs"]:
        state["recommendations"] = "No similar content found in the database."
        return state

    context = "\n\n".join(state["retrieved_docs"])
    prompt = f"""Based on the following document context, give the points:
                - give a standard of the oudit.
                - give the positive points.
                - give the negetive points.
                - give the suggestion to imporve.
                \n\n{context}"""
    response = llm.invoke(prompt)
    raw_text = response.content

    # Clean text
    cleaned_text = re.sub(r'\n+', ' ', raw_text)
    cleaned_text = re.sub(r'\d+\.\s*', '', cleaned_text)
    cleaned_text = cleaned_text.strip()

    state["recommendations"] = cleaned_text
    return state


# ------------------------------
# LangGraph definition
# ------------------------------
workflow = StateGraph(State)

workflow.add_node("extract", extract_text)
workflow.add_node("split", split_text)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

workflow.add_edge(START, "extract")
workflow.add_edge("extract", "split")
workflow.add_edge("split", "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

graph_app = workflow.compile()


# ------------------------------
# FastAPI route
# ------------------------------
@app.post("/recommend")
async def recommend(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        initial_state: State = {"pdf_path": tmp_path}

        final_state = graph_app.invoke(initial_state)

        return JSONResponse(content={"recommendations": final_state["recommendations"]})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
