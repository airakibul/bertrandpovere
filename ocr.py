import os
import io
import base64
import re
from dotenv import load_dotenv
from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from PyPDF2 import PdfReader
from docx import Document
import fitz  # PyMuPDF
from openai import OpenAI
from langgraph.graph import StateGraph, END
from PIL import Image

# ------------------- Load Environment ------------------- #
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

# ------------------- Pydantic Schema ------------------- #
class OCRResponse(BaseModel):
    extracted_text: str

# ------------------- Cleaning Function ------------------- #
def clean_text(text: str) -> str:
    text = re.sub(r'---+', '', text)
    text = re.sub(r'The text from the handwritten note is:|Here\'s the extracted text from the handwritten note:', '', text)
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    return text.strip()

# ------------------- OCR Functions ------------------- #
def extract_text_from_image_bytes(file_bytes: bytes) -> str:
    image_base64 = base64.b64encode(file_bytes).decode("utf-8")
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Extract the text from this handwritten note."},
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{image_base64}"}
            ]
        }]
    )
    return response.output_text

def extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    text = ""
    pdf_reader = PdfReader(io.BytesIO(file_bytes))
    for page in pdf_reader.pages:
        text += page.extract_text() or ""

    pdf_doc = fitz.open(stream=file_bytes, filetype="pdf")
    for page in pdf_doc:
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = pdf_doc.extract_image(xref)
            image_bytes = base_image["image"]
            try:
                text += f"\n{extract_text_from_image_bytes(image_bytes)}"
            except:
                pass
    return text

def extract_text_from_docx_bytes(file_bytes: bytes) -> str:
    text = ""
    doc = Document(io.BytesIO(file_bytes))
    for p in doc.paragraphs:
        text += p.text + "\n"

    for rel in doc.part._rels:
        rel_obj = doc.part._rels[rel]
        if "image" in rel_obj.target_ref:
            image_bytes = rel_obj.target_part.blob
            try:
                text += f"\n{extract_text_from_image_bytes(image_bytes)}"
            except:
                pass
    return text

# ------------------- LangGraph Nodes ------------------- #
def detect_file_type(state):
    filename = state["filename"]
    ext = filename.split('.')[-1].lower()
    state["ext"] = ext
    return state

def extract_text_node(state):
    file_bytes = state["file_bytes"]
    ext = state["ext"]

    if ext in ["jpg", "jpeg", "png"]:
        raw_text = extract_text_from_image_bytes(file_bytes)
    elif ext == "pdf":
        raw_text = extract_text_from_pdf_bytes(file_bytes)
    elif ext in ["doc", "docx"]:
        raw_text = extract_text_from_docx_bytes(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    state["raw_text"] = raw_text
    return state

def clean_text_node(state):
    state["extracted_text"] = clean_text(state["raw_text"])
    return state

# ------------------- Build LangGraph ------------------- #
graph = StateGraph(dict)
graph.add_node("detect_file_type", detect_file_type)
graph.add_node("extract_text", extract_text_node)
graph.add_node("clean_text", clean_text_node)

graph.set_entry_point("detect_file_type")
graph.add_edge("detect_file_type", "extract_text")
graph.add_edge("extract_text", "clean_text")
graph.add_edge("clean_text", END)

ocr_graph = graph.compile()

# ------------------- FastAPI Router ------------------- #
router = APIRouter()

@router.post("/ocr", response_model=OCRResponse)
async def run_ocr(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        state = {"file_bytes": file_bytes, "filename": file.filename}
        final_state = ocr_graph.invoke(state)
        return OCRResponse(extracted_text=final_state["extracted_text"])
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# ------------------- FastAPI App ------------------- #
app = FastAPI(title="OCR API with LangGraph")
app.include_router(router)




