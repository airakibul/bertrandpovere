import os
import io
import re
import json
import base64
from typing import List, Union

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from PyPDF2 import PdfReader
from docx import Document
import fitz  # PyMuPDF
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from PIL import Image

# ------------------- Load Environment ------------------- #
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

# ------------------- FastAPI App ------------------- #
app = FastAPI(title="OCR + Survey API")

# =========================================================
# =============== OCR SECTION =============================
# =========================================================

class OCRResponse(BaseModel):
    extracted_text: str

class Question(BaseModel):
    index: int = Field(..., description="Question index", ge=1)
    question: str = Field(..., description="The question text")
    answer: Union[str, List[str]] = Field(..., description="Answer type or options")
    isRequired: bool = Field(..., description="Is the question required")

class Page(BaseModel):
    pageIndex: int = Field(..., description="Page number", ge=1)
    title: str = Field(..., description="Page title")
    questions: List[Question] = Field(..., description="Questions on the page")

class Survey(BaseModel):
    templateName: str = Field(..., description="Survey name")
    templateDisc: str = Field(..., description="Survey description")
    pages: List[Page] = Field(..., description="Survey pages")

class OCRWithSurveyResponse(BaseModel):
    extracted_text: str
    survey: Survey

# ------------------- Text Cleaning ------------------- #
def clean_text(text: str) -> str:
    text = re.sub(r'---+', '', text)
    text = re.sub(r'The text from the handwritten note is:|Here\'s the extracted text from the handwritten note:', '', text)
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    return text.strip()

# ------------------- OCR Extraction ------------------- #
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

# ------------------- StateGraph Nodes ------------------- #
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

# ------------------- Build OCR Graph ------------------- #
ocr_graph = StateGraph(dict)
ocr_graph.add_node("detect_file_type", detect_file_type)
ocr_graph.add_node("extract_text", extract_text_node)
ocr_graph.add_node("clean_text", clean_text_node)
ocr_graph.set_entry_point("detect_file_type")
ocr_graph.add_edge("detect_file_type", "extract_text")
ocr_graph.add_edge("extract_text", "clean_text")
ocr_graph.add_edge("clean_text", END)
ocr_graph = ocr_graph.compile()

# =========================================================
# =============== SURVEY GENERATION =====================
# =========================================================

def build_survey_graph():
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    def generate_survey(state):
        topic = state["topic"]
        prompt = f"""
        Generate an **audit or research-type survey** in valid JSON matching this schema:
        {{
          "templateName": "creative template name based on the text",
          "templateDisc": "One paragraph describing the audit or research purpose and scope",
          "pages": [
            {{
              "pageIndex": int,
              "title": str,
              "questions": [
                {{
                  "index": int,
                  "question": str,
                  "answer": str or [str],
                  "isRequired": bool
                }}
              ]
            }}
          ]
        }}
        Rules:
        - Exactly 12 questions
        - 4 pages, 3 questions per page
        - Focus on compliance, process checks, risk assessment, or internal controls
        - Base the questions on the following text: {topic[:500]}...
        - Use varied answer types ("text", "numeric", or choice arrays like ["Yes", "No", "N/A"])
        - JSON only, no markdown fences
        """
        resp = llm.invoke(prompt)
        return {"survey": resp.content}

    graph = StateGraph(dict)
    graph.add_node("generate", generate_survey)
    graph.set_entry_point("generate")
    graph.add_edge(START, "generate")
    graph.add_edge("generate", END)
    return graph.compile()

survey_graph = build_survey_graph()

# =========================================================
# =============== OCR + SURVEY ENDPOINT ==================
# =========================================================

@app.post("/ocr", response_model=OCRWithSurveyResponse)
async def run_ocr_with_survey(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        state = {"file_bytes": file_bytes, "filename": file.filename}
        final_state = ocr_graph.invoke(state)
        extracted_text = final_state["extracted_text"]

        # Generate survey from extracted text
        survey_result = survey_graph.invoke({"topic": extracted_text})
        raw_survey = survey_result["survey"].strip()

        # Clean JSON string if it has markdown fences
        if raw_survey.startswith("```"):
            raw_survey = raw_survey.split("```")[1]
            if raw_survey.startswith("json"):
                raw_survey = raw_survey[len("json"):].strip()

        survey_data = json.loads(raw_survey)

        return OCRWithSurveyResponse(
            extracted_text=extracted_text,
            survey=Survey(**survey_data)
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating survey: {e}")
