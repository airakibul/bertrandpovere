from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List, Union
import os, json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph,START, END

load_dotenv()
app = FastAPI()

# ------------------ Pydantic Schemas ------------------ #

class Question(BaseModel):
    index: int = Field(..., description="Question index",ge=1)
    question: str = Field(..., description="The question text")
    answer: Union[str, List[str]] = Field(..., description="Answer type or options")
    isRequired: bool = Field(..., description="Is the question required")

class Page(BaseModel):
    pageIndex: int = Field(..., description="Page number",ge=1)
    title: str = Field(..., description="Page title")
    questions: List[Question] = Field(..., description="Questions on the page")

class Survey(BaseModel):
    templateName: str = Field(..., description="Survey name")
    templateDisc: str = Field(..., description="Survey description")
    pages: List[Page] = Field(..., description="Survey pages")

# ------------------ LangGraph Setup ------------------ #

def build_graph():
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    def generate_survey(state):
      topic = state["topic"]
      prompt = f"""
      Generate an **audit or research-type survey** in valid JSON matching this schema:
      {{
        "templateName": "creative tamplate name <topic>",
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
      - Use varied answer types ("text", "numeric", or choice arrays like ["Yes", "No", "N/A"])
      - JSON only, no markdown fences
      """
      resp = llm.invoke(prompt)
      return {"survey": resp.content}


    graph = StateGraph(dict)
    graph.add_node("generate", generate_survey)
    graph.set_entry_point("generate")
    graph.add_edge(START,"generate")
    graph.add_edge("generate", END)
    return graph.compile()

survey_graph = build_graph()

# ------------------ API Endpoints ------------------ #

class TopicRequest(BaseModel):
    topic: str

@app.post("/generate-survey", response_model=Survey)
async def generate_survey(req: TopicRequest):
    result = survey_graph.invoke({"topic": req.topic})
    raw = result["survey"].strip()

    # Clean if fenced
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[len("json"):].strip()

    try:
        data = json.loads(raw)
        return Survey(**data)  # Pydantic validation here
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Invalid survey JSON: {e}")
