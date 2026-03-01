# app/schemas.py
from pydantic import BaseModel
from typing import List, Optional

class PlanResponse(BaseModel):
    plan: List[str]

class SummaryResponse(BaseModel):
    summary: str

class ActionItemsResponse(BaseModel):
    action_items: List[str]

class RAGRequest(BaseModel):
    question: str

class Citation(BaseModel):
    id: int
    source: str
    page: Optional[int] = None
    excerpt: str

class RAGAnswerResponse(BaseModel):
    question: str
    answer: str
    summary_3lines: List[str]
    citations: List[Citation]

class RAGLLMOut(BaseModel):
    answer: str
    summary_3lines: List[str]