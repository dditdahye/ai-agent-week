# app/schemas.py
from pydantic import BaseModel
from typing import List

class PlanResponse(BaseModel):
    plan: List[str]

class SummaryResponse(BaseModel):
    summary: str

class ActionItemsResponse(BaseModel):
    action_items: List[str]

class RAGAnswerResponse(BaseModel):
    answer: str