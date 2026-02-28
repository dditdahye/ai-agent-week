# app/service.py
import json
import time
from uuid import uuid4

from .schemas import PlanResponse, RAGAnswerResponse
from .tools import summarize, extract_action_items
from .llm_client import parse_with_retry

from .config import MODEL_NAME, MAX_TASK_CHARS

MAX_TASK_CHARS = 20000  # ✅ 실무 가드레일(원하면 조정)


def planner(task: str) -> list[str]:
    prompt = f"""
You are an AI planning assistant.

Rules:
- Always write the plan steps in English.
- ALWAYS include this exact step as Step 1: "Summarize the text"
- If the task asks for action items, include this exact step: "Extract action items"
- Keep steps short and actionable.

User task:
{task}
"""
    parsed = parse_with_retry(
        model=MODEL_NAME,
        input=prompt,
        text_format=PlanResponse,
        max_output_tokens=200,
        max_attempts=2,
    )
    return [s.strip() for s in parsed.plan if s.strip()]


def executor(plan: list[str], task: str) -> dict:
    outputs: dict = {}

    for step in plan:
        s = step.lower()

        if s == "summarize the text":
            outputs["summary"] = summarize(task)

        if s == "extract action items":
            outputs["action_items"] = extract_action_items(task)

    return outputs


def run_agent(task: str) -> dict:
    if not task or not task.strip():
        raise ValueError("task is required")

    if len(task) > MAX_TASK_CHARS:
        raise ValueError(f"task is too long (>{MAX_TASK_CHARS} chars)")

    request_id = str(uuid4())
    t0 = time.time()

    plan = planner(task)
    outputs = executor(plan, task)

    latency_ms = int((time.time() - t0) * 1000)

    # ✅ 구조화 로그(간단 버전)
    print(json.dumps({
        "event": "agent_run",
        "request_id": request_id,
        "model": MODEL_NAME,
        "latency_ms": latency_ms,
        "has_summary": "summary" in outputs,
        "action_items_count": len(outputs.get("action_items", [])) if isinstance(outputs.get("action_items"), list) else 0,
    }, ensure_ascii=False))

    return {
        "request_id": request_id,
        "model": MODEL_NAME,
        "plan": plan,
        "outputs": outputs,
        "latency_ms": latency_ms,
    }


def rag_answer(question: str, context: str) -> str:
    prompt = f"""
You are a helpful assistant.
Answer the question using ONLY the context below.
If the context does not contain the answer, say you don't know.

Context:
{context}

Question:
{question}
"""
    parsed = parse_with_retry(
        model=MODEL_NAME,
        input=prompt,
        text_format=RAGAnswerResponse,
        max_output_tokens=400,
        max_attempts=2,
    )
    return parsed.answer.strip()