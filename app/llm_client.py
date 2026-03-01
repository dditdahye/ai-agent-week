# app/llm_client.py
from __future__ import annotations

import random
import time
from typing import Type, TypeVar

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError, APIStatusError
from app.schemas import RAGLLMOut

load_dotenv()

T = TypeVar("T")

# 실무: client 단일화 + timeout 고정
client = OpenAI(timeout=20.0)  # 초 단위 (원하면 15~30 사이로)


def parse_with_retry(
    *,
    model: str,
    input: str,
    text_format: Type[T],
    max_output_tokens: int,
    max_attempts: int = 2,
) -> T:
    """
    Structured Output(parse) 호출을 실무형으로 감싼 함수.
    - 429 / 네트워크 / 5xx만 retry
    - 지수 백오프 + 약간의 지터
    """
    last_err: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            resp = client.responses.parse(
                model=model,
                input=input,
                text_format=text_format,
                max_output_tokens=max_output_tokens,
            )
            return resp.output_parsed

        except RateLimitError as e:
            last_err = e
            _backoff_sleep(attempt)

        except (APITimeoutError, APIConnectionError) as e:
            last_err = e
            _backoff_sleep(attempt)

        except APIStatusError as e:
            # 5xx만 retry (4xx는 바로 실패)
            last_err = e
            status = getattr(e, "status_code", None)
            if status and status >= 500:
                _backoff_sleep(attempt)
            else:
                raise

    raise last_err if last_err else RuntimeError("Unknown LLM error")


def _backoff_sleep(attempt: int) -> None:
    base = 0.6  # seconds
    sleep_s = base * (2 ** (attempt - 1)) + random.uniform(0, 0.2)
    time.sleep(sleep_s)


def rag_answer_with_summary(question: str, context: str) -> RAGLLMOut:
    prompt = f"""
    You are a document-grounded assistant.

    Rules:
    - Use ONLY the provided context.
    - If insufficient, answer: "문서에서 해당 질문과 직접 관련된 근거를 찾지 못했습니다."
    - summary_3lines must be exactly 3 short lines in Korean.
    - summary_3lines must be exactly 3 Korean sentences summarizing the context (no bullet labels like '대표이사:').

    Context:
    {context}

    Question:
    {question}
    """

    out = parse_with_retry(
        model="gpt-4.1-mini",
        input=prompt,
        text_format=RAGLLMOut,
        max_output_tokens=450,
    )

    # 3줄 강제 보정
    lines = [s.strip() for s in (out.summary_3lines or []) if s.strip()]
    out.summary_3lines = (lines + [""] * 3)[:3]
    out.answer = (out.answer or "").strip()

    return out    