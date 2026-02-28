# app/llm_client.py
from __future__ import annotations

import random
import time
from typing import Type, TypeVar

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from openai import RateLimitError, APITimeoutError, APIConnectionError, APIStatusError

T = TypeVar("T")

# ✅ 실무: client 단일화 + timeout 고정
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