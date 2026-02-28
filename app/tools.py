# app/tools.py
from .schemas import SummaryResponse, ActionItemsResponse
from .llm_client import parse_with_retry
from .config import MODEL_NAME
import re



def _split_sentences(text: str) -> list[str]:
    # 한국어/영어 문장 분리 (대략적이지만 실무에서 충분히 사용됨)
    parts = re.split(r'(?<=[.!?…])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def _clamp_to_3_sentences(text: str) -> str:
    sentences = _split_sentences(text)
    return " ".join(sentences[:3]).strip()


def summarize(text: str) -> str:
    prompt = f"""
너는 한국어 요약 전문가야.
아래 글을 3~5문장으로 핵심만 간결하게 요약해줘.
수치/기간/대상 같은 정보는 유지해.

[본문]
{text}
"""
    parsed = parse_with_retry(
        model=MODEL_NAME,
        input=prompt,
        text_format=SummaryResponse,
        max_output_tokens=250,
        max_attempts=2,
    )
    
    summary = parsed.summary.strip()

    sentences = _split_sentences(summary)

    # 1️⃣ 빈 요약 → 재요청 (품질 문제)
    if not sentences:
        parsed_retry = parse_with_retry(
            model=MODEL_NAME,
            input=prompt + "\nMake sure the summary is not empty.",
            text_format=SummaryResponse,
            max_output_tokens=250,
            max_attempts=1,
        )
        summary = parsed_retry.summary.strip()
        sentences = _split_sentences(summary)

    # 2️⃣ 3문장 초과 → clamp (재요청 안 함)
    if len(sentences) > 3:
        summary = _clamp_to_3_sentences(summary)

    return summary


def extract_action_items(text: str) -> list[str]:
    prompt = f"""
너는 액션 아이템 추출기야.
아래 글에서 "누가/무엇을/언제" 형태의 실행 항목이 있으면 리스트로 정리해줘.
명확한 실행 항목이 없으면 빈 리스트로 반환해.

[본문]
{text}
"""
    parsed = parse_with_retry(
        model=MODEL_NAME,
        input=prompt,
        text_format=ActionItemsResponse,
        max_output_tokens=250,
        max_attempts=2,
    )
    return [i.strip() for i in parsed.action_items if i.strip()]