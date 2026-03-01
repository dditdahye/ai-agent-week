from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from openai import RateLimitError, APITimeoutError, APIConnectionError, APIStatusError

from .service import run_agent
from app.rag.pdf_loader import load_pdf_by_page
from app.rag.store import add_documents, search, split_text
from app.service import summarize, rag_answer
from app.config import RAG_DIST_THRESHOLD
from app.schemas import RAGRequest, RAGAnswerResponse
from app.llm_client import rag_answer_with_summary

import os

app = FastAPI(title="Week1 AI Agent")

class AgentRunRequest(BaseModel):
    task: str

class RAGRequest(BaseModel):
    question: str

def extract_relevant_excerpt(doc: str, keywords: list[str], window: int = 120):
    clean_doc = " ".join(doc.split())  # 줄바꿈/공백 정리

    for kw in keywords:
        idx = clean_doc.find(kw)
        if idx != -1:
            start = max(0, idx - window)
            end = min(len(clean_doc), idx + window)
            return clean_doc[start:end]

    return clean_doc[:200]  # 키워드 못 찾으면 앞부분 반환

def infer_requested_fields(question: str) -> list[str]:
    # “항목” 후보 사전(프로젝트에서 자주 쓰는 값들)
    field_keywords = [
        "대표이사", "CEO",
        "총자산", "자산",
        "매출", "매출액", "매출액은",
        "영업이익",
        "당기순이익", "순이익",
        "임직원", "직원", "인원",
        "근속", "평균 근속", "근속연수",
    ]

    q = question.replace("?", " ").replace(",", " ")
    hits = []
    for k in field_keywords:
        if k in q and k not in hits:
            hits.append(k)
    return hits


@app.post("/rag/ask", response_model=RAGAnswerResponse)
def ask_rag(req: RAGRequest):

    docs, metas, dists = search(req.question, k=8)

    # -------------------------------
    # 키워드 폴백: 벡터 검색이 약할 때(거리 큼) 키워드 포함 chunk를 우선 사용
    # -------------------------------
    best_dist = min(dists) if dists else None

    # 질문에서 아주 간단히 키워드 뽑기(기본형)
    keywords = ["대표이사", "영업이익", "매출", "총자산", "자산"]
    for token in req.question.replace("?", " ").replace(",", " ").split():
        # 너무 짧은 토큰은 제외(노이즈 방지)
        if len(token) >= 2:
            keywords.append(token)

    # 벡터 검색이 애매할 때만(예: best_dist가 큰 경우) 폴백
    if best_dist is not None and best_dist > 1.6:
        keyword_hits = []
        for doc, meta, dist in zip(docs, metas, dists):
            if any(k in doc for k in keywords):
                keyword_hits.append((doc, meta, dist))

        # 키워드로라도 잡히면 그 결과를 사용
        if keyword_hits:
            filtered = keyword_hits
        else:
            filtered = list(zip(docs, metas, dists))
    else:
        filtered = list(zip(docs, metas, dists))

    # (선택) 근거가 너무 약하면 거절 (LLM 환각 방지)
    RAG_DIST_THRESHOLD = float(os.getenv("RAG_DIST_THRESHOLD", "2.0"))
    if (best_dist is None) or (best_dist > RAG_DIST_THRESHOLD):
        return {
            "question": req.question,
            "answer": "문서에서 해당 질문과 직접 관련된 근거를 찾지 못했습니다.",
            "citations": []
        }

    # 여기부터는 기존 코드
    context_docs = [doc for doc, _, _ in filtered]
    context = "\n\n".join(context_docs)

    # LLM이 answer + summary_3lines 생성
    llm_out = rag_answer_with_summary(req.question, context)

    filtered_for_cite = [(doc, meta, dist) for (doc, meta, dist) in filtered if any(k in doc for k in keywords)]

    if not filtered_for_cite:
        filtered_for_cite = filtered

    citations = []
    for i, (doc, meta, _) in enumerate(filtered_for_cite[:3], start=1):
        excerpt = extract_relevant_excerpt(doc, keywords)
        citations.append({
            "id": i,
            "source": meta.get("source", ""),
            "page": meta.get("page", ""),
            "excerpt": excerpt
        })

    return {
        "question": req.question,
        "answer": llm_out.answer,
        "summary_3lines": llm_out.summary_3lines,
        "citations": citations
    }



@app.post("/agent/run")
def agent_run(req: AgentRunRequest):
    try:
        return run_agent(req.task)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except RateLimitError:
        raise HTTPException(status_code=429, detail="Rate limit / quota exceeded")

    except (APITimeoutError, APIConnectionError):
        raise HTTPException(status_code=503, detail="Upstream LLM unavailable or timed out")

    except APIStatusError as e:
        status = getattr(e, "status_code", 500)
        if status >= 500:
            raise HTTPException(status_code=503, detail="Upstream LLM server error")
        raise HTTPException(status_code=500, detail="LLM request failed")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/rag/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        file_path = f"./{file.filename}"

        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        pages = load_pdf_by_page(file_path)

        add_documents(pages, source=file.filename)

        return {"message": "Document uploaded and indexed successfully."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))