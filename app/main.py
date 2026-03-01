from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from openai import RateLimitError, APITimeoutError, APIConnectionError, APIStatusError

from .service import run_agent
from app.rag.pdf_loader import load_pdf_by_page
from app.rag.store import add_documents, search, split_text
from app.service import summarize, rag_answer
from app.config import RAG_DIST_THRESHOLD
from app.schemas import RAGRequest, RAGAnswerResponse

import os

app = FastAPI(title="Week1 AI Agent")

class AgentRunRequest(BaseModel):
    task: str

class RAGRequest(BaseModel):
    question: str


@app.post("/rag/ask", response_model=RAGAnswerResponse)
def ask_rag(req: RAGRequest):

    docs, metas, dists = search(req.question, k=3)

    # -------------------------------
    # 키워드 폴백: 벡터 검색이 약할 때(거리 큼) 키워드 포함 chunk를 우선 사용
    # -------------------------------
    best_dist = min(dists) if dists else None

    # 질문에서 아주 간단히 키워드 뽑기(기본형)
    keywords = ["대표이사", "영업이익", "매출", "총자산"]
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
    answer = rag_answer(req.question, "\n\n".join(context_docs))

    citations = []
    for i, (doc, meta, _) in enumerate(filtered, start=1):
        citations.append({
            "id": i,
            "source": meta.get("source", ""),
            "page": meta.get("page", ""),
            "excerpt": doc[:300]
        })

    return {
        "question": req.question,
        "answer": answer,
        "summary_3lines": [
            "요약 1",
            "요약 2",
            "요약 3",
        ],
        "citations": citations,
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