from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from openai import RateLimitError, APITimeoutError, APIConnectionError, APIStatusError

from .service import run_agent
from app.rag.pdf_loader import load_pdf_by_page
from app.rag.store import add_documents, search, split_text
from app.service import summarize, rag_answer
from app.config import RAG_DIST_THRESHOLD

from pydantic import BaseModel


app = FastAPI(title="Week1 AI Agent")

class AgentRunRequest(BaseModel):
    task: str

class RAGRequest(BaseModel):
    question: str


from app.service import rag_answer

@app.post("/rag/ask")
def ask_rag(req: RAGRequest):

    docs, metas, dists = search(req.question, k=3)

    filtered = [
        (doc, meta, dist)
        for doc, meta, dist in zip(docs, metas, dists)
        if dist <= RAG_DIST_THRESHOLD
    ]

    if not filtered:
        return {
            "question": req.question,
            "answer": "문서에서 해당 질문과 직접 관련된 근거를 찾지 못했습니다.",
            "citations": []
        }

    # 필터된 것만 사용
    docs = [x[0] for x in filtered]
    metas = [x[1] for x in filtered]

    context = "\n\n".join(docs)
    answer = rag_answer(req.question, context)

    citations = []
    for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
        citations.append({
            "id": i,
            "source": meta["source"],
            "page": meta["page"],
            "excerpt": doc.replace("\n", " ").strip()[:200]
        })

    return {
        "question": req.question,
        "answer": answer,
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