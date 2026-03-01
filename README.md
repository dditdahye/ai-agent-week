1. AI Agent + RAG 기반 문서 질의응답 시스템

본 프로젝트는 단순 LLM 호출이 아닌,  
검색(Search)–근거(Citation)–응답(Generation)을 분리한  
RAG(Retrieval-Augmented Generation) 기반 AI Agent입니다.  

PDF 문서를 업로드하면 해당 문서에 대한 질문에 대해  
근거(citation)와 함께 구조화된 답변을 제공합니다.

⸻

2. Live Demo  
	•	Swagger: https://ai-agent-week.onrender.com/docs  
	•	GitHub: https://github.com/dditdahye/ai-agent-week  

⸻

3. 프로젝트 목적  
	1.	LLM 환각(Hallucination) 최소화  
	2.	근거 기반 응답 설계  
	3.	실무 확장 가능한 AI Agent 아키텍처 구현  
	4.	Docker + CI/CD 기반 배포 자동화  

⸻

4. 시스템 아키텍처  

```
Client

  ↓
  
FastAPI (API Layer)

  ↓
  
Search Layer (Vector DB - Chroma)

  ↓
  
LLM Layer (Structured Output)

  ↓
  
Response (Answer + Summary + Citations)
```

5. 핵심 설계 포인트  
  
5.1 RAG 구조 설계  
	1.	PDF → Chunk 분할 (Sliding Window + Overlap)  
	2.	Embedding 생성  
	3.	Vector Store 저장 (ChromaDB)  
	4.	질문 → 벡터 검색 → 관련 Chunk 추출  
	5.	LLM 응답 생성 (근거 기반)  
  
단순 LLM 응답이 아닌, 문서 기반 검색 결과를 컨텍스트로 주입하는 구조입니다.  

⸻

5.2 Hallucination 통제 전략  
  
LLM 환각을 최소화하기 위해 다음 전략을 적용했습니다.  
	1.	Distance Threshold 기반 근거 필터링  
	2.	키워드 Fallback 전략  
	3.	근거 부족 시 응답 거절 처리  
	4.	Structured Output(Pydantic Schema) 강제  
  
근거가 충분하지 않을 경우 답변하지 않도록 설계했습니다.  

⸻

5.3 Structured Output 설계  
  
응답은 JSON 형태로 반환됩니다.  
```json
{

  "question": "...",
  
  "answer": "...",
  
  "summary_3lines": [
  
    "...",
	
    "...",
	
    "..."
	
  ],
  
  "citations": [
  
    {
	
      "id": 1,
	  
      "source": "...",
	  
      "page": 14,
	  
      "excerpt": "..."
	  
    }
	
  ]
  
}
```
  
1.	summary_3lines: 3줄 요약 강제  
2.	citations: 문서 페이지 및 근거 excerpt 포함  

⸻

6. API 구성  
  
6.1 /rag/upload  
	1.	PDF 업로드  
	2.	문서 Chunk 분할  
	3.	Embedding 생성 및 Vector DB 저장  
  
6.2 /rag/ask  
	1.	문서 기반 질의응답    
	2.	근거 포함 응답 반환  
  
6.3 /agent/run  
	1.	일반 LLM Task 실행  

⸻
  
7. 기술 스택  
	1.	Python  
	2.	FastAPI  
	3.	OpenAI API  
	4.	ChromaDB (Vector Store)  
	5.	Docker  
	6.	GitHub  
	7.	Render (CI/CD 자동 배포)  

⸻
  
8. CI/CD 및 배포  
	1.	GitHub 연동  
	2.	Docker 기반 컨테이너화  
	3.	Render 자동 빌드 및 자동 배포  
	4.	코드 Push 시 자동 재배포  
  
실행 환경 일관성을 유지하기 위해 Docker를 사용했습니다.  

⸻
  
9. 설계 의도  
  
본 프로젝트는 단순 기능 구현이 아닌 다음을 목표로 설계되었습니다.  
	1.	LLM 동작 원리 이해  
	2.	Prompt 제어 및 Structured Output 활용  
	3.	RAG 기반 근거 중심 응답 설계  
	4.	환각 통제 전략 구현  
	5.	배포 자동화 구조 구성  

