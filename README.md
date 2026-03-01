## 🔗 Live Demo

- Swagger: https://ai-agent-week.onrender.com/docs
- GitHub: https://github.com/dditdahye/ai-agent-week



1. AI Agent + RAG 기반 문서 질의응답 시스템
본 프로젝트는 단순 LLM 호출이 아닌,
검색(Search)–근거(Citation)–응답(Generation)을 분리한 RAG(Retrieval-Augmented Generation) 기반 AI Agent입니다.

PDF 문서를 업로드하면, 해당 문서에 대한 질문에 대해
근거(citation)와 함께 구조화된 답변을 제공합니다.





2. 프로젝트 목적
	•	LLM 환각(Hallucination) 최소화
	•	근거 기반 응답 설계
	•	실무 확장 가능한 AI Agent 아키텍처 구현
	•	Docker + CI/CD 기반 배포 자동화




3. 시스템 아키텍처
Client
  ↓
FastAPI (API Layer)
  ↓
Search Layer (Vector DB - Chroma)
  ↓
LLM Layer (Structured Output)
  ↓
Response (Answer + Summary + Citations)




4. 핵심 설계 포인트

1) RAG 구조 설계
	•	PDF → Chunk 분할 (Sliding Window + Overlap)
	•	Embedding 생성
	•	Vector Store 저장 (ChromaDB)
	•	질문 → 벡터 검색 → 관련 Chunk 추출
	•	LLM 응답 생성 (근거 기반)

단순 LLM 응답이 아닌,
문서 기반 검색 결과를 컨텍스트로 주입하는 RAG 구조입니다.

2) Hallucination 통제 전략

LLM의 환각을 최소화하기 위해 다음 전략을 적용했습니다:
	•	Distance Threshold 기반 근거 필터링
	•	키워드 Fallback 전략
	•	근거 부족 시 응답 거절 처리
	•	Structured Output(Pydantic Schema) 강제

근거가 부족하면 답변하지 않도록 설계했습니다.

3) Structured Output 설계

응답은 JSON 형태로 강제합니다.
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

•	summary_3lines → 3줄 요약 강제
•	citations → 문서 페이지 + 근거 excerpt 제공

4) API 구성

(1) /rag/upload
	•	PDF 업로드
	•	문서 Chunk 분할
	•	Embedding 생성 및 Vector DB 저장
  
(2) /rag/ask
	•	문서 기반 질의응답
	•	근거 포함 응답 반환
  
(3) /agent/run
	•	일반 LLM Task 실행




5. 기술 스택
	•	Python
	•	FastAPI
	•	OpenAI API
	•	ChromaDB (Vector Store)
	•	Docker
	•	GitHub
	•	Render (CI/CD 자동 배포)




6. CI/CD 및 배포
	•	GitHub 연동
	•	Docker 기반 컨테이너화
	•	Render 자동 빌드 & 자동 배포
	•	코드 Push 시 자동 재배포

실행 환경 일관성을 유지하기 위해 Docker를 사용했습니다.




7. 설계 의도

이 프로젝트는 단순 기능 구현이 아닌,
	•	LLM 동작 이해
	•	Prompt 제어
	•	Structured Output 활용
	•	RAG 기반 근거 설계
	•	환각 통제 전략
	•	배포 자동화 구성

을 목표로 설계되었습니다.

