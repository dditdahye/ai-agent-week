import os

# LLM
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")

# Embedding
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

# RAG
TOP_K = int(os.getenv("TOP_K", "3"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# Safety / limits
MAX_TASK_CHARS = int(os.getenv("MAX_TASK_CHARS", "20000"))

RAG_DIST_THRESHOLD = float(os.getenv("RAG_DIST_THRESHOLD", "0.9"))