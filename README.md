# PDF Chat RAG – FastAPI + React (no build)

A full-stack web app that lets users upload a PDF and ask conversational questions. It uses a Retrieval-Augmented Generation (RAG) pipeline with FAISS vector search and OpenAI models.

## Features
- Upload large PDFs (tested with 500+ pages).
- Background-friendly chunking with token-aware splitting.
- FAISS similarity search with OpenAI embeddings (`text-embedding-3-large`).
- GPT-based answers grounded in retrieved context.
- Chat-like UI (React via CDN; no build step).
- Cited context snippets with page numbers.
- Session-based stores (multiple documents supported).
- Optional: Summarize document & extract key clauses (for contracts).

## Run locally

### 1) Backend
```bash
cd backend
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # then put your keys
uvicorn main:app --reload --port 8000
```

### 2) Frontend
Just open `frontend/index.html` in your browser, or use a static server:
```bash
cd ../frontend
python -m http.server 5173
# then open http://localhost:5173
```
(If serving from a different origin, the backend already has permissive CORS for dev.)

## Environment
Create a `.env` inside `backend`:
```
OPENAI_API_KEY=sk-...           # required
EMBEDDING_MODEL=text-embedding-3-large
GENERATION_MODEL=gpt-4o         # or gpt-5 if you have access
MAX_OUTPUT_TOKENS=800
EMBEDDING_DIM=3072              # 3072 for text-embedding-3-large; 1536 for -3-small
```

## Notes
- All data is stored under `backend/storage/<session_id>` (index + chunks metadata).
- For large PDFs, embeddings are batched to stay within rate limits.
- This demo uses FAISS on-disk indexes for simplicity; can be swapped for Pinecone, Qdrant, etc.

## Bonus features included
- **Smart citations:** each answer includes the top context chunks with page numbers.
- **Quick doc summary:** auto-summarize after upload.
- **Clause mining (contracts):** detects key clauses (termination, indemnity, payment, liability) for fast navigation.
- **System prompts for safety & grounding:** the model must not hallucinate and says "Not found in document" when appropriate.
