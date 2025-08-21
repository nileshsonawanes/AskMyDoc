\
import os
import io
import uuid
import json
import traceback
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import numpy as np
import tiktoken
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from rag import (
    split_into_chunks, 
    RAGStore, 
    rank_by_similarity, 
    summarize_document, 
    mine_contract_clauses, 
    make_grounded_answer,
    embed_texts_batched
)
from groq import Groq

# Load environment variables
load_dotenv()

# Load model configurations
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "llama-3.3-70b-versatile")
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "800"))
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))

# Initialize the embedding model
EMBEDDING_MODEL = None
try:
    print("Loading embedding model...")
    EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    print("Embedding model loaded successfully")
except Exception as e:
    import traceback
    print("Error loading embedding model:")
    print(traceback.format_exc())
    raise


# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI(title="PDF Chat RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    session_id: str
    message: str
    top_k: int = 6

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)) -> JSONResponse:
    try:
        print(f"Received upload request for file: {file.filename}")
        
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
            
        session_id = str(uuid.uuid4())
        print(f"Processing PDF with session ID: {session_id}")
        
        content = await file.read()
        reader = PdfReader(io.BytesIO(content))

        pages = []
        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
            except Exception as e:
                print(f"Error extracting text from page {i+1}: {str(e)}")
                text = ""
            pages.append({"page": i + 1, "text": text})
        
        print(f"Extracted {len(pages)} pages from PDF")

        enc = tiktoken.get_encoding("cl100k_base")
        chunks, chunk_meta = split_into_chunks(pages, enc, target_tokens=600, overlap_tokens=120)
        print(f"Split into {len(chunks)} chunks")

        try:
            print("Generating embeddings...")
            vectors = embed_texts_batched([c["text"] for c in chunks], EMBEDDING_MODEL)
            print(f"Generated embeddings with shape: {vectors.shape}")
            
            store = RAGStore(session_id=session_id, dim=vectors.shape[1])
            store.save(chunks, chunk_meta, vectors)
            print("Saved chunks and embeddings to store")
            
            # Use Groq for summarization and clause extraction
            print("Generating document summary...")
            doc_summary = summarize_document(client, GENERATION_MODEL, pages)
            print("Extracting contract clauses...")
            clauses = mine_contract_clauses(client, GENERATION_MODEL, pages)
            print("Document processing complete")

            return JSONResponse({
                "session_id": session_id,
                "num_pages": len(pages),
                "num_chunks": len(chunks),
                "summary": doc_summary,
                "clauses": clauses,
            })
            
        except Exception as e:
            print(f"Error during document processing: {str(e)}")
            print(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in upload_pdf: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        print(f"Received chat request for session: {req.session_id}")
        
        store = RAGStore(session_id=req.session_id, dim=EMBEDDING_DIM)
        if not store.exists():
            print(f"Invalid session ID: {req.session_id}")
            return JSONResponse({"error": "Invalid session_id"}, status_code=400)

        print("Loading stored chunks and embeddings...")
        chunks, meta, vectors = store.load()
        print(f"Loaded {len(chunks)} chunks")

        try:
            print("Generating query embedding...")
            q_vec = embed_texts_batched([req.message], EMBEDDING_MODEL)[0:1]
            print("Finding similar chunks...")
            top_idx, scores = rank_by_similarity(q_vec, vectors, top_k=req.top_k)
            context_chunks = [chunks[i] for i in top_idx]
            
            print("Generating answer with Groq...")
            answer, _ = make_grounded_answer(
                client=client,
                model=GENERATION_MODEL,
                question=req.message,
                contexts=context_chunks,
                max_output_tokens=int(MAX_OUTPUT_TOKENS),
            )
            print("Generated answer successfully")

            cites = []
            for rank, idx in enumerate(top_idx):
                cites.append({
                    "rank": rank + 1,
                    "page": meta[idx]["page"],
                    "score": float(scores[rank]),
                    "excerpt": chunks[idx]["text"][:300]
                })

            return JSONResponse({
                "answer": answer,
                "citations": cites,
            })
            
        except Exception as e:
            print(f"Error during chat processing: {str(e)}")
            print(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in chat endpoint: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
