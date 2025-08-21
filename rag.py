\
import os
import json
from typing import List, Dict, Any, Tuple, Union
import numpy as np
import faiss
from groq import Groq

STORAGE_DIR = os.path.join(os.path.dirname(__file__), "storage")
os.makedirs(STORAGE_DIR, exist_ok=True)

def split_into_chunks(pages: List[Dict[str, Any]], enc, target_tokens=600, overlap_tokens=120):
    """Token-aware splitter that keeps page labels for citation."""
    chunks = []
    meta = []
    buf = []
    buf_tokens = 0

    def toks(s): 
        return len(enc.encode(s)) if s else 0

    current_page_for_meta = 1
    for p in pages:
        text = p["text"]
        if not text.strip():
            current_page_for_meta = p["page"]
            continue
        paras = [para.strip() for para in text.split("\n") if para.strip()]
        for para in paras:
            pt = toks(para)
            if buf_tokens + pt > target_tokens and buf:
                chunk_text = "\n".join(buf)
                chunks.append({"text": chunk_text})
                meta.append({"page": current_page_for_meta})
                # overlap
                overlap_ids = enc.encode(chunk_text)[-overlap_tokens:]
                overlap_text = enc.decode(overlap_ids) if overlap_ids else ""
                buf = ([overlap_text] if overlap_text else []) + [para]
                buf_tokens = toks("\n".join(buf))
            else:
                buf.append(para)
                buf_tokens += pt
            current_page_for_meta = p["page"]

    if buf:
        chunk_text = "\n".join(buf)
        chunks.append({"text": chunk_text})
        meta.append({"page": current_page_for_meta})

    return chunks, meta

def embed_texts_batched(texts: List[str], emb_model, batch_size: int = 64):
    """Get embeddings for texts in batches using the local SentenceTransformer model."""
    if emb_model is None:
        raise ValueError("Embedding model not initialized")
    
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        # Use the local model to get embeddings
        embeddings = emb_model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        all_embeddings.extend(embeddings)
    
    return np.array(all_embeddings, dtype=np.float32)

def rank_by_similarity(q_vectors: np.ndarray, vectors: np.ndarray, top_k: int = 6):
    if vectors.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    faiss.normalize_L2(q_vectors)
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    D, I = index.search(q_vectors, top_k)
    return I[0], D[0]

class RAGStore:
    """Stores chunks, metadata, and FAISS vectors on disk per session_id."""
    def __init__(self, session_id: str, dim: int):
        self.session_id = session_id
        self.dim = dim
        self.base = os.path.join(STORAGE_DIR, session_id)
        self.chunks_path = os.path.join(self.base, "chunks.jsonl")
        self.meta_path = os.path.join(self.base, "meta.jsonl")
        self.vecs_path = os.path.join(self.base, "vectors.npy")
        os.makedirs(self.base, exist_ok=True)

    def save(self, chunks: List[Dict[str, Any]], meta: List[Dict[str, Any]], vectors: np.ndarray):
        with open(self.chunks_path, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        with open(self.meta_path, "w", encoding="utf-8") as f:
            for m in meta:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        np.save(self.vecs_path, vectors)

    def exists(self) -> bool:
        return os.path.exists(self.chunks_path) and os.path.exists(self.meta_path) and os.path.exists(self.vecs_path)

    def load(self):
        chunks = [json.loads(l) for l in open(self.chunks_path, "r", encoding="utf-8").read().splitlines()]
        meta = [json.loads(l) for l in open(self.meta_path, "r", encoding="utf-8").read().splitlines()]
        vectors = np.load(self.vecs_path)
        return chunks, meta, vectors

def summarize_document(client: Groq, model: str, pages: List[Dict[str, Any]]) -> str:
    head_text = "\n".join((p["text"] or "") for p in pages[:30])[:3000]
    prompt = f"Summarize the following document in 5 bullet points focusing on key scope, stakeholders, dates, obligations:\n\n{head_text}"
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.2,
    )
    return response.choices[0].message.content or ""

CLAUSE_LIST = [
    "Term & Termination",
    "Payment Terms",
    "Confidentiality",
    "IP Ownership",
    "Indemnification",
    "Limitation of Liability",
    "Governing Law",
    "Dispute Resolution",
    "Warranties"
]

def mine_contract_clauses(client: Groq, model: str, pages: List[Dict[str, Any]]):
    head = "\n".join((p['text'] or '') for p in pages[:50])[:6000]
    prompt = (
        "From this (possibly partial) contract, find the presence and summary of clauses: "
        + ", ".join(CLAUSE_LIST)
        + ". Return a JSON object with keys as clause names and values as short summaries (or 'not found'). Text:\n"
        + head
    )
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a contract analysis assistant that identifies and summarizes key contract clauses."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=800,
        temperature=0.2,
        response_format={"type": "json_object"}
    )
    
    try:
        result = response.choices[0].message.content
        return json.loads(result)
    except Exception as e:
        print(f"Error parsing contract clauses: {e}")
        return {clause: "not found" for clause in CLAUSE_LIST}

def make_grounded_answer(client: Groq, model: str, question: str, contexts: List[Dict[str, Any]], max_output_tokens=800):
    context_text = "\n\n---\n\n".join(c["text"][:2000] for c in contexts)
    system = (
        "You are a careful assistant that answers only using the provided document context. "
        "If the answer is not present, say 'Not found in the document.' Keep answers concise, cite page numbers when evident."
    )
    user = f"Question: {question}\n\nDocument context (may be partial, from the user's PDF):\n{context_text}"
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_output_tokens,
        temperature=0.2,
    )
    
    return response.choices[0].message.content, contexts
