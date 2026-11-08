import os, uuid, time, collections
from datetime import datetime, timedelta

import fitz  # PyMuPDF
import chromadb
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, String, Integer, Text, DateTime, func
from sqlalchemy.orm import declarative_base, sessionmaker

# ---------- Config ----------
DATA_DIR = os.path.expanduser("~/rag_data")
os.makedirs(DATA_DIR, exist_ok=True)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768
EMBED_PROVIDER = os.environ.get("EMBED_PROVIDER", "local")  # "local" ya "gemini"
MAX_MB, MAX_CHUNKS, DOC_TTL_HOURS = 10, 300, 24
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")

# ---------- PostgreSQL ----------
db_url = os.environ.get("DATABASE_URL", f"sqlite:///{DATA_DIR}/local.db")  # local test ke liye sqlite fallback
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql+psycopg2://", 1)
elif db_url.startswith("postgresql://"):
    db_url = db_url.replace("postgresql://", "postgresql+psycopg2://", 1)
engine = create_engine(db_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class Document(Base):
    __tablename__ = "documents"
    id = Column(String, primary_key=True)
    filename = Column(String)
    pages = Column(Integer)
    created_at = Column(DateTime, server_default=func.now())


class Chunk(Base):  # source of truth: Chroma data ud jaye toh yahin se rebuild hota hai
    __tablename__ = "chunks"
    id = Column(Integer, primary_key=True, autoincrement=True)
    doc_id = Column(String, index=True)
    idx = Column(Integer)
    page = Column(Integer)
    text = Column(Text)


class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, autoincrement=True)
    doc_id = Column(String, index=True)
    role = Column(String)
    content = Column(Text)
    created_at = Column(DateTime, server_default=func.now())


Base.metadata.create_all(engine)

# ---------- ChromaDB ----------
if os.environ.get("CHROMA_API_KEY"):
    chroma = chromadb.CloudClient(
        tenant=os.environ["CHROMA_TENANT"],
        database=os.environ["CHROMA_DATABASE"],
        api_key=os.environ["CHROMA_API_KEY"],
    )
else:
    chroma = chromadb.PersistentClient(path=f"{DATA_DIR}/chroma")
    collection = chroma.get_or_create_collection(
    f"pdf_chunks_{EMBED_PROVIDER}", metadata={"hnsw:space": "cosine"}, embedding_function=None
)
#     chroma = chromadb.PersistentClient(path="./chroma_data")
# collection = chroma.get_or_create_collection(
#     "pdf_chunks", metadata={"hnsw:space": "cosine"}, embedding_function=None
# )

# ---------- App ----------
app = FastAPI(title="PDF RAG")
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["*"], allow_headers=["*"])
hits_log = collections.defaultdict(collections.deque)


def rate_limit(request: Request, name: str, limit: int, per: int = 60):
    ip = (request.headers.get("x-forwarded-for") or request.client.host).split(",")[0].strip()
    dq, now = hits_log[(name, ip)], time.time()
    while dq and now - dq[0] > per:
        dq.popleft()
    if len(dq) >= limit:
        raise HTTPException(429, "Bahut zyada requests, thodi der baad try karo")
    dq.append(now)


_local_ef = None

def embed(texts, task):
    if not GEMINI_API_KEY:
        raise HTTPException(500, "GEMINI_API_KEY set nahi hai")
    out = []
    for i in range(0, len(texts), 20):
        batch = texts[i:i + 20]
        body = {"requests": [{
            "model": f"models/{EMBED_MODEL}",
            "content": {"parts": [{"text": t}]},
            "taskType": task,
            "outputDimensionality": EMBED_DIM,
        } for t in batch]}
        for attempt in range(4):
            r = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{EMBED_MODEL}:batchEmbedContents",
                headers={"x-goog-api-key": GEMINI_API_KEY}, json=body, timeout=60)
            if r.status_code == 429 and attempt < 3:
                time.sleep(20); continue
            break
        if r.status_code != 200:
            raise HTTPException(502, f"Embedding error: {r.text[:200]}")
        out += [e["values"] for e in r.json()["embeddings"]]
        time.sleep(1.5)
    return out

# def embed(texts, task):
#     global _local_ef
#     if EMBED_PROVIDER == "local":
#         if _local_ef is None:
#             from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
#             _local_ef = DefaultEmbeddingFunction()
#         return [[float(x) for x in v] for v in _local_ef(texts)]
#     # --- yahan se neeche tera purana Gemini wala code waisa hi rahega ---
#     if not GEMINI_API_KEY:
   
#         raise HTTPException(500, "GEMINI_API_KEY set nahi hai")
#     out = []
#     for i in range(0, len(texts), 100):
#         batch = texts[i:i + 100]
#         body = {"requests": [{
#             "model": f"models/{EMBED_MODEL}",
#             "content": {"parts": [{"text": t}]},
#             "taskType": task,
#             "outputDimensionality": EMBED_DIM,
#         } for t in batch]}
#         for attempt in range(3):
            
#             r = requests.post(
#                 f"https://generativelanguage.googleapis.com/v1beta/models/{EMBED_MODEL}:batchEmbedContents",
#                 headers={"x-goog-api-key": GEMINI_API_KEY}, json=body, timeout=60)
#             if r.status_code == 429 and attempt < 2:
#                 time.sleep(5 * (attempt + 1)); continue
#             break
#         if r.status_code != 200:
#             raise HTTPException(502, f"Embedding error: {r.text[:200]}")
#         out += [e["values"] for e in r.json()["embeddings"]]
#     return out


def chunk_pages(doc, size=200, overlap=40):
    chunks = []
    for pno, page in enumerate(doc, 1):
        words = page.get_text().split()
        for i in range(0, len(words), size - overlap):
            piece = " ".join(words[i:i + size])
            if len(piece) > 30:
                chunks.append((pno, piece))
    return chunks


def index_chunks(doc_id, rows):  # rows: list of (idx, page, text)
    vecs = embed([r[2] for r in rows], "RETRIEVAL_DOCUMENT")
    for i in range(0, len(rows), 200):
        part = slice(i, i + 200)
        collection.add(
            ids=[f"{doc_id}_{r[0]}" for r in rows[part]],
            embeddings=vecs[part],
            documents=[r[2] for r in rows[part]],
            metadatas=[{"doc_id": doc_id, "page": r[1]} for r in rows[part]],
        )


def cleanup_old(db):
    cutoff = datetime.utcnow() - timedelta(hours=DOC_TTL_HOURS)
    for d in db.query(Document).filter(Document.created_at < cutoff).all():
        try:
            collection.delete(where={"doc_id": d.id})
        except Exception:
            pass
        db.query(Chunk).filter(Chunk.doc_id == d.id).delete()
        db.query(Message).filter(Message.doc_id == d.id).delete()
        db.delete(d)
    db.commit()


@app.get("/")
def health():
    return {"status": "ok"}

@app.get("/debug")
def debug():
    import resource
    return {
        "embed_provider": EMBED_PROVIDER,
        "has_gemini_key": bool(GEMINI_API_KEY),
        "has_groq_key": bool(GROQ_API_KEY),
        "chroma_cloud": bool(os.environ.get("CHROMA_API_KEY")),
        "memory_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024,
    }


@app.post("/upload")
def upload(request: Request, file: UploadFile = File(...)):
    rate_limit(request, "upload", 5)
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Sirf PDF allowed hai")
    data = file.file.read()
    if len(data) > MAX_MB * 1024 * 1024:
        raise HTTPException(400, f"File {MAX_MB}MB se choti honi chahiye")
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception:
        raise HTTPException(400, "PDF read nahi ho paayi")
    chunks = chunk_pages(doc)
    if not chunks:
        raise HTTPException(400, "PDF mein text nahi mila (scanned PDF ho sakti hai)")
    if len(chunks) > MAX_CHUNKS:
        raise HTTPException(400, "PDF bahut badi hai, chhoti PDF try karo")
    doc_id = uuid.uuid4().hex
    rows = [(i, p, t) for i, (p, t) in enumerate(chunks)]
    with SessionLocal() as db:
        cleanup_old(db)
        index_chunks(doc_id, rows)
        db.add(Document(id=doc_id, filename=file.filename, pages=len(doc)))
        db.add_all([Chunk(doc_id=doc_id, idx=i, page=p, text=t) for i, p, t in rows])
        db.commit()
    return {"session_id": doc_id, "pages": len(doc), "chunks": len(rows)}


class Ask(BaseModel):
    session_id: str
    question: str


@app.post("/ask")
def ask(request: Request, body: Ask):
    rate_limit(request, "ask", 20)
    if not GROQ_API_KEY:
        raise HTTPException(500, "GROQ_API_KEY set nahi hai")
    with SessionLocal() as db:
        if not db.get(Document, body.session_id):
            raise HTTPException(404, "Document nahi mila ya expire ho gaya, PDF dobara upload karo")
        # Chroma data lost ho gaya ho toh Postgres se rebuild
        if not collection.get(where={"doc_id": body.session_id}, limit=1)["ids"]:
            saved = db.query(Chunk).filter(Chunk.doc_id == body.session_id).order_by(Chunk.idx).all()
            index_chunks(body.session_id, [(c.idx, c.page, c.text) for c in saved])
        qv = embed([body.question], "RETRIEVAL_QUERY")[0]
        res = collection.query(query_embeddings=[qv], n_results=5, where={"doc_id": body.session_id})
        docs, metas = res["documents"][0], res["metadatas"][0]
        context = "\n\n".join(f"[Page {m['page']}] {t}" for t, m in zip(docs, metas))
        past = (db.query(Message).filter(Message.doc_id == body.session_id)
                .order_by(Message.id.desc()).limit(6).all())[::-1]
        messages = [{"role": "system", "content": "Tum ek PDF assistant ho. Sirf diye gaye context se jawab do. Agar answer context mein nahi hai toh bolo 'PDF mein yeh info nahi mili'. User jis language mein pooche usi mein jawab do."}]
        messages += [{"role": m.role, "content": m.content} for m in past]
        messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {body.question}"})
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={"model": LLM_MODEL, "temperature": 0.2, "messages": messages}, timeout=60)
        if r.status_code != 200:
            raise HTTPException(502, f"LLM error: {r.text[:200]}")
        answer = r.json()["choices"][0]["message"]["content"]
        db.add_all([Message(doc_id=body.session_id, role="user", content=body.question),
                    Message(doc_id=body.session_id, role="assistant", content=answer)])
        db.commit()
    return {"answer": answer, "pages": sorted({m["page"] for m in metas})}


@app.get("/history/{doc_id}")
def history(doc_id: str):
    with SessionLocal() as db:
        msgs = db.query(Message).filter(Message.doc_id == doc_id).order_by(Message.id).all()
        return [{"role": m.role, "content": m.content} for m in msgs]
