from pydantic import BaseModel, ValidationError
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4
from index_store import load_startup_index, save_index
from fastembed import TextEmbedding
from sentence_transformers import CrossEncoder
from rag_pipeline import answer_question
from config import RETRIEVAL_K, RERANK_TOP_N, RETRIEVAL_MIN_SCORE
from fastapi import FastAPI, HTTPException, UploadFile
import requests
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from config import INDEX_DIR
from pdf_loader import extract_pdf_pages
from tokenizers import Tokenizer
from knowledge_base import prepare_knowledge_from_documents
import json
from threading import Lock

upload_lock = Lock()
UPLOAD_ROOT = Path(__file__).parent / "output" / "uploaded_indexes"
class AskRequest(BaseModel):
    query: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Khởi động ứng dụng")
    index_dir = Path(__file__).parent / INDEX_DIR
    app.state.knowledge = load_startup_index(index_dir, UPLOAD_ROOT)
    app.state.embedding_model = TextEmbedding( model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    app.state.reranker = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",device="cpu")
    app.state.tokenizer = Tokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    app.state.tokenizer.no_padding()
    app.state.tokenizer.no_truncation()
    yield
    print("Dừng ứng dụng")

app = FastAPI(lifespan=lifespan)

@app.post("/ask")
def ask(payload: AskRequest):
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống")
    knowlege = app.state.knowledge
    try:
        rag_result = answer_question(query=query, model=app.state.embedding_model, chunk_records=knowlege["chunk_records"], chunks_vectors=knowlege["chunks_vectors"],
                             k=RETRIEVAL_K, min_score=RETRIEVAL_MIN_SCORE, reranker=app.state.reranker, top_n=RERANK_TOP_N)
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Chờ phản hồi từ Ollama quá lâu")
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="Không kết nối được với Ollama")
    except ValidationError:
        raise HTTPException(status_code=502,
                            detail="Phản hồi từ mô hình không đúng định dạng.")

    retrieved_sources = []
    for score, record in rag_result["retrieved_results"]:
        retrieved_sources.append({
            "chunk_id": record["chunk_id"],
            "source": record["source"],
            "text": record["parent_text"],
            "page_number": record.get("page_number")
        })
    return {"received_query": query,
            "answer": rag_result["answer"],
            "retrieved_sources": retrieved_sources
            }

@app.get("/health")
def health():
    print("Hàm health vừa được gọi")
    return{"status": "ok",
           "chunk_count": len(app.state.knowledge["chunk_records"])
           }

@app.post("/upload")
def upload_document(file: UploadFile):
    try:
        reader = PdfReader(file.file)
        page_count = len(reader.pages)
        pages = extract_pdf_pages(reader, file.filename)
    except PdfReadError:
        raise HTTPException(status_code=400, detail="Không đọc được file PDF.")
    if not pages:
        raise HTTPException(status_code=400, detail="PDF không có văn bản để xử lý.")
    new_knowledge = prepare_knowledge_from_documents(documents=pages, model=app.state.embedding_model, tokenizer=app.state.tokenizer, chunk_size=64, overlap=16)
    upload_dir = UPLOAD_ROOT / uuid4().hex
    save_index(new_knowledge, upload_dir)
    manifest_tmp = upload_dir.parent / f".active-{upload_dir.name}.tmp"
    manifest_tmp.write_text(json.dumps({"index_id": upload_dir.name}),
                            encoding="utf-8")
    with upload_lock:
        manifest_tmp.replace(upload_dir.parent / "active.json")
        app.state.knowledge = new_knowledge
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "page_count": page_count,
        "text_page_count": len(pages),
        "chunk_count": len(new_knowledge["chunk_records"])
    }