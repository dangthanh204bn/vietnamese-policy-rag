from pydantic import BaseModel
from contextlib import asynccontextmanager
from pathlib import Path
from index_store import load_index
from fastembed import TextEmbedding
from sentence_transformers import CrossEncoder
from rag_pipeline import answer_question
from config import RETRIEVAL_K, RERANK_TOP_N, RETRIEVAL_MIN_SCORE
from fastapi import FastAPI, HTTPException
import requests
from config import INDEX_DIR

class AskRequest(BaseModel):
    query: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Khởi động ứng dụng")
    index_dir = Path(__file__).parent / INDEX_DIR
    app.state.knowledge = load_index(index_dir)
    app.state.embedding_model = TextEmbedding( model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    app.state.reranker = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",device="cpu")
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