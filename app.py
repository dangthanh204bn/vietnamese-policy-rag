from fastembed import TextEmbedding
from pathlib import Path
from rag_pipeline import answer_question
import requests
from sentence_transformers import CrossEncoder
from config import RETRIEVAL_K
from config import RERANK_TOP_N
from config import RETRIEVAL_MIN_SCORE
from index_store import load_index
from config import INDEX_DIR

def main():
    model = TextEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    index_dir = Path(__file__).parent / INDEX_DIR
    knowledge = load_index(index_dir)
    chunk_records = knowledge["chunk_records"]
    chunks_vectors = knowledge["chunks_vectors"]
    k= RETRIEVAL_K
    reranker = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", device="cpu")


    while True:
        query = input("\nCâu hỏi (gõ exit để thoát):").strip()

        if query.lower() == 'exit':
            break

        if query == "":
            continue
        try:
            rag_result = answer_question(query, model, chunk_records, chunks_vectors, k=k, min_score=RETRIEVAL_MIN_SCORE, reranker=reranker, top_n=RERANK_TOP_N)
            print(rag_result["answer"])
            for score, record in rag_result["retrieved_results"]:
                print(f"Nguồn: {record['source']} | Mở rộng từ [chunk_{record['chunk_id']}]")
                print(record["parent_text"])

        except requests.exceptions.Timeout:
            print("Chờ phản hồi từ Ollama quá lâu")

        except requests.exceptions.ConnectionError:
            print("Không kết nối được với Ollama. Hãy mở Ollama rồi thử lại")


if __name__ == "__main__":
    main()
