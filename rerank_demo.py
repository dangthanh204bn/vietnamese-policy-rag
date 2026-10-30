from sentence_transformers import CrossEncoder
from semantic_retrieval import retrieve
from rag_pipeline import deduplicate_parents
from reranking import rerank_results
from fastembed import TextEmbedding
from tokenizers import Tokenizer
from pathlib import Path
from knowledge_base import prepare_knowledge


reranker = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", device="cpu")

model = TextEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
tokenizer = Tokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
tokenizer.no_truncation()
tokenizer.no_padding()
file_path =Path(__file__).with_name("store_policy_demo.txt")
knowledge = prepare_knowledge(file_path, model, tokenizer, chunk_size=64, overlap=16)
chunk_records = knowledge["chunk_records"]
chunks_vectors = knowledge["chunks_vectors"]


query = "Điều kiện để đổi sản phẩm trong 14 ngày là gì?"

all_candidates = retrieve(
    query, model, chunk_records, chunks_vectors,
    k=len(chunk_records), min_score=-1.0
)

for rank, (score, record) in enumerate(all_candidates, start=1):
    if record["parent_text"].startswith("5."):
        print(rank, score, record["chunk_id"])


candidates = retrieve(query, model, chunk_records, chunks_vectors, k=16, min_score=0.3)
candidates = deduplicate_parents(candidates)
print("\nTrước reranking:")
for score, record in candidates:
    print(score, record["parent_text"].splitlines()[0])




reranked = rerank_results(query, candidates, reranker, top_n=3)
print("\nSau reranking:")
for score, record in reranked:
    print(score, record["parent_text"].splitlines()[0])

