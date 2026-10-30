from policy_evaluation_cases import test_cases
from pathlib import Path
from load_documents import load_documents
from chunking import chunk_text
from sentence_transformers import CrossEncoder
from rag_pipeline import answer_question
from fastembed import TextEmbedding
from tokenizers import Tokenizer
from knowledge_base import prepare_knowledge
from evaluation import extract_citation_ids
from config import RERANK_TOP_N
from config import RETRIEVAL_MIN_SCORE
from config import RETRIEVAL_K
import json

file_path = Path(__file__).with_name("store_policy_demo.txt")

policy_data = load_documents(file_path)

reranker = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", device="cpu")
model = TextEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
tokenizer = Tokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
tokenizer.no_truncation()
tokenizer.no_padding()

knowledge = prepare_knowledge(file_path, model, tokenizer, chunk_size=64, overlap=16)
chunk_records = knowledge["chunk_records"]
chunks_vectors = knowledge["chunks_vectors"]


evaluation_results = []
for case in test_cases:
    query = case["query"]
    rag_result = answer_question(query, model, chunk_records, chunks_vectors, k=RETRIEVAL_K, min_score=RETRIEVAL_MIN_SCORE, reranker=reranker, top_n=RERANK_TOP_N)
    answer = rag_result["answer"]
    ranked_result = rag_result["retrieved_results"]
    fact_found = None
    if case["answerable"]:
        expected_fact = case["expected_fact"]
        fact_found = expected_fact.lower() in answer.lower()

    cited_ids = extract_citation_ids(answer)
    allowed_ids = set()
    for score, record in ranked_result:
        allowed_ids.add(record['chunk_id'])
    invalid_ids = cited_ids - allowed_ids


    evaluation_results.append(
        {
            "query": query,
            "answer": answer,
            "expected_answerable": case["answerable"],
            "fact_found": fact_found,
            "allowed_ids": sorted(allowed_ids),
            "cited_ids": sorted(cited_ids),
            "invalid_ids": sorted(invalid_ids)
        }
    )

output_path = Path(__file__).with_name(
    f"evaluation_policy_rerank_k{RETRIEVAL_K}_n{len(test_cases)}.json"
)
json_text = json.dumps(evaluation_results, ensure_ascii=False, indent=2)
output_path.write_text(json_text, encoding='utf-8')
    