from semantic_retrieval import retrieve
from prompt_builder import build_prompt
from context_builder import build_context
from llm_client import generate_answer
from reranking import rerank_results
from answer_formatter import format_answer

def answer_question(query, model, chunk_records, chunks_vectors, k=2, min_score=0.3, reranker= None, top_n=3):
    ranked_results = retrieve(query, model, chunk_records, chunks_vectors, k=k, min_score=min_score)
    ranked_results = deduplicate_parents(ranked_results)
    if reranker is not None:
        ranked_results = rerank_results(query, ranked_results, reranker, top_n=top_n)

    if len(ranked_results) == 0:
        answer = "Không có đoạn nào đạt ngưỡng tìm kiếm."
    else:
        context = build_context(ranked_results)
        prompt = build_prompt(query, context)
        result = generate_answer(prompt)
        answer = format_answer(result)

    return {
        "answer": answer,
        "retrieved_results": ranked_results
    }

def deduplicate_parents(results):
    seen_parent_ids = set()
    unique_results = []
    for score, record in results:
        if record["parent_id"] not in seen_parent_ids:
            seen_parent_ids.add(record["parent_id"])
            unique_results.append((score, record))

    return unique_results