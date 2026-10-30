import json
from fastembed import TextEmbedding
from pathlib import Path
from load_documents import load_documents
from tokenizers import Tokenizer
from chunking import chunk_text_by_tokens
from context_builder import build_context
from prompt_builder import build_prompt
from llm_client import generate_answer
from evaluation_cases import test_cases
from evaluation import extract_citation_ids
from semantic_retrieval import retrieve
from rag_pipeline import answer_question
from knowledge_base import prepare_knowledge

model = TextEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
tokenizer = Tokenizer.from_pretrained(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
tokenizer.no_truncation()
tokenizer.no_padding()
file_path = Path(__file__).with_name("knowledge.txt")


knowledge = prepare_knowledge(file_path, model, tokenizer)
chunk_records = knowledge["chunk_records"]
chunks_vectors = knowledge["chunks_vectors"]

print(chunk_records)



evaluation_results = []
for case in test_cases:
    query = case["query"]
    print("\nCâu hỏi", query)
    
    k = 2
    rag_result = answer_question(query, model, chunk_records, chunks_vectors, k=k, min_score= 0.3)
    answer = rag_result["answer"]
    ranked_results = rag_result["retrieved_results"]
    print(answer)


    allowed_ids = set()
    for score, record in ranked_results[:k]:
        allowed_ids.add(record["chunk_id"])

    cited_ids = extract_citation_ids(answer)
    invalid_ids = cited_ids - allowed_ids
    print("ID trích dẫn ngoài context", invalid_ids)



    fact_found = None
    if case["answerable"]:
        expected_fact = case["expected_fact"]
        fact_found = expected_fact.lower() in answer.lower()

    evaluation_results.append({
        "query": query,
        "answer": answer,
        "expected_answerable": case["answerable"],
        "fact_found": fact_found,
        "allowed_ids": sorted(allowed_ids),
        "cited_ids": sorted(cited_ids),
        "invalid_ids": sorted(invalid_ids)
    })

    print("Có cụm đáp án mong đợi:", fact_found)
print("Số trường hợp đã ghi nhận:", len(evaluation_results))

output_path = Path(__file__).with_name("evaluation_result.json")
json_text = json.dumps(evaluation_results, ensure_ascii=False, indent=2)
output_path.write_text(json_text, encoding="utf-8")

print("Đã lưu kết quả:", output_path)


