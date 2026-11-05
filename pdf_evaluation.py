import requests
from pdf_evaluation_cases import test_cases
from evaluation import extract_citation_ids
import json
from pathlib import Path


def ask_api(query):
    response = requests.post(url= "http://127.0.0.1:8000/ask", json={"query": query}, timeout=180)
    response.raise_for_status()
    return response.json()
def find_missing_facts(answer, expected_facts_groups):
    normalized_answer = " ".join(answer.lower().split())
    missing_facts = []
    for fact_group in expected_facts_groups:
        group_found = False
        for fact in fact_group:
            normalized_fact = " ".join(fact.lower().split())
            if normalized_fact in normalized_answer:
                group_found = True
                break

        if not group_found:
            missing_facts.append(fact_group)

    return missing_facts

def find_invalid_citation_ids(answer, retrieved_sources):
    cited_ids = extract_citation_ids(answer)
    allowed_ids = set()
    for source in retrieved_sources:
        allowed_ids.add(source["chunk_id"])
    invalid_ids = cited_ids - allowed_ids
    return invalid_ids

def has_expected_cited_source(answer, retrieved_sources, expected_source, expected_page):
    cited_ids = extract_citation_ids(answer)
    for source in retrieved_sources:
        if source["chunk_id"] in cited_ids and source["source"] == expected_source and source.get("page_number") == expected_page:
            return True

    return False

if __name__ == "__main__":
    evaluation_results = []

    for case in test_cases:
        try:
            result = ask_api(case["query"])

            source_match = has_expected_cited_source(result["answer"], result["retrieved_sources"], case["expected_source"], case["expected_page"])
            print("Trích dẫn đúng nguồn và trang:", source_match)
            invalid_ids = find_invalid_citation_ids(result["answer"], result["retrieved_sources"])
            print("ID trích dẫn ngoài context:", invalid_ids)
            missing_facts = find_missing_facts(result["answer"], case["expected_fact_groups"])
            print("Các cụm còn thiếu:", missing_facts)
            print(result["answer"])
            passed = not missing_facts and not invalid_ids and source_match
            evaluation_results.append({
                "query": case["query"],
                "answer": result["answer"],
                "retrieved_sources": result["retrieved_sources"],
                "missing_facts": missing_facts,
                "invalid_ids": sorted(invalid_ids),
                "source_match": source_match,
                "passed": passed,
                "error": None
            })

            print("Câu hỏi:", case["query"])
            print("Đạt kiểm tra:", passed)
        except requests.exceptions.RequestException as error:
            evaluation_results.append({
                "query": case["query"],
                "answer": None,
                "retrieved_sources": [],
                "missing_facts": None,
                "invalid_ids": None,
                "source_match": None,
                "passed": False,
                "error": str(error)
            })
            continue
    output_path = Path(__file__).with_name("pdf_evaluation_report.json")
    json_text = json.dumps(evaluation_results, ensure_ascii=False, indent=2)
    output_path.write_text(json_text, encoding="utf-8")