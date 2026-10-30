import json
from pathlib import Path

file_path = Path(__file__).with_name("evaluation_policy_rerank_k16_n8.json")
results = json.loads(file_path.read_text(encoding="utf-8"))
print(results)

total_cases = len(results)
answerable_cases = 0
expected_fact_match_cases = 0
invalid_citation_cases = 0
for result in results:
    if result["expected_answerable"] == True:
        answerable_cases += 1
        if result["fact_found"] == True:
            expected_fact_match_cases += 1
    if result["invalid_ids"] != []:
        invalid_citation_cases += 1

print("Tổng ca:",total_cases)
print("Ca có thông tin:",answerable_cases)
print("Ca khớp cụm mong đợi:",expected_fact_match_cases)
print("Ca thiếu thông tin:", total_cases - answerable_cases)
print("Ca có ID trích dẫn ngoài context:",invalid_citation_cases)