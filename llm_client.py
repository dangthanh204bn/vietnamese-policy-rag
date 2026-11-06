import requests
from answer_schema import LLMAnswer

def generate_answer(prompt):
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": "qwen3:4b",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "format": LLMAnswer.model_json_schema(),
        "stream": False,
        "options": {
            "temperature": 0,
            "seed": 42
        }
    }
    response = requests. post(url, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    answer_result = LLMAnswer.model_validate_json(
        data["message"]["content"]
    )
    print("Trạng thái model:", answer_result.status)
    print("ID model chọn:", answer_result.cited_chunk_ids)
    return answer_result.model_dump()

if __name__ == "__main__":
    answer = generate_answer("Hãy chào tôi bằng một câu Tiếng Việt.")
    print(answer)