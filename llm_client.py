import requests
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
        "stream": False
    }
    response = requests. post(url, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    return data["message"]["content"]

if __name__ == "__main__":
    answer = generate_answer("Hãy chào tôi bằng một câu Tiếng Việt.")
    print(answer)