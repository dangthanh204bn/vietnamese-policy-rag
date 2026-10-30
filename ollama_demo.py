import requests

url = "http://localhost:11434/api/chat"

payload = {
    "model": "qwen3:4b",
    "messages": [
        {
            "role": "user",
            "content": "Hãy chào tôi bằng một câu tiếng Việt."
        }
    ],
    "stream": False
}

response = requests.post(url, json=payload, timeout=120)
response.raise_for_status()

data = response.json()
print(data["message"]["content"])