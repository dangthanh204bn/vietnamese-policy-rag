import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).with_name(".env")
load_dotenv(env_path)

has_key = bool(os.getenv("OPENAI_API_KEY"))
print("Đã đọc được API key:", has_key)