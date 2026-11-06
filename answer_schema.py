from typing import Literal
from pydantic import BaseModel

class LLMAnswer(BaseModel):
    status: Literal["supported", "insufficient_context"]
    answer: str
    cited_chunk_ids: list[int]