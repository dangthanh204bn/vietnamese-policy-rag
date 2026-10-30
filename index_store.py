import json
from pathlib import Path
import numpy as np

def save_index(knowledge, index_idr):
    output_path = Path(index_idr)

    output_path.mkdir(parents=True, exist_ok=True)
    json_text = json.dumps(knowledge["chunk_records"], ensure_ascii=False, indent=2)
    metadata_path = output_path / "chunk_records.json"
    metadata_path.write_text(json_text, encoding="utf-8")


    vector = np.asanyarray(knowledge["chunks_vectors"], dtype=np.float32)
    np.save(output_path / "chunk_vectors.npy", vector, allow_pickle=False)

def load_index(index_dir):
    load_path = Path(index_dir)

    json_text = (load_path / "chunk_records.json").read_text(encoding="utf-8")
    chunk_records = json.loads(json_text)
    chunks_vectors = np.load(load_path / "chunk_vectors.npy", allow_pickle=False)
    if len(chunk_records) != len(chunks_vectors):
        raise ValueError("Số chunk và số vector không khớp")


    return{
        "chunk_records": chunk_records,
        "chunks_vectors": chunks_vectors

    }