import json
from pathlib import Path
import numpy as np

def save_index(knowledge, index_dir):
    output_path = Path(index_dir)

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

def load_startup_index(default_dir, upload_root):
    root = Path(upload_root)

    try:
        manifest = json.loads(
            (root / "active.json").read_text(encoding="utf-8")
        )
        index_id = manifest.get("index_id") if isinstance(manifest, dict) else None

        if (
            not isinstance(index_id, str)
            or len(index_id) != 32
            or any(char not in "0123456789abcdef" for char in index_id)
        ):
            raise ValueError("index_id trong active.json không hợp lệ")
        return load_index(root / index_id)

    except (OSError, ValueError, TypeError, EOFError) as error:
        print(f"Dùng chỉ mục mặc định. Lý do: {error}")

    return load_index(default_dir)