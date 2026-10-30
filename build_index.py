from fastembed import TextEmbedding
from tokenizers import Tokenizer
from pathlib import Path
from knowledge_base import prepare_knowledge
from index_store import save_index

def main():
    model = TextEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    tokenizer = Tokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    tokenizer.no_truncation()
    tokenizer.no_padding()

    file_path = Path(__file__).with_name("store_policy_demo.txt")
    knowledge = prepare_knowledge(file_path, model, tokenizer, chunk_size=64, overlap=16)

    index_dir = Path(__file__).parent / "index"
    save_index(knowledge, index_dir)
    print("Số chunk:", len(knowledge["chunk_records"]))
    print("Số vector:",len(knowledge["chunks_vectors"]))
    print("Đã lưu vào:", index_dir)


if __name__ == "__main__":
    main()
