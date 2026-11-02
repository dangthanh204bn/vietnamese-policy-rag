from fastembed import TextEmbedding
from tokenizers import Tokenizer
from pathlib import Path
from knowledge_base import prepare_knowledge
from index_store import save_index

def main():
    model = TextEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    tokenizer = Tokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    tokenizer.no_padding()
    tokenizer.no_truncation()

    file_path = Path(__file__).with_name("sample.pdf")
    knowledge = prepare_knowledge(file_path, model, tokenizer, chunk_size=64, overlap=16)
    index_dir = Path(__file__).parent / "index_pdf"
    save_index(knowledge, index_dir)

    chunk_records = knowledge["chunk_records"]
    chunks_vectors = knowledge["chunks_vectors"]
    print("Số chunk:",len(chunk_records))
    print("Số vector:",len(chunks_vectors))
    print(chunk_records)
    print(index_dir.name)


if __name__ == "__main__":
    main()