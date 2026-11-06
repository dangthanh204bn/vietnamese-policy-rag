from document_loader import load_source
from chunking import chunk_text_by_tokens


def prepare_knowledge(file_path, model, tokenizer, chunk_size = 16, overlap = 2):
    documents = load_source(file_path)
    return prepare_knowledge_from_documents(documents, model, tokenizer, chunk_size, overlap)

def prepare_knowledge_from_documents(
        documents, model, tokenizer, chunk_size=16, overlap=2
):
    all_chunks = []
    chunk_records = []
    for parent_id, document in enumerate(documents, start=1):
        text = document["text"]
        document_chunk = chunk_text_by_tokens(text, tokenizer, chunke_size=chunk_size, overlap=overlap)
        for chunk in document_chunk:
            chunk_records.append({
                "parent_id": parent_id,
                "parent_text": text,
                "chunk_id": len(chunk_records)+1,
                "text": chunk,
                "source": document["source"],
                "page_number": document["page_number"]
            })
            all_chunks.append(chunk)
    for chunk in all_chunks:
        encoded = tokenizer.encode(chunk, add_special_tokens = True)
        if len(encoded.ids) > 128:
            raise ValueError("Chunk vượt quá giới hạn 128 token.")
    chunks_vectors = list(model.embed(all_chunks))

    return{
        "chunk_records": chunk_records,
        "chunks_vectors": chunks_vectors
    }
