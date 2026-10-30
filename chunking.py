def chunk_text(text, chunk_size, overlap):
    if chunk_size <= 0:
        raise ValueError("Lỗi kích thước chunk")
    elif 0 > overlap or overlap >= chunk_size:
        raise ValueError("Lỗi overlap")
    chunks = []
    words = text.split()
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunks.append(" ".join(chunk_words))
        if i + chunk_size >= len(words):
           break
    return chunks

def chunk_text_by_tokens(text, tokenizer, chunke_size, overlap):
    if chunke_size <= 0:
        raise ValueError("Lỗi kích thước chunk")
    elif 0 > overlap or overlap >= chunke_size:
        raise ValueError("Lỗi overlap")

    encoded = tokenizer.encode(text, add_special_tokens = False)
    offsets = encoded.offsets
    chunks = []
    for i in range(0, len(encoded), chunke_size - overlap):
        start = offsets[i][0]
        last_token = min(i + chunke_size, len(offsets)) -1
        end = offsets[last_token][1]
        chunks.append(text[start:end].strip())
        if i + chunke_size >= len(offsets):
            break
    return chunks



if __name__ == "__main__":
    print(chunk_text("A B C D E F", 4, 2))