def build_context(results):
    blocks = []
    for score, record in results:
        line =f"[chunk_{record['chunk_id']}] Nguồn: {record['source']}\nNgữ cảnh mở rộng:\n{record['parent_text']}"
        blocks.append(line)
    return "\n\n".join(blocks)