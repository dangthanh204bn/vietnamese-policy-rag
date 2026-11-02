def build_context(results):
    blocks = []
    for score, record in results:
        page_number = record.get("page_number")
        if page_number != None:
            line =f"[chunk_{record['chunk_id']}] Nguồn: {record['source']} | Trang: {page_number}\nNgữ cảnh mở rộng:\n{record['parent_text']}"
        else:
            line =f"[chunk_{record['chunk_id']}] Nguồn: {record['source']}\nNgữ cảnh mở rộng:\n{record['parent_text']}"
        blocks.append(line)

    return "\n\n".join(blocks)