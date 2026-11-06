def format_answer(result):
    if result["status"] == "insufficient_context":
        return "Các đoạn được cung cấp chưa đủ thông tin để trả lời câu hỏi này."
    elif result["status"] == "supported":
        text = result["answer"]
        labels = []
        for id in result["cited_chunk_ids"]:
            labels.append(f"[chunk_{id}]")
        citations = " ".join(labels)
        return f"{text} {citations}"
