from pathlib import Path

def load_documents(file_path):
    text = file_path.read_text(encoding = "utf-8")
    clean_text = text.strip()
    parts = clean_text.split("\n\n")
    documents = []

    for i in range(len(parts)):
        clean_part = parts[i].strip()
        if clean_part != "":
            documents.append(clean_part)

    return documents

if __name__ == "__main__":
    file_path = Path(__file__).with_name("knowledge.txt")
    documents = load_documents(file_path)
    print(len(documents))