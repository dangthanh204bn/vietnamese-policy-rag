from pdf_loader import load_pdf
from load_documents import load_documents

def load_source(file_path):
    if file_path.suffix.lower() == ".pdf":
        return load_pdf(file_path)
    elif file_path.suffix.lower() == ".txt":
        documents = load_documents(file_path)
        records = []
        for document in documents:
            records.append(
                {
                "source": file_path.name,
                "page_number": None,
                "text": document
            })
        return records
    else:
        raise ValueError("Không thể load") 