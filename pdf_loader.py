from pypdf import PdfReader

def load_pdf(file_path):
    reader = PdfReader(file_path)
    ls = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text().strip()
        if text != "":
            ls.append({
                "source": file_path.name,
                "page_number": page_number,
                "text": text
            })

    return ls
