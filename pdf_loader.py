from pypdf import PdfReader

def load_pdf(file_path):
    reader = PdfReader(file_path)
    return extract_pdf_pages(reader, file_path.name)

def extract_pdf_pages(reader, source_name):
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({
                "source": source_name,
                "page_number": page_number,
                "text": text
            })
    return pages