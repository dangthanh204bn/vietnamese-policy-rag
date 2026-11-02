from pypdf import PdfReader
from pathlib import Path

file_path = Path(__file__).with_name("sample.pdf")

reader = PdfReader(file_path)

for page_number, page in enumerate(reader.pages, start=1):
    print(page_number)
    print(page.extract_text())