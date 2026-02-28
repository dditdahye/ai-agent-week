from pypdf import PdfReader

def load_pdf_by_page(file_path: str) -> list[tuple[int, str]]:
    reader = PdfReader(file_path)
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):  # 1부터 page 번호
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i, text))
    return pages