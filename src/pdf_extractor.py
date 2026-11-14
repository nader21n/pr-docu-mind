"""PDF text extraction for DocuMind."""

import pymupdf


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """Extract text from a PDF, page by page.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        A list of dictionaries, each with:
            - "page_number": the page number (starting at 1)
            - "text": the extracted text from that page
    """
    doc = pymupdf.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        pages.append(
            {
                "page_number": page_num,
                "text": text,
            }
        )
    doc.close()
    return pages
