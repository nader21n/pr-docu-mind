"""Test the PDF extraction function."""

from src.pdf_extractor import extract_text_from_pdf


def main():
    pdf_path = "data/sample.pdf"
    pages = extract_text_from_pdf(pdf_path)

    print(f"Number of pages: {len(pages)}")
    print()

    for page in pages[:3]:
        print(f"--- Page {page['page_number']} (first 300 chars) ---")
        print(page["text"][:300])
        print()


if __name__ == "__main__":
    main()
