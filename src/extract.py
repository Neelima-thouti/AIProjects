import pdfplumber
import os

def extract_text_from_pdf(pdf_path):
    """
    Opens a PDF and pulls out all the text, page by page.
    Returns a list of dicts: [{"page": 1, "text": "..."}, ...]
    We keep page numbers because later we want to cite
    'this answer came from page 4' - important for insurance trust.
    """
    pages_content = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:  # skip blank pages
                pages_content.append({
                    "page": page_num,
                    "text": text,
                    "source_file": os.path.basename(pdf_path)
                })
    
    return pages_content


def extract_all_policies(folder_path="data/policies"):
    """
    Loops through every PDF in the folder and extracts text from all of them.
    Returns one combined list covering all policy documents.
    """
    all_content = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            full_path = os.path.join(folder_path, filename)
            print(f"Extracting: {filename}")
            pages = extract_text_from_pdf(full_path)
            all_content.extend(pages)
    
    print(f"Total pages extracted: {len(all_content)}")
    return all_content


if __name__ == "__main__":
    content = extract_all_policies()
    # print first page as a sanity check
    if content:
        print("\n--- Sample extracted text (first page) ---")
        print(content[0]["source_file"], "- Page", content[0]["page"])
        print(content[0]["text"][:500])  # first 500 characters