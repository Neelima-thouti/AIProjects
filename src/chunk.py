from langchain_text_splitters import RecursiveCharacterTextSplitter
from extract import extract_all_policies

def chunk_documents(pages_content, chunk_size=800, chunk_overlap=100):
    """
    Takes the page-level text and splits it into smaller overlapping chunks.
    
    chunk_size=800 characters (~150-200 words) - small enough for focused
    embeddings, large enough to keep a full clause/sentence together.
    
    chunk_overlap=100 characters - shares text between consecutive chunks
    so we don't cut a sentence in half and lose meaning.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
        # tries to split on paragraph breaks first, then lines,
        # then sentences, then words - keeps natural text boundaries
    )
    
    all_chunks = []
    
    for page in pages_content:
        text_pieces = splitter.split_text(page["text"])
        
        for i, piece in enumerate(text_pieces):
            all_chunks.append({
                "chunk_id": f"{page['source_file']}_p{page['page']}_c{i}",
                "text": piece,
                "source_file": page["source_file"],
                "page": page["page"]
            })
    
    return all_chunks


if __name__ == "__main__":
    pages = extract_all_policies()
    chunks = chunk_documents(pages)
    
    print(f"\nTotal chunks created: {len(chunks)}")
    print("\n--- Sample chunk ---")
    print("ID:", chunks[0]["chunk_id"])
    print("Source:", chunks[0]["source_file"], "- Page", chunks[0]["page"])
    print("Text:", chunks[0]["text"][:300])