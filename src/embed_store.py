from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from chunk import chunk_documents
from extract import extract_all_policies

# This model runs locally, free, no API needed - good balance of
# speed and quality for a portfolio project (not the biggest model,
# but accurate enough and fast on a normal laptop)
MODEL_NAME = "all-MiniLM-L6-v2"

def build_vector_store(chunks, save_path="data/vector_store"):
    """
    Converts each chunk's text into a vector, stores all vectors in a
    FAISS index for fast similarity search, and saves the chunk metadata
    separately (FAISS only stores numbers, not our text/page/source info).
    """
    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    texts = [chunk["text"] for chunk in chunks]

    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    # Build the FAISS index. IndexFlatL2 = exact search using
    # straight-line distance between vectors - simplest, most accurate
    # option, fine for a project this size (thousands of chunks, not millions)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save the FAISS index itself
    faiss.write_index(index, f"{save_path}.index")

    # Save the chunk metadata (text, source, page) in the SAME order
    # as the vectors, so we can map a search result back to its text later
    with open(f"{save_path}_metadata.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print(f"Vector store saved: {save_path}.index + {save_path}_metadata.pkl")
    return index, chunks


if __name__ == "__main__":
    pages = extract_all_policies()
    chunks = chunk_documents(pages)
    build_vector_store(chunks)