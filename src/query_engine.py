import faiss
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()  # reads your .env file so GROQ_API_KEY becomes available

MODEL_NAME = "all-MiniLM-L6-v2"


class InsuranceRAGEngine:
    def __init__(self, store_path="data/vector_store"):
        print("Loading embedding model and vector store...")
        self.embed_model = SentenceTransformer(MODEL_NAME)
        self.index = faiss.read_index(f"{store_path}.index")

        with open(f"{store_path}_metadata.pkl", "rb") as f:
            self.chunks = pickle.load(f)

        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def retrieve(self, question, top_k=4):
        """
        Converts the question into a vector, finds the top_k most similar
        chunks in FAISS, and returns them along with a similarity score.
        Lower distance = more similar (FAISS L2 distance, not similarity %).
        """
        question_vector = self.embed_model.encode([question]).astype("float32")
        distances, indices = self.index.search(question_vector, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            chunk = self.chunks[idx]
            results.append({
                "text": chunk["text"],
                "source_file": chunk["source_file"],
                "page": chunk["page"],
                "distance": float(dist)
            })
        return results

    def retrieve_by_source(self, question, source_file, top_k=4):
        """
        Same as retrieve(), but only searches chunks belonging to one
        specific policy document. We do this by embedding the question
        once, then filtering AFTER the search to keep only chunks whose
        source_file matches - FAISS itself doesn't support filtering,
        so we over-fetch and filter in Python.
        """
        question_vector = self.embed_model.encode([question]).astype("float32")

        # Over-fetch (search more than top_k) because after filtering
        # by source, we might lose some matches - fetching extra
        # ensures we still end up with enough relevant chunks
        distances, indices = self.index.search(question_vector, top_k * 5)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            chunk = self.chunks[idx]
            if chunk["source_file"] == source_file:
                results.append({
                    "text": chunk["text"],
                    "source_file": chunk["source_file"],
                    "page": chunk["page"],
                    "distance": float(dist)
                })
            if len(results) >= top_k:
                break

        return results

    def build_prompt(self, question, retrieved_chunks):
        """
        Builds the prompt sent to the LLM. Critically, we instruct the model
        to ONLY use the provided context, and to say so explicitly if the
        answer isn't in the context - this is what prevents hallucination,
        which matters a lot for insurance (wrong answers = real consequences).
        """
        context_text = "\n\n".join(
            f"[Source: {c['source_file']}, Page {c['page']}]\n{c['text']}"
            for c in retrieved_chunks
        )

        prompt = f"""You are an insurance policy assistant. Answer the question
using ONLY the context below. If the context does not contain enough
information to answer confidently, say so explicitly - do not guess or
use outside knowledge.

Context:
{context_text}

Question: {question}

Answer clearly. At the end, list which sources you used."""

        return prompt

    def answer(self, question, top_k=4):
        retrieved_chunks = self.retrieve(question, top_k)

        # Confidence flag: if the best match's distance is too high,
        # the retrieved content probably isn't actually relevant.
        # This threshold is empirical - tune it once you see real
        # distances from your own documents.
        best_distance = retrieved_chunks[0]["distance"]
        low_confidence = best_distance > 1.0  # placeholder threshold

        prompt = self.build_prompt(question, retrieved_chunks)

        response = self.groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2  # low temperature = more factual, less creative
        )

        answer_text = response.choices[0].message.content

        return {
            "answer": answer_text,
            "sources": [(c["source_file"], c["page"]) for c in retrieved_chunks],
            "low_confidence": low_confidence
        }

    def compare_policies(self, question, file_a, file_b, top_k=4):
        """
        Retrieves relevant chunks separately from each policy, then
        asks the LLM to compare them side by side. Grounding is kept
        strict - just like single-policy Q&A, no outside knowledge.
        """
        chunks_a = self.retrieve_by_source(question, file_a, top_k)
        chunks_b = self.retrieve_by_source(question, file_b, top_k)

        if not chunks_a or not chunks_b:
            return {
                "answer": "Could not find relevant content in one or both policies for this topic.",
                "sources_a": [],
                "sources_b": []
            }

        context_a = "\n\n".join(f"[Page {c['page']}]\n{c['text']}" for c in chunks_a)
        context_b = "\n\n".join(f"[Page {c['page']}]\n{c['text']}" for c in chunks_b)

        prompt = f"""You are an insurance policy assistant. Compare the two
policies below on the following topic. Use ONLY the context provided for
each policy - do not use outside knowledge. Clearly structure your answer
as: what Policy A says, what Policy B says, and then the key differences.

Topic: {question}

--- Policy A: {file_a} ---
{context_a}

--- Policy B: {file_b} ---
{context_b}

Provide a clear, structured comparison."""

        response = self.groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        return {
            "answer": response.choices[0].message.content,
            "sources_a": [(c["source_file"], c["page"]) for c in chunks_a],
            "sources_b": [(c["source_file"], c["page"]) for c in chunks_b]
        }


if __name__ == "__main__":
    engine = InsuranceRAGEngine()

    test_question = "What is covered under accidental death benefit?"
    result = engine.answer(test_question)

    print("\n--- ANSWER ---")
    print(result["answer"])
    print("\n--- SOURCES ---")
    for src, page in result["sources"]:
        print(f"- {src}, page {page}")
    print("\nLow confidence:", result["low_confidence"])