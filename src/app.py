import streamlit as st
from query_engine import InsuranceRAGEngine
import os

st.set_page_config(page_title="Insurance Policy Assistant", layout="wide")
st.title("📄 Insurance Policy Q&A Assistant")

@st.cache_resource
def load_engine():
    return InsuranceRAGEngine()

engine = load_engine()

# Get list of available policy files from the metadata, so the
# dropdown always matches what's actually in the vector store
available_files = sorted(set(c["source_file"] for c in engine.chunks))

tab1, tab2 = st.tabs(["Ask a Question", "Compare Two Policies"])

with tab1:
    st.caption("Ask questions about your policies — answers are grounded in the actual documents, with sources cited.")
    question = st.text_input("Ask a question:", placeholder="e.g., What is covered under accidental death benefit?")
    top_k = st.slider("Number of source chunks to retrieve", 2, 8, 4, key="single_topk")

    if st.button("Get Answer") and question:
        with st.spinner("Searching and generating answer..."):
            result = engine.answer(question, top_k=top_k)

        if result["low_confidence"]:
            st.warning("⚠️ Low confidence — please verify with the actual policy document.")

        st.subheader("Answer")
        st.write(result["answer"])
        st.subheader("Sources")
        for src, page in result["sources"]:
            st.write(f"- **{src}**, page {page}")

with tab2:
    st.caption("Compare how two policies handle the same topic — e.g., exclusions, claim process, coverage limits.")

    col1, col2 = st.columns(2)
    with col1:
        policy_a = st.selectbox("Policy A", available_files, key="policy_a")
    with col2:
        remaining = [f for f in available_files if f != policy_a]
        policy_b = st.selectbox("Policy B", remaining, key="policy_b")

    compare_question = st.text_input("What do you want to compare?",
                                      placeholder="e.g., What are the exclusions for pre-existing conditions?")

    if st.button("Compare") and compare_question:
        with st.spinner("Retrieving from both policies and comparing..."):
            result = engine.compare_policies(compare_question, policy_a, policy_b)

        st.subheader("Comparison")
        st.write(result["answer"])

        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Sources from {policy_a}:**")
            for src, page in result["sources_a"]:
                st.write(f"- page {page}")
        with col2:
            st.write(f"**Sources from {policy_b}:**")
            for src, page in result["sources_b"]:
                st.write(f"- page {page}")