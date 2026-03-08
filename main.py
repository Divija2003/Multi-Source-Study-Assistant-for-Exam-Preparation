import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import pipeline
import pickle
import os

st.set_page_config(page_title="exam assistant", layout="wide")
st.title("Multi-Source study assistant for exam preparation")

INDEX_FILE = "faiss_index.pkl"


# SIDEBAR: SEPARATE URL BARS

st.sidebar.header("Enter URLs")

url1 = st.sidebar.text_input("URL 1")
url2 = st.sidebar.text_input("URL 2")
url3 = st.sidebar.text_input("URL 3")

urls = [u for u in [url1, url2, url3] if u.strip()]

# PROCESS URLS

if st.sidebar.button("Process URLs"):
    if not urls:
        st.warning("Please enter at least one URL.")
        st.stop()

    documents = []
    headers = {"User-Agent": "Mozilla/5.0"}

    st.subheader("URL Content Preview")

    for url in urls:
        try:
            r = requests.get(url, headers=headers, timeout=10)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")

            paragraphs = [
                p.get_text().strip()
                for p in soup.find_all("p")
                if len(p.get_text(strip=True)) > 50
            ]

            if not paragraphs:
                st.warning(f"No readable text found in {url}")
                continue

            with st.expander(f"Preview: {url}"):
                st.write(" ".join(paragraphs[:2]))

            for p in paragraphs:
                documents.append(
                    Document(page_content=p, metadata={"source": url.rstrip("/")})

                )

        except Exception as e:
            st.error(f"Failed to load {url}: {e}")

    if not documents:
        st.error("No valid content could be loaded.")
        st.stop()

 
    # SPLIT + EMBEDDINGS
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    db = FAISS.from_documents(docs, embeddings)

    with open(INDEX_FILE, "wb") as f:
        pickle.dump(db, f)

    st.success("✅ URLs processed and indexed!")

# QUESTION ANSWERING

st.markdown("---")
st.subheader("Ask a Question")

question = st.text_input("Type your question")

if question:
    if not os.path.exists(INDEX_FILE):
        st.error("Please process URLs first.")
        st.stop()

    with open(INDEX_FILE, "rb") as f:
        db = pickle.load(f)

    retrieved_docs = db.similarity_search(question, k=7)
    context_text = " ".join(d.page_content for d in retrieved_docs)

    # 🔹 ONLY MODEL CHANGE IS HERE
    qa = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        device=-1
    )

    prompt = f"""
Answer the question using the context below.
If the answer is not present in the context, say you don't know.

Context:
{context_text}

Question:
{question}
"""

    result = qa(prompt, max_length=200)

    st.subheader("Answer")
    st.write(result[0]["generated_text"])

    st.subheader("Sources")

    unique_sources = set()

    for d in retrieved_docs:
      source = d.metadata.get("source")
      if source:
        unique_sources.add(source)

    for src in unique_sources:
       st.write(src)
