import os
from uuid import uuid4
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import login

from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings


# ---------------------
# CONFIG & SETUP
# ---------------------
load_dotenv()

CHUNK_SIZE = 1000
EMBEDDING_MODEL = "Alibaba-NLP/gte-base-en-v1.5"
VECTOR_STORE_DIR = Path(__file__).parent / "resources/vector_store"
COLLECTION_NAME = "real_estate"
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not hf_token:
    raise ValueError("❌ Hugging Face API token not found. Please set HUGGINGFACEHUB_API_TOKEN in .env or Streamlit secrets.")

# Authenticate with Hugging Face
login(token=hf_token)

llm = None
vector_store = None


# ---------------------
# CORE FUNCTIONS
# ---------------------
def initialize_components():
    global llm, vector_store

    if llm is None:
        llm = ChatGroq(model="llama3-70b-8192", temperature=0.9, max_tokens=500)

    if vector_store is None:
        ef = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"trust_remote_code": True}
        )
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=ef,
            persist_directory=str(VECTOR_STORE_DIR)
        )


def process_urls(urls):
    yield "Initializing components...✅"
    initialize_components()

    vector_store.delete_collection()
    yield "Resetting vector store...✅"

    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    yield "Splitting text into chunks...✅"
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=CHUNK_SIZE
    )
    docs = text_splitter.split_documents(data)

    yield "Adding chunks to vector database...✅"
    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(docs, ids=uuids)

    yield "Done adding docs to vector database...✅"


def generate_answer(query):
    if not vector_store:
        raise RuntimeError("Vector Database is not initialized")

    chain = RetrievalQAWithSourcesChain.from_llm(
        llm=llm, retriever=vector_store.as_retriever()
    )
    result = chain.invoke({"question": query})
    sources = result.get("sources", "")

    return result["answer"], sources


# ---------------------
# STREAMLIT APP (UI)
# ---------------------
st.set_page_config(page_title="Smart URL Answer Bot", page_icon="🔗", layout="centered")

st.title("🔗 Your Smart URL Answer Bot")
st.markdown("Ask questions based on content from your favorite web pages.")

# Sidebar - URL input
st.sidebar.header("📥 Enter URLs to Process")
url1 = st.sidebar.text_input("URL 1")
url2 = st.sidebar.text_input("URL 2")
url3 = st.sidebar.text_input("URL 3")

status_placeholder = st.empty()

if st.sidebar.button("🚀 Process URLs"):
    urls = [url for url in (url1, url2, url3) if url.strip() != ""]
    if not urls:
        status_placeholder.error("⚠️ Please enter at least one valid URL.")
    else:
        for status in process_urls(urls):
            status_placeholder.info(status)

st.markdown("---")

# Question Input
st.subheader("💬 Ask a Question")
query = st.text_input("Type your question here and press Enter:")

if query:
    try:
        answer, sources = generate_answer(query)
        st.success("✅ Answer Generated!")

        st.markdown("### 🧠 Answer")
        st.write(answer)

        if sources:
            st.markdown("### 📚 Sources")
            for source in sources.strip().split("\n"):
                if source.strip():
                    st.markdown(f"- {source}")
    except RuntimeError:
        st.error("⚠️ You must process the URLs first before asking a question.")
