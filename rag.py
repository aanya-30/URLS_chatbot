#IMPORTING LIBRARIES

from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from huggingface_hub import login
import os

load_dotenv()

CHUNK_SIZE = 1000
EMBEDDING_MODEL="Alibaba-NLP/gte-base-en-v1.5"
VECTOR_STORE_DIR = Path("__file__"),"parent/resources/vector_store"
COLLECTION_NAME = "real_estate"
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

#ENSURE TOKEN IS SET
if not hf_token:
    raise ValueError("Hugging Face API token not found. Please set the HUGGINGFACEHUB_API_TOKEN in .env or a .")

#AUTHENTICATE BEFORE USING THE MODEL

login(hf_token)
llm=None
vector_store=None

def initialize_components():
    global llm, vector_store
   
    if llm is None:
        llm = ChatGroq(model="llama-3.3-70b-versatile",temperature=0.9, max_tokens=500)

    if vector_store is None:
        ef=HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"trust_remote_code": True},
        )

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=ef,
            persist_directory=str(VECTOR_STORE_DIR) 
       )

def process_urls(urls):
    """

    This function scrapes data from the urls and stores it in a vectpr db
    :param urls: input urls
    :return
    """

    #PRINT ("initialize components")
    yield "Initializing components...✅"
    initialize_components()
    vector_store.reset_collection()

    #PRINT ("load data from urls")
    yield "Resetting vector store...✅"
    loader=UnstructuredURLLoader(urls=urls)
    data=loader.load()

    #PRINT ("split text")
    yield "Splitting text into chunks...✅"
    text_splitter=RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", ""],
        chunk_size=CHUNK_SIZE,
    )


    docs=text_splitter.split_documents(data)
    
    #PRINT ("Add docs to vector db")
    yield "Adding chunks to vector store...✅"
    uuids=[str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(docs,ids=uuids)

    yield "Done adding docs to vector database...✅"

def generate_answer(query):
    if not vector_store:
        raise ValueError("Vector database is not initialized")

    chain=RetrievalQAWithSourcesChain.from_llm(llm=llm,retriever=vector_store.as_retriever())
    result=chain.invoke({"question":query}, return_only_outputs=True)
    sources=result.get("sources","")

    return result["answer"], sources
if __name__ == "__main__":

    process_urls(urls)
    answer, sources = generate_answer("Tell me what was the 30 year fixed mortage rate along with the date?")
    print(f"Answer:{answer}")
    print(f"Sources:{sources}")