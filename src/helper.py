from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import config

#Extract Data From the PDF File
def load_pdf_file(data):
    loader = DirectoryLoader(data,
                            glob=config.GLOB_PATTERNS,
                            loader_cls=PyPDFLoader)

    documents = loader.load()
    return documents


#Split the Data into Text Chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


#Download the Embeddings from HuggingFace 
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    return embeddings


#Download the Embeddings from Google
def download_google_embeddings(api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model=config.EMBEDDING_MODEL, google_api_key=api_key)
    return embeddings
