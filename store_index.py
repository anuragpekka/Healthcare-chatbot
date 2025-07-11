from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from tqdm.auto import tqdm
import os

from src.helper import load_pdf_file, text_split, download_google_embeddings
import config


load_dotenv()

PINECONE_API_KEY=os.environ.get("PINECONE_API_KEY")
GOOGLE_API_KEY=os.environ.get("GOOGLE_API_KEY")

print(f"Extracting data from files in {config.DATA_DIR} directory.")
extracted_data=load_pdf_file(data=config.DATA_DIR)

print(f"Creating chuncks of data extracted data.")
text_chunks=text_split(extracted_data)

print(f"Fetching Huggingface embeddings {config.EMBEDDING_MODEL}")
embeddings = download_google_embeddings(GOOGLE_API_KEY)

index_name = config.INDEX_NAME

print(f"Creating new index {index_name} in Pinecone DB.")
pc = Pinecone(api_key=PINECONE_API_KEY)
try:
    pc.create_index(
        name=index_name,
        dimension=config.INDEX_EMBEDDING_SIZE, 
        metric=config.INDEX_METRIC, 
        spec=ServerlessSpec(
            cloud=config.INDEX_CLOUD, 
            region=config.INDEX_REGION
        ) 
    ) 
    print(f"Index '{index_name}' created successfully.")
except Exception as e:
    print(f"Error while creating index {index_name}: {e}")

print(f"Upserting data into index {index_name} in Pinecone DB.")
try:
    batch_size = 50

    # Iterate through text_chunks in batches and upsert
    for i in tqdm(range(0, len(text_chunks), batch_size)):
        batch_documents = text_chunks[i:i + batch_size]

        PineconeVectorStore.from_documents(
            documents=batch_documents,
            index_name=index_name,
            embedding=embeddings
        )
    print("All documents upserted to Pinecone.")
except Exception as e:
    print(f"Error while upserting data to Pinecone")