import os
from langchain_docling.loader import DoclingLoader
from langchain_community.document_loaders import UnstructuredPDFLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from langchain.vectorstores import Qdrant
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain.embeddings import HuggingFaceEmbeddings, JinaEmbeddings, OllamaEmbeddings
import uuid

from conf import conf

def load_documents(path: str):
    # print("Loading Doc")
    # loader = DoclingLoader(path)
    loader = PyMuPDFLoader(path)
    return loader.load()

def store_the_chunks(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    chunks = splitter.split_documents(docs)
    chunks = [chunk for chunk in chunks if chunk.page_content]
    print(f"@@@@@@@@@@@@@@@@@@@@@@@@@@@\n\n\nAll the chunks: \n{chunks}\n\n\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    # embeddings = HuggingFaceEmbeddings(model_name = conf["embed_model"])
    embeddings = OllamaEmbeddings(model="nomic-embed-text") 
    client = QdrantClient(":memory:")

    if client.collection_exists(conf["collection_name"]):
        client.delete_collection(collection_name= conf["collection_name"])
    client.create_collection(collection_name= conf["collection_name"],
                                 vectors_config = VectorParams(size = 768, distance= Distance.COSINE))
    
    qdrant = Qdrant(
        client = client,
        collection_name = conf["collection_name"],
        embeddings=embeddings,
        
        content_payload_key="page_content"
    )
    qdrant.add_documents(chunks)
    # return client
    return qdrant

    # texts = [chunk.page_content for chunk in chunks]
    # metadatas = [chunk.metadata for chunk in chunks]
    # vectors = embeddings.embed_documents(texts)

    # points = []
    # for idx, (vector, metadata, text) in enumerate(zip(vectors, metadatas, texts)):
    #     metadata["page_content"] = text
    #     points.append(PointStruct(
    #         id = str(uuid.uuid4()),
    #         vector = vector,
    #         payload = metadata
    #     ))

    # client.upsert(collection_name=conf["collection_name"], points = points)

    # there is a langchain abstraction for this insertion process:
    # we can use langchain.vectorstores.Qdrant.from_documents()
    # this function does everything under the hood.
