conf = {
    "collection_name" : "research_papers_collection",
    "db_path" : "qdrant_db_data",
    "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
    "llm" : "qwen2.5:7b",
    "top_k" : 4,

    "prompt": """Use the following context to answer the question concisely and most importantly accurately. If the context does not provide sufficient information, say you do not know. Do not Hallucinate.
    Context: {context}"

    Question: {query}"
  
    Answer: """
}