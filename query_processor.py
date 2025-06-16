from langchain.embeddings import HuggingFaceEmbeddings, OllamaEmbeddings
from qdrant_client import QdrantClient
from langchain.vectorstores import Qdrant
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from conf import conf


def get_context(llm, query: str, qdrant):
    # embeddings = HuggingFaceEmbeddings(model_name = conf["embed_model"])
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    retriever = qdrant.as_retriever()
    # client = QdrantClient(":memory:")
    #Qdrant(client=client, collection_name=conf["collection_name"], embeddings=embeddings)

    # contexts = db.similarity_search(query=query, k = conf["top_k"])
    Query_prompt = PromptTemplate(input_variables=["question"],
                                  template="""You are a helpful language assistant. Your task is to generate 2 different versions of the given query. This is to retrieve relevant document chunks from a vector database. This helps in having multiple perspectives on the given query. Your goal is to help the user overcome some of the limitations of the distance-based similarity search. Provide these alternative questions below separated by newlines. Question: {question}""",)
    
    multi_query_retriever = MultiQueryRetriever.from_llm(
                retriever,llm, prompt=Query_prompt
    )
    contexts = multi_query_retriever.get_relevant_documents(query)
    print(f"@@@@@@@@@@@@@@@@@@@\n\n\n Extracted contexts: \n{contexts}\n\n\n@@@@@@@@@@@")
    return contexts

def generate_answer(llm, query: str, client):
    contexts = get_context(llm, query, client)
    joined_context = "\n\n".join([context.page_content for context in contexts])
    prompt = PromptTemplate(input_variables=["context", "query"], template=conf["prompt"])
    prompt = prompt.format(context = joined_context, query = query)
    messages = [SystemMessage(content = "You are a helpful assistant that answers questions based on given context related to research papers"),
                HumanMessage(content = prompt)]
    answer = llm.invoke(messages)
    return answer.content.strip()





    


