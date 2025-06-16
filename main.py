import streamlit as st
from langchain_ollama import ChatOllama
from doc_processor import load_documents, store_the_chunks
from query_processor import generate_answer
from conf import conf
import tempfile
import os

st.set_page_config(page_title="Research Paper chatbot")
st.title("Research Paper Assistant")


def initialize():
    return ChatOllama(model = conf["llm"])


if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = ""

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def chat_interface(llm):
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if query:= st.chat_input("Enter your query"):
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})
        with st.spinner("Give me a second..."):
            response = generate_answer(llm, query= query, client=st.session_state["db"])
        with st.chat_message("assistant"): 
            st.write(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.conversation_history += f"Human: {query}\n\nAI: {response}\n\n"


uploaded_doc = st.file_uploader(label="Upload your pdf file", type=["docx", "pdf"], accept_multiple_files=False)

if uploaded_doc:
    if 'qdrant_initialized' not in st.session_state:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_doc.read())
            tmp_path = tmp_file.name
        with st.spinner("Processing the file"):
            st.session_state["db"] = store_the_chunks(load_documents(tmp_path))
            st.success("Document processed successfully")
            st.session_state.qdrant_initialized = True
    llm = initialize()
    chat_interface(llm = llm)


    