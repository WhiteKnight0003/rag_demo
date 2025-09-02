import streamlit as st
from utils.completion import generate_completion
from utils.chunking import chunk_text
from utils.embedding import get_embedding
from utils.prompt import prompt
from utils.retrieval import retrieve_chunks, load_faiss_index

st.title("RAG Application with Streamlit")
st.write("Ask questions grounded in the life and mission of sudhanshun kumar")

query = st.text_input("Enter your question about sudhanshu:")

if query:
    index, chunk_mapping = load_faiss_index()
    top_chunks = retrieve_chunks(query, index, chunk_mapping, k=4)
    prompts = prompt(top_chunks, query)
    response = generate_completion(prompts)
    st.write("### Answer:")
    st.write(response)

    with st.expander('Retrieved Chunks'):
        for chunk in top_chunks:
            st.markdown(f"- {chunk}")