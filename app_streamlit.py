import streamlit as st

st.set_page_config(page_title="JFK RAG System", page_icon="📚")
st.title("JFK Documents RAG System")
st.write("This is a minimal version to get the application running.")

st.info("The complete RAG system is temporarily unavailable due to dependency issues. Please use the React frontend at https://lennoxanderson.com/mffrag/ for querying the system.")

st.markdown("""
### About
This RAG (Retrieval-Augmented Generation) system uses:
- Custom Euclidean distance-based retrieval
- OpenAI for embeddings and answer generation
- JFK assassination document corpus
""")