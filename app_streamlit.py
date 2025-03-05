import streamlit as st
import os
import sys
import threading

st.set_page_config(
    page_title="JFK Documents RAG System",
    page_icon="📚",
    layout="wide"
)

try:
    from dotenv import load_dotenv
    import pandas as pd
    from api_server import run_server
    from rag_pipeline import create_rag_pipeline_from_env
    from text_similarity import TextSimilarityCalculator
except ImportError as e:
    st.error(f"Error importing required dependencies: {str(e)}")
    st.info("Please check that all required packages are installed correctly.")
    st.stop()

api_thread = threading.Thread(target=run_server, daemon=True)
api_thread.start()

try:
    load_dotenv()
except Exception as e:
    st.warning(f"Warning loading environment variables: {str(e)}")

@st.cache_resource
def load_models():
    try:
        pipeline = create_rag_pipeline_from_env()
        
        openai_api_key = os.getenv("OPENAI_API_KEY")
        similarity_calculator = TextSimilarityCalculator(openai_api_key=openai_api_key)
        
        return pipeline, similarity_calculator
    except Exception as e:
        st.error(f"Error initializing models: {str(e)}")
        return None, None

pipeline, similarity_calculator = load_models()

st.title("📚 JFK Documents RAG System")
st.markdown("Ask questions about JFK assassination documents and get AI-powered answers with citations and similarity metrics.")

st.info("""
## API Server Running
Your React frontend can now connect to this Streamlit app via the API endpoint.

**Endpoint:** `https://cusfur3mwz8svmncsjjvvd.streamlit.app/api/query`  
**Method:** POST  
**Request Format:**
```json
{
  "query": "Your question about JFK",
  "with_citations": true,
  "include_similarity": true
}
```
""")

with st.sidebar:
    st.header("Settings")
    show_citations = st.checkbox("Include citations", value=True)
    show_similarity = st.checkbox("Show similarity metrics", value=True)
    
    st.header("API Information")
    st.markdown("""
    The API server is running on port 8000.
    
    Your React app can make POST requests to:
    - `/api/query` - Process queries
    - `/api/health` - Check server status
    """)
    
    st.header("About")
    st.markdown("""
    This RAG (Retrieval-Augmented Generation) system uses:
    - Custom Euclidean distance-based retrieval
    - OpenAI for embeddings and answer generation
    - JFK assassination document corpus
    """)

query = st.text_input("Ask a question about JFK:", placeholder="e.g., What happened to JFK?")

submit = st.button("Search")

if submit and query:
    with st.spinner("Searching documents and generating response..."):
        try:
            result = pipeline.process_query(query, with_citations=show_citations)
            
            if show_similarity and similarity_calculator:
                similarity_result = similarity_calculator.process_results_with_similarity(
                    query, 
                    result["retrieved_docs"]
                )
                result["retrieved_docs"] = similarity_result["results"]
                result["confidence"] = similarity_result["confidence"]
            
            st.markdown("### Response")
            st.markdown(result["response"])
            
            if show_similarity and "confidence" in result:
                confidence = result["confidence"]
                
                st.markdown("### Confidence")
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    level_color = {
                        "Very High": "green",
                        "High": "blue",
                        "Moderate": "orange",
                        "Low": "red",
                        "Very Low": "darkred"
                    }
                    level = confidence["level"]
                    color = level_color.get(level, "gray")
                    st.markdown(f"<h3 style='color: {color};'>{level}</h3>", unsafe_allow_html=True)
                    
                with col2:
                    st.progress(confidence["score"])
                    st.caption(f"{int(confidence['score'] * 100)}% - {confidence['explanation']}")
            
            if result["retrieved_docs"]:
                st.markdown("### Source Documents")
                
                doc_data = []
                for doc in result["retrieved_docs"]:
                    doc_data.append({
                        "Title": doc["doc_title"],
                        "Similarity": doc.get("similarity", 0),
                        "Category": doc.get("similarity_category", "N/A"),
                        "Text": doc["text"][:150] + "..." if len(doc["text"]) > 150 else doc["text"]
                    })
                
                df = pd.DataFrame(doc_data)
                
                if show_similarity and all("similarity" in doc for doc in result["retrieved_docs"]):
                    st.subheader("Document Relevance to Query")
                    st.bar_chart(df.set_index("Title")["Similarity"])
                
                for i, doc in enumerate(doc_data):
                    with st.expander(f"{i+1}. {doc['Title']}"):
                        if show_similarity and "Similarity" in doc:
                            st.caption(f"Match score: {doc['Similarity']:.2f} ({doc['Category']})")
                        st.text(doc["Text"])
            
            if show_citations and "citations" in result and result["citations"]:
                st.markdown("### Citations")
                citations = result["citations"]
                for key, source in citations.items():
                    st.markdown(f"**[{key}]** {source['title']}")
            
            st.caption(f"Found {result['num_docs_retrieved']} relevant documents")
            
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
else:
    if submit:  
        st.warning("Please enter a question to search.")
    else:  
        st.info("Enter a question above and click 'Search' to get started.")