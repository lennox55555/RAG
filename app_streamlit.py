import streamlit as st
import os
import sys
import threading

# Page configuration first, so we can display errors properly
st.set_page_config(
    page_title="JFK Documents RAG System",
    page_icon="📚",
    layout="wide"
)

# Try importing dependencies and handle errors gracefully
try:
    from dotenv import load_dotenv
    import pandas as pd
    # Import API server module
    from api_server import run_server
    # Only import these after confirming the basic imports work
    from rag_pipeline import create_rag_pipeline_from_env
    from text_similarity import TextSimilarityCalculator
except ImportError as e:
    st.error(f"Error importing required dependencies: {str(e)}")
    st.info("Please check that all required packages are installed correctly.")
    st.stop()

# Start the API server in a separate thread
api_thread = threading.Thread(target=run_server, daemon=True)
api_thread.start()

# Load environment variables
try:
    load_dotenv()
except Exception as e:
    st.warning(f"Warning loading environment variables: {str(e)}")

# Initialize the RAG pipeline and similarity calculator
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

# Header
st.title("📚 JFK Documents RAG System")
st.markdown("Ask questions about JFK assassination documents and get AI-powered answers with citations and similarity metrics.")

# API Information Box
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

# Sidebar for settings
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

# Main query input
query = st.text_input("Ask a question about JFK:", placeholder="e.g., What happened to JFK?")

# Submit button
submit = st.button("Search")

if submit and query:
    # Show spinner while processing
    with st.spinner("Searching documents and generating response..."):
        try:
            # Process the query
            result = pipeline.process_query(query, with_citations=show_citations)
            
            # Add similarity data if requested
            if show_similarity and similarity_calculator:
                similarity_result = similarity_calculator.process_results_with_similarity(
                    query, 
                    result["retrieved_docs"]
                )
                result["retrieved_docs"] = similarity_result["results"]
                result["confidence"] = similarity_result["confidence"]
            
            # Display the response
            st.markdown("### Response")
            st.markdown(result["response"])
            
            # Display confidence if available
            if show_similarity and "confidence" in result:
                confidence = result["confidence"]
                
                # Create a confidence meter
                st.markdown("### Confidence")
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    # Display confidence level with appropriate color
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
                    # Create a progress bar for confidence score
                    st.progress(confidence["score"])
                    st.caption(f"{int(confidence['score'] * 100)}% - {confidence['explanation']}")
            
            # Display source documents with similarity metrics
            if result["retrieved_docs"]:
                st.markdown("### Source Documents")
                
                # Create a DataFrame for the documents
                doc_data = []
                for doc in result["retrieved_docs"]:
                    doc_data.append({
                        "Title": doc["doc_title"],
                        "Similarity": doc.get("similarity", 0),
                        "Category": doc.get("similarity_category", "N/A"),
                        "Text": doc["text"][:150] + "..." if len(doc["text"]) > 150 else doc["text"]
                    })
                
                df = pd.DataFrame(doc_data)
                
                # Use streamlit native bar chart instead of plotly
                if show_similarity and all("similarity" in doc for doc in result["retrieved_docs"]):
                    st.subheader("Document Relevance to Query")
                    st.bar_chart(df.set_index("Title")["Similarity"])
                
                # Display document content
                for i, doc in enumerate(doc_data):
                    with st.expander(f"{i+1}. {doc['Title']}"):
                        if show_similarity and "Similarity" in doc:
                            st.caption(f"Match score: {doc['Similarity']:.2f} ({doc['Category']})")
                        st.text(doc["Text"])
            
            # Display citations
            if show_citations and "citations" in result and result["citations"]:
                st.markdown("### Citations")
                citations = result["citations"]
                for key, source in citations.items():
                    st.markdown(f"**[{key}]** {source['title']}")
            
            # Display metadata
            st.caption(f"Found {result['num_docs_retrieved']} relevant documents")
            
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
else:
    if submit:  # If the submit button was pressed but there's no query
        st.warning("Please enter a question to search.")
    else:  # First load of the page
        st.info("Enter a question above and click 'Search' to get started.")