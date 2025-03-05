import streamlit as st
import os
import sys

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
    import plotly.express as px
    # Only import these after confirming the basic imports work
    from rag_pipeline import create_rag_pipeline_from_env
    from text_similarity import TextSimilarityCalculator
    
    # Import FastAPI server
    from streamlit_api import app as api_app, api_thread  # This will start the API server
except ImportError as e:
    st.error(f"Error importing required dependencies: {str(e)}")
    st.info("Please check that all required packages are installed correctly.")
    st.stop()

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

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    # Check if we have URL parameters for these settings
    url_show_citations = st.experimental_get_query_params().get("show_citations", [True])[0]
    url_show_similarity = st.experimental_get_query_params().get("show_similarity", [True])[0]
    
    # Convert string query parameters to boolean
    if isinstance(url_show_citations, str):
        url_show_citations = url_show_citations.lower() == "true"
    if isinstance(url_show_similarity, str):
        url_show_similarity = url_show_similarity.lower() == "true"
    
    show_citations = st.checkbox("Include citations", value=url_show_citations)
    show_similarity = st.checkbox("Show similarity metrics", value=url_show_similarity)
    
    st.header("API Access")
    st.markdown(f"""
    This application exposes an API for integration with your front-end.
    API endpoint: http://localhost:8000/query
    """)
    
    st.header("About")
    st.markdown("""
    This RAG (Retrieval-Augmented Generation) system uses:
    - Custom Euclidean distance-based retrieval
    - OpenAI for embeddings and answer generation
    - JFK assassination document corpus
    """)

# Check if we have a query parameter
url_query = st.experimental_get_query_params().get("query", [""])[0]
url_submit = st.experimental_get_query_params().get("submit", [False])[0]

# Convert string query parameter to boolean
if isinstance(url_submit, str):
    url_submit = url_submit.lower() == "true"

# Main query input - use URL parameter if available
query = st.text_input("Ask a question about JFK:", value=url_query, placeholder="e.g., What happened to JFK?")

# Submit button
submit = st.button("Search") or url_submit

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
                
                # Plot similarity scores if similarity data is available
                if show_similarity and all("similarity" in doc for doc in result["retrieved_docs"]):
                    fig = px.bar(
                        df, 
                        x="Title", 
                        y="Similarity",
                        color="Category",
                        color_discrete_map={
                            "Very High": "green",
                            "High": "blue",
                            "Moderate": "orange",
                            "Low": "red",
                            "Very Low": "darkred"
                        },
                        labels={"Similarity": "Match Score", "Title": "Document"},
                        title="Document Relevance to Query"
                    )
                    st.plotly_chart(fig)
                
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