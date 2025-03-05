import streamlit as st
import os
import sys

# Page configuration first, so we can display errors properly
st.set_page_config(
    page_title="JFK Documents RAG System",
    page_icon="📚",
    layout="wide"
)

# Initialize variables to avoid undefined errors
show_citations = True
show_similarity = True

# Try importing dependencies and handle errors gracefully
missing_packages = []
try:
    from dotenv import load_dotenv
except ImportError as e:
    missing_packages.append("python-dotenv")

try:
    import pandas as pd
except ImportError as e:
    missing_packages.append("pandas")

try:
    import plotly.express as px
except ImportError as e:
    missing_packages.append("plotly")

# Only import these after confirming the basic imports work
if not missing_packages:
    try:
        from rag_pipeline import create_rag_pipeline_from_env
        from text_similarity import TextSimilarityCalculator
    except ImportError as e:
        st.error(f"Error importing RAG components: {str(e)}")
        st.info("Please check that the RAG pipeline modules are installed correctly.")
        st.stop()

# If any required packages are missing, show installation instructions
if missing_packages:
    st.error(f"Missing required packages: {', '.join(missing_packages)}")
    st.info("Please install the missing packages using pip:")
    st.code(f"pip install {' '.join(missing_packages)}")
    
    # Provide instructions for updating requirements.txt
    st.markdown("### Fixing the Streamlit App")
    st.markdown("""
    It looks like some required packages are missing. To fix this issue:
    
    1. Make sure your `requirements.txt` file includes these packages:
    ```
    streamlit>=1.30.0
    setuptools>=68.0.0
    wheel>=0.40.0
    numpy>=1.26.0
    fastapi==0.103.1
    uvicorn==0.23.2
    python-dotenv==1.0.0
    pinecone-client==2.2.2
    openai==0.28.0
    plotly==5.18.0
    pandas>=2.1.0
    ```
    
    2. Push the updated requirements.txt to your repository
    3. Restart the Streamlit app
    """)
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
    query_params = st.query_params
    url_show_citations = query_params.get("show_citations", ["true"])[0].lower() == "true" if "show_citations" in query_params else True
    url_show_similarity = query_params.get("show_similarity", ["true"])[0].lower() == "true" if "show_similarity" in query_params else True
    
    show_citations = st.checkbox("Include citations", value=url_show_citations)
    show_similarity = st.checkbox("Show similarity metrics", value=url_show_similarity)
    
    st.header("Frontend Integration")
    st.markdown(f"""
    This Streamlit app can be integrated with your React frontend.
    Streamlit URL: https://cusfur3mwz8svmncsjjvvd.streamlit.app/
    
    **URL Parameters:**
    - query: Your search query
    - show_citations: true/false
    - show_similarity: true/false
    - submit: true to automatically run the search
    """)
    
    st.header("About")
    st.markdown("""
    This RAG (Retrieval-Augmented Generation) system uses:
    - Custom Euclidean distance-based retrieval
    - OpenAI for embeddings and answer generation
    - JFK assassination document corpus
    """)

# Check if we have a query parameter
url_query = st.query_params.get("query", [""])[0] if "query" in st.query_params else ""
url_submit = st.query_params.get("submit", ["false"])[0].lower() == "true" if "submit" in st.query_params else False

# Main query input - use URL parameter if available
query = st.text_input("Ask a question about JFK:", value=url_query, placeholder="e.g., What happened to JFK?")

# Submit button
submit = st.button("Search") or url_submit

if submit and query:
    # Check if pipeline is initialized
    if not pipeline or not similarity_calculator:
        st.error("The RAG pipeline is not properly initialized. Please check the configuration.")
        st.stop()
        
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
                    try:
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
                    except Exception as e:
                        st.warning(f"Could not generate plotly chart: {str(e)}")
                        # Fallback to a text representation
                        st.write(df)
                
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
            import traceback
            st.code(traceback.format_exc())
else:
    if submit:  # If the submit button was pressed but there's no query
        st.warning("Please enter a question to search.")
    else:  # First load of the page
        st.info("Enter a question above and click 'Search' to get started.")

# Add JSON response endpoint for API-like functionality
if 'format' in st.query_params and st.query_params['format'][0] == 'json' and query and submit:
    import json
    from streamlit.components.v1 import html
    
    # Create a simple JSON response
    response_data = {
        "query": query,
        "response": result.get("response", ""),
        "num_docs_retrieved": result.get("num_docs_retrieved", 0)
    }
    
    if show_citations and "citations" in result:
        response_data["citations"] = result["citations"]
        
    if show_similarity and "confidence" in result:
        response_data["confidence"] = result["confidence"]
    
    # Display as JSON in a hidden div for scraping
    html(f"""
    <div id="api-response" style="display:none;">
        {json.dumps(response_data)}
    </div>
    """)