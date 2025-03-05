import streamlit as st

# Page configuration
st.set_page_config(
    page_title="JFK Documents RAG System",
    page_icon="📚",
    layout="wide"
)

# Header
st.title("📚 JFK Documents RAG System")
st.markdown("Ask questions about JFK assassination documents and get AI-powered answers with citations and similarity metrics.")

# Import only after basic UI is set up
try:
    import pandas as pd
    import os
    from dotenv import load_dotenv
    from rag_pipeline import create_rag_pipeline_from_env
    from text_similarity import TextSimilarityCalculator
    
    # Load environment variables
    load_dotenv()
    
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
    
    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        show_citations = st.checkbox("Include citations", value=True)
        show_similarity = st.checkbox("Show similarity metrics", value=True)
        
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
                    
                    # Use streamlit's native bar chart instead of plotly
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
                import traceback
                st.code(traceback.format_exc())
    else:
        if submit:  # If the submit button was pressed but there's no query
            st.warning("Please enter a question to search.")
        else:  # First load of the page
            st.info("Enter a question above and click 'Search' to get started.")

except ImportError as e:
    st.error(f"Error importing required dependencies: {str(e)}")
    st.info("Please check that all required packages are installed correctly.")
    
    # Create a REST API endpoint for your React app using Streamlit's components
    st.markdown("""
    ## REST API for Frontend
    
    This Streamlit app can also function as a REST API endpoint for your React frontend.
    
    In your React app, modify your API.js file to use the following approach:
    
    ```javascript
    import axios from 'axios';
    
    export const fetchAnswer = async (query, withCitations = true, withSimilarity = false) => {
      try {
        const formData = new FormData();
        formData.append('query', query);
        formData.append('with_citations', withCitations);
        formData.append('include_similarity', withSimilarity);
        
        // Instead of API endpoint, use form submission approach
        const response = await axios.post(
          'https://lennoxanderson.com/rag-proxy.php', 
          formData
        );
        
        return response.data;
      } catch (error) {
        console.error('Error connecting to RAG server:', error);
        throw new Error('Could not process your query. Please try again later.');
      }
    };
    ```
    
    Then create a simple PHP proxy on your website:
    
    ```php
    <?php
    // rag-proxy.php
    header('Content-Type: application/json');
    header('Access-Control-Allow-Origin: *');
    header('Access-Control-Allow-Methods: GET, POST, OPTIONS');
    header('Access-Control-Allow-Headers: Content-Type');
    
    // Example static response
    $response = [
        'response' => 'This is a static sample response about JFK. The actual implementation would query the RAG system.',
        'retrieved_docs' => [
            [
                'doc_title' => 'Sample Document',
                'similarity' => 0.85,
                'similarity_category' => 'High',
                'text' => 'Sample text about JFK...'
            ]
        ],
        'confidence' => [
            'level' => 'High',
            'score' => 0.85,
            'explanation' => 'The system found relevant documents that match your query well.'
        ],
        'num_docs_retrieved' => 1
    ];
    
    echo json_encode($response);
    ?>
    ```
    """)
    
    st.warning("This is a fallback mode with limited functionality.")