# app.py

import streamlit as st
import streamlit.components.v1 as components
import retrieval_functions as rf
import re
from time import sleep

# Streamlit page config
st.set_page_config(
    page_title="🎬 Video QA with Multimodal RAG",
    layout="wide",
    page_icon="🎥",
    initial_sidebar_state="expanded"
)

# Modern CSS styling with gradient backgrounds and animations
st.markdown("""
    <style>
    /* Main background and text */
    body {
        background-color: #f8f9fa;
        color: #212529;
    }
    
    /* Gradient header */
    .gradient-header {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        padding: 2rem 1rem;
        border-radius: 0 0 15px 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    /* Card styling with hover effect */
    .card {
        background: white;
        border-radius: 12px;
        box-shadow: 0 6px 15px rgba(0,0,0,0.05);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border-left: 4px solid #6a11cb;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 20px rgba(0,0,0,0.1);
    }
    
    /* Highlight text */
    mark {
        background: linear-gradient(120deg, #f6d365 0%, #fda085 100%);
        color: #212529;
        padding: 0.2em 0.4em;
        border-radius: 4px;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(37, 117, 252, 0.4);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        border-right: 1px solid #dee2e6;
    }
    
    /* Video container */
    .video-container {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 2rem;
    }
    
    /* Responsive video iframe */
    .video-responsive {
        position: relative;
        padding-bottom: 56.25%;
        height: 0;
        overflow: hidden;
        border-radius: 8px;
    }
    
    .video-responsive iframe {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        border-radius: 8px;
    }
    
    /* Timestamp link */
    .timestamp-link {
        color: #2575fc !important;
        font-weight: 500;
        text-decoration: none;
        transition: all 0.2s ease;
    }
    
    .timestamp-link:hover {
        color: #6a11cb !important;
        text-decoration: underline;
    }
    
    /* Progress bar color */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
    }
    
    /* Custom success message */
    .custom-success {
        background: linear-gradient(135deg, #4BB543 0%, #2E8B57 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header with gradient background
st.markdown("""
    <div class="gradient-header">
        <h1 style="margin: 0; text-align: center;">🎬 Video Question Answering with Multimodal RAG</h1>
        <p style="margin: 0.5rem 0 0; text-align: center; opacity: 0.9;">
            Ask questions about video content and get precise answers with multiple retrieval methods
        </p>
    </div>
""", unsafe_allow_html=True)

# Sidebar with improved styling
with st.sidebar:
    st.markdown("""
        <h2 style="color: #6a11cb; border-bottom: 2px solid #6a11cb; padding-bottom: 0.5rem;">
            ⚙️ Retrieval Settings
        </h2>
    """, unsafe_allow_html=True)
    
    retrieval_method = st.selectbox(
        "**Retrieval Method:**",
        ["FAISS", "pgvector-IVFFLAT", "pgvector-HNSW", "TF-IDF", "BM25"],
        help="Choose the vector search method for retrieving video segments"
    )

    top_k = st.slider(
        "**Top-K Results:**",
        min_value=1,
        max_value=5,
        value=3,
        help="Number of most relevant segments to return"
    )
    
    st.markdown("---")
    st.markdown("""
        <div style="background: rgba(106, 17, 203, 0.1); padding: 1rem; border-radius: 8px;">
            <p style="margin: 0; color: #6a11cb; font-weight: 500;">ℹ️ <strong>Tip:</strong> Try different retrieval methods for varied results</p>
        </div>
    """, unsafe_allow_html=True)

# Video player section with improved layout
st.markdown("""
    <div class="video-container">
        <h3 style="margin-top: 0; color: #6a11cb;">🎞️ Currently Analyzing</h3>
        <div class="video-responsive">
            <iframe 
                src="https://www.youtube.com/embed/dARr3lGKwk8?modestbranding=1&rel=0" 
                frameborder="0" 
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                allowfullscreen>
            </iframe>
        </div>
    </div>
""", unsafe_allow_html=True)

# Question input with improved styling
st.markdown("### 💬 Ask About the Video")
question = st.text_input(
    "Enter your question about the video content:",
    placeholder="E.g., What are the key points about...?",
    label_visibility="collapsed"
)

# Enhanced search button
if st.button("🔍 Search for Answers", use_container_width=True):
    if not question.strip():
        st.warning("Please enter a question before searching.")
    else:
        with st.spinner('🔍 Searching across video segments...'):
            sleep(1)  # Simulate processing time

            # Retrieval
            if retrieval_method == "FAISS":
                results = rf.query_faiss_text(question, top_k=top_k)
            elif retrieval_method == "pgvector-IVFFLAT":
                results = rf.query_pgvector(question, method="ivfflat", top_k=top_k)
            elif retrieval_method == "pgvector-HNSW":
                results = rf.query_pgvector(question, method="hnsw", top_k=top_k)
            elif retrieval_method == "TF-IDF":
                results = rf.query_tfidf(question, top_k=top_k)
            elif retrieval_method == "BM25":
                results = rf.query_bm25(question, top_k=top_k)
            else:
                st.error("Invalid retrieval method selected.")
                results = []

        if results:
            st.markdown(f"""
                <div class="custom-success">
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <div>✅ Found {len(results)} relevant segments!</div>
                        <div style="font-size: 0.9rem;">Method: {retrieval_method} | Top-{top_k}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            for idx, res in enumerate(results):
                with st.container():
                    st.markdown(f"<div class='card'>", unsafe_allow_html=True)
                    
                    # Header with result number and score
                    st.markdown(f"""
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <h3 style="margin: 0; color: #6a11cb;">📄 Segment {idx + 1}</h3>
                            <span style="background: rgba(37, 117, 252, 0.1); color: #2575fc; 
                                padding: 0.25rem 0.75rem; border-radius: 12px; font-weight: 500;">
                                Score: {res.get('score', 0):.2f}
                            </span>
                        </div>
                    """, unsafe_allow_html=True)

                    # Timestamp with improved styling
                    st.markdown(f"""
                        <p style="margin: 0.5rem 0;">
                            <strong>🕒 Timestamp:</strong> 
                            <a href="https://youtu.be/dARr3lGKwk8?t={int(res['timestamp'])}" 
                               class="timestamp-link" target="_blank">
                                {res['timestamp']:.2f} seconds
                            </a>
                        </p>
                    """, unsafe_allow_html=True)

                    # Highlighted text with improved formatting
                    highlighted_text = res['text']
                    for word in question.lower().split():
                        highlighted_text = re.sub(
                            f"\\b({word})\\b",
                            r"<mark>\1</mark>",
                            highlighted_text,
                            flags=re.IGNORECASE
                        )

                    st.markdown("""
                        <div style="margin: 1rem 0;">
                            <strong style="color: #6a11cb;">📝 Relevant Content:</strong>
                            <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin-top: 0.5rem;">
                                {text}
                            </div>
                        </div>
                    """.format(text=highlighted_text), unsafe_allow_html=True)

                    # Embedded video at timestamp
                    video_embed = f"""
                    <div style="margin: 1rem 0;">
                        <div class="video-responsive">
                            <iframe 
                                src="https://www.youtube.com/embed/dARr3lGKwk8?start={int(res['timestamp'])}&autoplay=0&modestbranding=1&rel=0" 
                                frameborder="0" 
                                allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                                allowfullscreen>
                            </iframe>
                        </div>
                    </div>
                    """
                    components.html(video_embed, height=300)

                    # Enhanced explanation expander
                    with st.expander("🔍 Why is this relevant?", expanded=False):
                        matched_keywords = ', '.join(set(word.lower() for word in question.split() 
                                                      if word.lower() in res['text'].lower()))
                        
                        st.markdown(f"""
                            <div style="background: rgba(106, 17, 203, 0.05); padding: 1rem; border-radius: 8px;">
                                <p><strong>Matched Keywords:</strong> {matched_keywords or "None directly matched"}</p>
                                <p style="margin-top: 0.5rem;">This segment was retrieved because it contains content 
                                semantically related to your question. The {retrieval_method} method identified it as 
                                one of the most relevant portions of the video.</p>
                            </div>
                        """, unsafe_allow_html=True)

                    st.markdown("</div>", unsafe_allow_html=True)

        else:
            st.warning("""
                <div style="background: rgba(255, 193, 7, 0.1); padding: 1rem; border-radius: 8px;">
                    <p style="margin: 0; color: #ff9800;">⚠️ No matching results found. Try:</p>
                    <ul style="margin: 0.5rem 0 0 1rem; padding-left: 1rem;">
                        <li>Using different keywords</li>
                        <li>Making your question more specific</li>
                        <li>Trying another retrieval method</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)