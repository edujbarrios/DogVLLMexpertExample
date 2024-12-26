import streamlit as st
from utils import process_image, DogAnalyzer, generate_response
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="DogVLLMexpert - Advanced Canine Analysis",
    page_icon="🐕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styles with dark mode
st.markdown("""
    <style>
    /* Dark mode styles */
    .stApp {
        background-color: #1E1E1E;
        color: #E0E0E0;
    }
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 8px;
        background-color: #2D2D2D;
    }
    .upload-text {
        font-size: 1.2rem;
        margin-bottom: 1rem;
        color: #64B5F6;
    }
    .expert-header {
        color: #90CAF9;
        font-size: 2.8rem;
        font-weight: bold;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .expert-subheader {
        color: #64B5F6;
        font-size: 1.8rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    .feature-box {
        background-color: #2D2D2D;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border-left: 4px solid #64B5F6;
        color: #E0E0E0;
    }
    .analysis-title {
        color: #90CAF9;
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    /* Override Streamlit's default light mode */
    .stTextInput > div > div {
        background-color: #3D3D3D;
        color: #E0E0E0;
    }
    .stButton > button {
        background-color: #64B5F6;
        color: #1E1E1E;
    }
    .stButton > button:hover {
        background-color: #90CAF9;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<p class="expert-header">🐕 DogVLLMexpert</p>', unsafe_allow_html=True)
st.markdown("""
    <p class="expert-subheader">Advanced Canine Analysis System with Artificial Intelligence</p>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="feature-box">
    <p class="analysis-title">State-of-the-Art Canine Analysis Technology</p>

    Our system integrates multiple AI technologies to provide comprehensive analysis:

    - 🔍 **Advanced Visual Analysis**: Using cutting-edge computer vision models
    - 🧬 **Precise Breed Identification**: Detection of specific characteristics and variations
    - 📊 **Detailed Morphological Analysis**: Evaluation of proportions and anatomical structures
    - 🎨 **Pattern Analysis**: Detailed study of coat coloration and textures
    - 📋 **Behavioral Assessment**: Interpretation of postures and body language
    - 🌍 **Multilingual Support**: Analysis and responses in multiple languages
    </div>

    <div class="feature-box">
    <p class="analysis-title">Analytical Capabilities</p>

    The system can perform detailed analysis of:
    - Specific breed characteristics and lineage
    - Coloration patterns and distinctive markings
    - Body structure and anatomical proportions
    - Visible health indicators
    - Age estimation and developmental stage
    - Breed standard compatibility
    </div>
""", unsafe_allow_html=True)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

# Initialize analyzer
if 'analyzer' not in st.session_state:
    try:
        with st.spinner('Initializing expert analysis system... Please wait.'):
            st.session_state.analyzer = DogAnalyzer()
        st.success('Expert analysis system initialized successfully!')
    except Exception as e:
        st.error(f"Error initializing analyzer: {str(e)}")
        logger.error(f"Initialization error: {str(e)}")
        st.stop()

# Image upload interface
st.markdown('<p class="upload-text">📤 Upload a dog image for expert analysis:</p>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=['png', 'jpg', 'jpeg', 'webp'])

# Image processing
if uploaded_file:
    try:
        image, image_bytes = process_image(uploaded_file)
        if image and image_bytes:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(image, caption="Image for analysis", use_container_width=True)

            with col2:
                st.markdown("""
                    <div class="feature-box">
                    <p class="analysis-title">💡 Tips for Better Results</p>

                    - Ensure good lighting
                    - Capture the entire dog
                    - Avoid extreme angles
                    - Prefer neutral background
                    </div>
                """, unsafe_allow_html=True)

            # Question form
            with st.form("question_form"):
                question = st.text_input("💭 Ask our expert system about the image:", 
                                    placeholder="Example: Can you provide a detailed analysis of this dog's breed characteristics?")
                submitted = st.form_submit_button("Analyze Image")

                if submitted and question:
                    with st.spinner('Performing expert analysis... 🔍'):
                        try:
                            response = generate_response(
                                st.session_state.analyzer,
                                image,
                                question
                            )

                            # Add to history
                            st.session_state.history.append({
                                "question": question,
                                "response": response
                            })

                        except Exception as e:
                            st.error(f"Error generating expert response: {str(e)}")
                            logger.error(f"Error in generate_response: {str(e)}")
        else:
            st.error("Error: Could not process the image. Please ensure it's a valid image and not too large.")

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        logger.error(f"Error processing image: {str(e)}")

# Display history
if st.session_state.history:
    st.markdown('<p class="analysis-title">📝 Analysis History</p>', unsafe_allow_html=True)
    for i, qa in enumerate(st.session_state.history):
        with st.expander(f"📌 Analysis {i+1}: {qa['question']}", 
                        expanded=i==len(st.session_state.history)-1):
            st.markdown(f"""
                <div class="feature-box">
                <p><strong>🤖 Expert Analysis:</strong></p>
                {qa['response']}
                </div>
            """, unsafe_allow_html=True)

# Additional information in sidebar
with st.sidebar:
    st.markdown("""
    <div class="feature-box">
    <p class="analysis-title">💡 Analysis Suggestions:</p>

    1. **Breed Analysis**
    - What are the main breed characteristics of this dog?
    - What distinctive traits does it present?

    2. **Physical Analysis**
    - Could you analyze the coat pattern and texture?
    - How would you assess the size and proportions?

    3. **Behavioral Analysis**
    - What does the dog's posture indicate?
    - What body language signals do you observe?

    4. **Detailed Analysis**
    - What are the most prominent anatomical features?
    - Could you provide a complete analysis of what you observe?
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="feature-box">
    <p class="analysis-title">ℹ️ About DogVLLMexpert</p>

    This advanced system utilizes cutting-edge technology in:
    - 🤖 State-of-the-art Artificial Intelligence
    - 👁️ High-precision Computer Vision
    - 🧠 Advanced Natural Language Processing
    - 📊 Detailed Morphological Analysis

    Designed to provide professional and detailed analysis of canine images,
    identifying subtle characteristics, complex patterns, and providing
    expert observations based on the latest advances in AI.
    </div>
    """, unsafe_allow_html=True)

    # Clear history button
    if st.button("🧹 Clear Analysis History"):
        st.session_state.history = []
        st.success("Analysis history cleared successfully")