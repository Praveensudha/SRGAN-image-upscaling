import streamlit as st
from PIL import Image
import numpy as np
from io import BytesIO

# Import the RealESRGAN model (requires 'realesrgan' package installation)
try:
    from realesrgan import RealESRGAN
except ImportError:
    st.error("❌ The 'realesrgan' package is not installed. Please run: pip install realesrgan")
    st.stop()

# --- Model Configuration ---
# Use the best available model for general photo enhancement
MODEL_NAME = 'RealESRGAN_x4plus' 
DEFAULT_SCALE = 4
DEVICE = 'cpu' # Use 'cuda' if you have an NVIDIA GPU set up, otherwise use 'cpu'

# Predict function for image upscaling using RealESRGAN
@st.cache_resource
def load_model(model_name, device):
    """Loads and caches the RealESRGAN model."""
    try:
        model = RealESRGAN(device, scale=DEFAULT_SCALE)
        model.load_weights(model_name)
        return model
    except Exception as e:
        st.error(f"Error loading RealESRGAN model weights: {e}")
        return None

def predict_realesrgan(img, model):
    """Predict function using RealESRGAN model."""
    # RealESRGAN takes a PIL Image as input
    sr_img = model.predict(img)
    return sr_img

# Page configuration and custom CSS (same as before)
st.set_page_config(
    page_title="Real-ESRGAN Super Resolution",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1fa2ff 0%, #12d8fa 50%, #a6ffcb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #888;
        text-align: center;
        margin-bottom: 2rem;
    }
    .upload-container {
        border: 2px dashed #1fa2ff;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f0f8ff;
        margin: 1rem 0;
    }
    .stat-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1fa2ff;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #1fa2ff 0%, #a6ffcb 100%);
        color: white;
        font-size: 1.1rem;
        padding: 0.75rem;
        border-radius: 8px;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(31, 162, 255, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">✨ Real-ESRGAN Image Super Resolution (SOTA)</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">The best model for photo-realistic, accurate enhancement of real-world low-res images.</p>', unsafe_allow_html=True)

# Initialize session state
if 'realesrgan_model' not in st.session_state:
    st.session_state.realesrgan_model = load_model(MODEL_NAME, DEVICE)
if 'upscaled_image' not in st.session_state:
    st.session_state.upscaled_image = None
if 'original_image' not in st.session_state:
    st.session_state.original_image = None


# Sidebar
with st.sidebar:
    st.header("⚙️ Model Details")
    st.info(f"""
    **Architecture:** Real-ESRGAN (State-of-the-Art)
    
    **Active Weights:** `{MODEL_NAME}`
    
    **Upscaling Factor:** {DEFAULT_SCALE}x
    
    **Device:** {DEVICE.upper()}
    """)
    st.markdown("---")
    st.subheader("⚠️ Quality Note")
    st.markdown("""
    The previous SRGAN model struggled with your low-res image. **Real-ESRGAN is specifically trained to restore detail in faces, noise, and compression artifacts, providing much clearer results.**
    """)
    st.markdown("---")
    st.subheader("📋 Instructions")
    st.markdown("1. Upload your image on the left.\n2. Click **Upscale Now**.\n3. Wait for the high-fidelity result.")


# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Original Image")
    st.markdown('<div class="upload-container">', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose an image to upscale", 
        type=("jpg", "png", "jpeg"),
        help="Upload a low-resolution image (JPG, PNG, or JPEG)"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        st.session_state.original_image = Image.open(uploaded_file).convert("RGB")
        original = st.session_state.original_image
        
        st.image(original, caption='Original Low-Resolution Image', use_column_width=True)
        
        upscale_button = st.button("🚀 Upscale Now", type="primary", use_container_width=True)
        
        if upscale_button and st.session_state.realesrgan_model:
            with st.spinner("🔄 Processing image with Real-ESRGAN... This takes longer but provides superior results."):
                try:
                    pred = predict_realesrgan(original, st.session_state.realesrgan_model)
                    st.session_state.upscaled_image = pred
                    st.success("✅ Image enhanced successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error during Real-ESRGAN processing. Error: {str(e)}")
                    st.session_state.upscaled_image = None
        elif upscale_button and not st.session_state.realesrgan_model:
            st.error("Model failed to load. Please check console for Real-ESRGAN installation errors.")

with col2:
    st.header(f"✨ Enhanced Image ({DEFAULT_SCALE}x)")
    
    if st.session_state.upscaled_image is not None:
        upscaled = st.session_state.upscaled_image
        
        st.image(upscaled, caption='Real-ESRGAN Super-Resolved Image', use_column_width=True)
        
        # Enhanced image statistics
        st.markdown(f"""
        <div style='text-align: center; margin-top: 1rem;'>
            <span class="stat-value">{upscaled.size[0]} x {upscaled.size[1]}</span> 
            <span class="stat-label">Output Resolution</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Download button
        buf = BytesIO()
        upscaled.save(buf, format="PNG")
        buf.seek(0)
        st.download_button(
            label="📥 Download Enhanced Image",
            data=buf,
            file_name="realesrgan_upscaled.png",
            mime="image/png",
            use_container_width=True
        )
        
    else:
        st.info("👈 Upload an image and click 'Upscale Now' for the high-fidelity enhancement.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6c757d; padding: 1rem;'>
    <p>Using Real-ESRGAN for state-of-the-art photo-realistic upscaling.</p>
</div>
""", unsafe_allow_html=True)
