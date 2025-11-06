import streamlit as st 
from PIL import Image
import numpy as np
from ISR.models import RRDN
from io import BytesIO

# --- Model Configuration ---
# List of available weights for the RRDN model in the ISR library
# 'rrdn-gans' is typically the best and sharpest for perceptual quality (ESRGAN-like)
# 'gans' is perceptually optimized but sometimes less sharp than rrdn-gans
# 'psnr-large' is typically smoother/blurrier but has the best technical accuracy (PSNR)
WEIGHT_OPTIONS = {
    'RRDN (Sharp/Best)': 'rrdn-gans',
    'RRDN (Perceptual)': 'gans',
    'RRDN (Smoother/PSNR)': 'psnr-large',
}

# Predict function for image upscaling
def predict(img):
    """Predict function for image upscaling"""
    lr_img = np.array(img)
    model = st.session_state.model
    # The ISR library handles the model input and output resolution automatically
    sr_img = model.predict(np.array(lr_img))
    return Image.fromarray(sr_img)

# Page configuration
st.set_page_config(
    page_title="Super Resolution GAN",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #6c757d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .upload-container {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f8f9fa;
        margin: 1rem 0;
    }
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .stats-container {
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
    }
    .stat-item {
        text-align: center;
        padding: 0.5rem;
    }
    .stat-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #667eea;
    }
    .stat-label {
        font-size: 0.9rem;
        color: #6c757d;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.1rem;
        padding: 0.75rem;
        border-radius: 8px;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
# --- FIX: Changing the default to 'gans' as 'rrdn-gans' seems unavailable on your system ---
if 'model_name' not in st.session_state:
    st.session_state.model_name = WEIGHT_OPTIONS['RRDN (Perceptual)'] 
# --- FIX END ---
if 'upscaled_image' not in st.session_state:
    st.session_state.upscaled_image = None
if 'original_image' not in st.session_state:
    st.session_state.original_image = None

# Header
st.markdown('<h1 class="main-header">🔍 Super Resolution GAN</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Transform low-resolution images into high-quality super-resolution images using AI</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("---")
    
    st.subheader("Model Weights Selector")
    
    # Selector for model weights
    selected_key = st.selectbox(
        "Choose Upscaling Quality/Model:",
        options=list(WEIGHT_OPTIONS.keys()),
        # --- FIX: Setting the default index to 1 (RRDN Perceptual / 'gans' weights)
        index=1, 
        help="Sharpness: RRDN (Sharp/Best) > RRDN (Perceptual) > RRDN (Smoother/PSNR)"
    )
    
    # Update the model if the selection changes
    new_model_name = WEIGHT_OPTIONS[selected_key]
    if new_model_name != st.session_state.model_name:
        st.session_state.model = None # Reset model to force reload
        st.session_state.model_name = new_model_name
        st.info(f"Model selection changed to: **{selected_key}**. Click 'Upscale Now' to reload.")

    st.markdown("---")
    st.subheader("Model Information")
    st.info(f"""
    **Architecture:** RRDN (ESRGAN Core)
    
    **Active Weights:** `{st.session_state.model_name}`
    
    **Upscaling Factor:** 4x (Fixed by model)
    """)
    
    st.markdown("---")
    st.subheader("📋 Instructions")
    st.markdown("""
    1. **Upload** a low-resolution image
    2. Select the desired **Quality** in the selector above
    3. Click **Upscale Now** button
    4. View the enhanced result
    """)

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload Image")
    st.markdown('<div class="upload-container">', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose an image to upscale", 
        type=("jpg", "png", "jpeg"),
        help="Upload a low-resolution image (JPG, PNG, or JPEG)"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Store original image
        st.session_state.original_image = Image.open(uploaded_file).convert("RGB")
        original = st.session_state.original_image
        
        # Display uploaded image with stats
        st.subheader("📷 Original Image")
        st.image(original, caption='Original Low-Resolution Image', use_column_width=True)
        
        # Image statistics
        st.markdown('<div class="stats-container">', unsafe_allow_html=True)
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        
        with col_stat1:
            st.markdown('<div class="stat-item"><div class="stat-value">{}</div><div class="stat-label">Width (px)</div></div>'.format(original.size[0]), unsafe_allow_html=True)
        with col_stat2:
            st.markdown('<div class="stat-item"><div class="stat-value">{}</div><div class="stat-label">Height (px)</div></div>'.format(original.size[1]), unsafe_allow_html=True)
        with col_stat3:
            st.markdown('<div class="stat-item"><div class="stat-value">{}</div><div class="stat-label">Mode</div></div>'.format(original.mode), unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Upscale button
        st.markdown("---")
        upscale_button = st.button("🚀 Upscale Now", type="primary", use_container_width=True)
        
        if upscale_button:
            with st.spinner(f"🔄 Processing image with {selected_key} weights... This may take a moment."):
                try:
                    # Load model if not already loaded, or if model selection changed
                    if st.session_state.model is None or st.session_state.model.name != st.session_state.model_name:
                        with st.spinner(f"📥 Loading AI model: {st.session_state.model_name}... (Downloading weights if necessary)"):
                            # This is the key change: dynamically loading the selected weights
                            st.session_state.model = RRDN(weights=st.session_state.model_name)
                    
                    # Predict
                    pred = predict(st.session_state.original_image)
                    st.session_state.upscaled_image = pred
                    st.success("✅ Image upscaled successfully! Check the Enhanced Image panel.")
                    
                except Exception as e:
                    # Added the available weights to the error message for clarity
                    st.error(f"❌ Error during upscaling. Check console for detailed error. If the error is related to weights, try selecting a different model. Error: Available RRDN network weights: {RRDN.available_weights()}")
                    st.session_state.upscaled_image = None
                
with col2:
    st.header("✨ Enhanced Image")
    
    if st.session_state.upscaled_image is not None:
        upscaled = st.session_state.upscaled_image
        
        # Display upscaled image
        st.image(upscaled, caption=f'Super Resolution Enhanced Image ({selected_key})', use_column_width=True)
        
        # Enhanced image statistics
        st.markdown('<div class="stats-container">', unsafe_allow_html=True)
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        
        with col_stat1:
            if st.session_state.original_image is not None:
                upscale_ratio = upscaled.size[0] / st.session_state.original_image.size[0]
            else:
                upscale_ratio = 4.0 # Default for RRDN
            st.markdown('<div class="stat-item"><div class="stat-value">{}</div><div class="stat-label">Width (px)</div></div>'.format(upscaled.size[0]), unsafe_allow_html=True)
        with col_stat2:
            st.markdown('<div class="stat-item"><div class="stat-value">{}</div><div class="stat-label">Height (px)</div></div>'.format(upscaled.size[1]), unsafe_allow_html=True)
        with col_stat3:
            st.markdown('<div class="stat-item"><div class="stat-value">{:.1f}x</div><div class="stat-label">Upscale Ratio</div></div>'.format(upscale_ratio), unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Download button
        st.markdown("---")
        buf = BytesIO()
        upscaled.save(buf, format="PNG")
        buf.seek(0)
        st.download_button(
            label="📥 Download Enhanced Image",
            data=buf,
            file_name="upscaled_image.png",
            mime="image/png",
            use_container_width=True
        )
        
    else:
        st.info("👈 Upload an image and click 'Upscale Now' to see the enhanced version here")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6c757d; padding: 1rem;'>
    <p>Powered by Super Resolution GAN (RRDN/ESRGAN Architecture) | Transform your images with AI</p>
</div>
""", unsafe_allow_html=True)
