# 🧠 SRGAN Image Upscaling

This project implements **Super-Resolution Generative Adversarial Network (SRGAN)** — a deep learning model that upscales low-resolution images into sharp, high-resolution outputs using deep learning and adversarial training.  
It includes a **Streamlit web app** for easy real-time image enhancement.

---

## 📸 Overview

The **SRGAN** architecture consists of:
- 🧩 **Generator** — Converts low-resolution images into high-resolution ones.
- 🕵️ **Discriminator** — Distinguishes between real and generated high-resolution images.
- 🎯 **Perceptual Loss** — Ensures the generated images are perceptually similar to true high-res images.

---

## 🚀 Features

✅ Train SRGAN on your own dataset (e.g., DIV2K)  
✅ Test and visualize model performance  
✅ Run a **Streamlit web app** to upscale images in real time  
✅ Supports easy customization and fine-tuning  

---

## 🧩 Folder Structure

Super-Resolution-GAN/
│
├── app.py # Streamlit web app for real-time image upscaling
├── inference.py # Model testing / prediction
├── srgan_training.ipynb # Jupyter notebook for model training
├── requirements.txt # Required dependencies
├── log_file/ # Training logs
├── demo.gif # Demo animation
├── LICENSE
└── README.md

yaml
Copy code

---

## ⚙️ Installation

### 1️⃣ Clone this repository
```bash
git clone https://github.com/Rian8817/SRGAN-image-upscaling.git
cd SRGAN-image-upscaling
2️⃣ Create and activate a virtual environment
bash
Copy code
python -m venv isr_env
isr_env\Scripts\activate   # (Windows)
3️⃣ Install dependencies
bash
Copy code
pip install -r requirements.txt
🧠 Running the Streamlit App
Once setup is complete, run:

bash
Copy code
streamlit run app.py
Then open your browser and go to:
👉 http://localhost:8501/

You can now upload a low-resolution image and view the super-resolved output directly on the web interface.

📊 Dataset
This project can be trained on the DIV2K dataset, which contains high-quality image pairs for super-resolution tasks.
📥 Download here: DIV2K Dataset

🧠 Model Architecture
Component	Description
Generator	Deep residual network with skip connections
Discriminator	CNN that distinguishes real vs generated HR images
Loss Function	Combines perceptual, content, and adversarial losses



