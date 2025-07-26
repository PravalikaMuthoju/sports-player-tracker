

# 🏃‍♂️ Player Re-Identification in Sports Footage

This project is developed as part of the AI Internship assignment by **Liat AI**. The objective is to implement a **Player Re-Identification system** in sports videos using computer vision. The same player should retain their identity even if they move across different camera views or leave and re-enter the frame.

## 🚀 Task Options

* Re-identification within a Single Camera Feed`
- Track players throughout a 15-second sports video.
- Ensure that players who leave and re-enter the frame keep the same ID.
- Implemented with a Hugging Face DETR model (Transformer-based detection).
- Built using Streamlit for an interactive frontend.
  
  ## Hugging face Deployment : https://huggingface.co/spaces/Pravalikamuthoju2005/player-reid-liat-ai

## 🛠️ Tech Stack

- `Python`
- `Streamlit`
- `Hugging Face Transformers (DETR)`
- `PyTorch`
- `OpenCV`

## 🧠 Model Used

- `facebook/detr-resnet-50` — a pretrained Transformer-based object detection model.
- Only detects "person" class from the labels.

## 📂 Project Structure

├── app.py # Streamlit frontend and processing logic
├── requirements.txt # Python dependencies
├── Dockerfile # For Hugging Face Space deployment (Docker SDK)
├── README.md # This file



## ▶️ How to Run Locally

1. **Clone the repository**
git clone https://github.com/yourusername/player-reid-liatai.git
cd player-reid-liatai
Create a virtual environment


python -m venv venv
# Activate it
source venv/bin/activate          # On Windows: venv\Scripts\activate
Install dependencies


pip install -r requirements.txt
Run the app

streamlit run task1.py
