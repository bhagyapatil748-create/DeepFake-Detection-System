# 🎭 DeepFake Detection System using EfficientNet-B0, FaceNet & LSTM

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=for-the-badge&logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?style=for-the-badge&logo=streamlit)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)

</p>

---

# 📌 Project Overview

The **DeepFake Detection System** is an AI-powered application capable of detecting manipulated **images** and **videos** using Deep Learning and Computer Vision.

The system combines:

- 🖼 EfficientNet-B0 for Image Classification
- 👤 FaceNet for Face Detection & Feature Extraction
- 🎥 LSTM for Temporal Video Analysis
- 🌐 Streamlit for Interactive Web Deployment

The application predicts whether uploaded media is **Real** or **Fake** while displaying confidence scores and inference time.

---

# 🚀 Features

✅ Image DeepFake Detection

✅ Video DeepFake Detection

✅ EfficientNet-B0 Image Classifier

✅ FaceNet Face Extraction

✅ LSTM Temporal Sequence Analysis

✅ Confidence Percentage

✅ Processing Time Display

✅ Interactive Streamlit Interface

✅ Upload Images & Videos

✅ Prediction History

✅ CSV Export

✅ Responsive UI

---

# 🧠 Model Architecture

## Image Detection Pipeline

```
Image
   │
   ▼
Preprocessing
   │
   ▼
EfficientNet-B0
   │
   ▼
Dropout
   │
   ▼
Fully Connected Layer
   │
   ▼
Softmax
   │
   ▼
Real / Fake
```

---

## Video Detection Pipeline

```
Video
   │
   ▼
Frame Extraction
   │
   ▼
FaceNet
(Face Detection)
   │
   ▼
EfficientNet-B0
(Frame Features)
   │
   ▼
LSTM
(Temporal Learning)
   │
   ▼
Fully Connected Layer
   │
   ▼
Softmax
   │
   ▼
Real / Fake
```

---

# 🛠 Tech Stack

### Programming Language

- Python

### Deep Learning

- PyTorch
- TorchVision
- FaceNet-PyTorch
- EfficientNet-B0
- LSTM

### Computer Vision

- OpenCV
- Pillow
- Albumentations

### Frontend

- Streamlit

### Data Processing

- NumPy
- Pandas

### Visualization

- Plotly
- Matplotlib

---

# 📂 Project Structure

```
DeepFake-Detection-System/

│── app.py
│── predict.py
│── utils.py
│── requirements.txt
│── README.md
│── .gitignore

│
├── models/
│   ├── best_image_model.pth
│   └── best_video_model.pth
│
├── assets/
│
├── history/
│
├── reports/
│
└── screenshots/
```

---

# ⚙ Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/DeepFake-Detection-System.git
```

Move into the project folder

```bash
cd DeepFake-Detection-System
```

Create virtual environment

```bash
python -m venv venv
```

Activate virtual environment

Windows

```bash
venv\Scripts\activate
```

Linux / Mac

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶ Running the Application

```bash
streamlit run app.py
```

Open

```
http://localhost:8501
```

---

# 📸 Screenshots

## Home Page

(Add Screenshot Here)

---

## Image Detection

(Add Screenshot Here)

---

## Video Detection

(Add Screenshot Here)

---

## Prediction Result

(Add Screenshot Here)

---

# 📊 Model Workflow

```
User Upload
      │
      ▼
Media Validation
      │
      ▼
Preprocessing
      │
      ▼
AI Model
      │
      ▼
Prediction
      │
      ▼
Confidence Score
      │
      ▼
History Storage
```

---

# 🎯 Future Improvements

- Live Webcam Detection
- Batch Image Detection
- Batch Video Detection
- Explainable AI (Grad-CAM)
- Cloud Deployment
- REST API using FastAPI
- Mobile Application
- Real-Time Monitoring

---

# 📈 Performance

The project evaluates the model using:

- Accuracy
- Precision
- Recall
- F1 Score
- CrossEntropy Loss
- Confidence Score

---

# 💡 Applications

- Social Media Verification
- News Media Authentication
- Digital Forensics
- Cyber Security
- Identity Verification
- Fake News Detection

---

# 👨‍💻 Author

**Nitish Sanjay Tiwari**

AI & Software Developer

Python Developer

Generative AI Enthusiast

---

# 📜 License

This project is licensed under the MIT License.

---

# ⭐ Support

If you like this project,

⭐ Star this repository

🍴 Fork it

📢 Share it

---
