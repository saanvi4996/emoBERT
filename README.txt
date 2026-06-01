EmoBERT — Emotion Detection via REST API
=========================================
Name: Saanvi Chauhan
Registration No: 23FE10CII00225
Accuracy: 93.5% | F1 Macro: 89%

DESCRIPTION
-----------
EmoBERT detects 6 emotions (sadness, joy, love, anger, fear, surprise)
from social media text using BERT fine-tuned on dair-ai/emotion dataset.
Features a FastAPI REST backend and Streamlit frontend with analysis history.

PROJECT STRUCTURE
-----------------
emoBERT/
  ├── api.py             FastAPI backend — loads model, serves /predict endpoint
  ├── app.py             Streamlit frontend — calls API, shows results + history
  ├── requirements.txt   All dependencies
  ├── README.txt         This file
  └── emobert_model/     Trained BERT model (download from Colab)

API ENDPOINTS
-------------
GET  /          → API status
GET  /health    → Health check
POST /predict   → Predict emotion

Example request:
  POST http://localhost:8000/predict
  {"text": "I just got my dream job!"}

Example response:
  {
    "emotion": "joy",
    "confidence": 97.3,
    "scores": {"sadness": 0.2, "joy": 97.3, "love": 1.1, ...},
    "emoji": "😄"
  }

HOW TO RUN
----------
1. pip install -r requirements.txt

2. Start the API (Terminal 1):
   uvicorn api:app --reload
   → API runs at http://localhost:8000
   → Docs at http://localhost:8000/docs

3. Start the frontend (Terminal 2):
   streamlit run app.py
   → App runs at http://localhost:8501

TECH STACK
----------
- BERT (bert-base-uncased) via HuggingFace Transformers
- FastAPI + Uvicorn (REST API backend)
- Streamlit + Plotly (frontend)
- PyTorch (model inference)
- Dataset: dair-ai/emotion (16k tweets, 6 emotions)
