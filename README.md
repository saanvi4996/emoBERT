# 🧠 EmoBERT — Emotion-Aware Social Media Analyzer

**Name:** Saanvi Chauhan  
**Accuracy:** 93.5% | **F1 Macro:** 89%

---

## What it does
EmoBERT detects 6 emotions from social media text using BERT
fine-tuned on the `dair-ai/emotion` dataset.

Emotions: sadness 😢 · joy 😄 · love ❤️ · anger 😡 · fear 😨 · surprise 😲

---

## Tech Stack
- BERT (`bert-base-uncased`) via HuggingFace Transformers
- PyTorch · Streamlit · Plotly
- Dataset: `dair-ai/emotion` (16k tweets, 6 emotions)
- Training: Google Colab T4 GPU

---

## Project Structure
emoBERT/
├── app.py            → Streamlit app
├── requirements.txt  → Dependencies
└── emobert_model/    → Fine-tuned BERT model

---

## Run locally
pip install -r requirements.txt
streamlit run app.py
