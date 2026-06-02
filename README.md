# EmoBERT — Emotion-Aware Social Media Analyzer

### Saanvi Chauhan

**Accuracy:** 93.5% · **Macro F1 Score:** 89%

## Overview

EmoBERT is a BERT-based NLP application that detects emotions in social media text using a fine-tuned transformer model trained on the `dair-ai/emotion` dataset.

### Supported Emotions

* Sadness 😢
* Joy 😄
* Love ❤️
* Anger 😡
* Fear 😨
* Surprise 😲

## Tech Stack

* BERT (`bert-base-uncased`)
* Hugging Face Transformers
* PyTorch
* Streamlit
* Plotly
* Google Colab (T4 GPU)

## Features

* Real-time emotion prediction
* Confidence score visualization
* Interactive Streamlit interface
* Fine-tuned transformer-based NLP pipeline
* Public cloud deployment

## Dataset

* `dair-ai/emotion`
* ~16k labeled social media text samples
* 6 emotion classes

## Project Structure

```bash
emoBERT/
├── app.py
├── requirements.txt
└── README.md
```

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
