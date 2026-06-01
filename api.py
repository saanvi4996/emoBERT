# api.py — EmoBERT FastAPI Backend
# Run with: uvicorn api:app --reload
# API available at: http://localhost:8000

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
from typing import Dict

# ── Config ─────────────────────────────────────────────────────────────────
MODEL_PATH = "./emobert_model"
MAX_LEN    = 128

# ── Load model once at startup ─────────────────────────────────────────────
print("Loading EmoBERT model...")
tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
model     = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()
print("Model loaded successfully.")

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="EmoBERT API",
    description="Emotion detection API powered by BERT",
    version="1.0.0"
)

# Allow Streamlit frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response schemas ─────────────────────────────────────────────
class TextInput(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    emotion: str
    confidence: float
    scores: dict[str, float]
    emoji: str

# ── Emotion metadata ───────────────────────────────────────────────────────
EMOJI_MAP = {
    "sadness":  "😢",
    "joy":      "😄",
    "love":     "❤️",
    "anger":    "😠",
    "fear":     "😨",
    "surprise": "😲",
}

# ── Routes ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "EmoBERT API is running", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "healthy", "model": MODEL_PATH}

@app.post("/predict", response_model=PredictionResponse)
def predict(input: TextInput):
    if not input.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if len(input.text) > 1000:
        raise HTTPException(status_code=400, detail="Text too long, max 1000 characters")

    # Tokenize
    inputs = tokenizer(
        input.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
    )

    # Predict
    with torch.no_grad():
        logits = model(**inputs).logits

    probs  = torch.softmax(logits, dim=-1).squeeze().tolist()
    labels = list(model.config.id2label.values())
    scores = {label: round(prob * 100, 1) for label, prob in zip(labels, probs)}

    top_emotion = max(scores, key=scores.get)

    return PredictionResponse(
        emotion    = top_emotion,
        confidence = scores[top_emotion],
        scores     = scores,
        emoji      = EMOJI_MAP.get(top_emotion, "🤔"),
    )
