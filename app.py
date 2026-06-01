import streamlit as st
import torch
import plotly.express as px
import pandas as pd

from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
)

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="EmoBERT",
    page_icon="🧠",
    layout="centered",
)

# ---------------------------------------------------
# LABELS + EMOJIS
# ---------------------------------------------------

LABELS = [
    "sadness",
    "joy",
    "love",
    "anger",
    "fear",
    "surprise"
]

EMOJIS = {
    "sadness": "😢",
    "joy": "😄",
    "love": "❤️",
    "anger": "😡",
    "fear": "😨",
    "surprise": "😲",
}

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

MODEL_PATH = "emobert_model"

@st.cache_resource
def load_model():

    tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)

    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

    model.eval()

    return tokenizer, model

tokenizer, model = load_model()

# ---------------------------------------------------
# PREDICTION FUNCTION
# ---------------------------------------------------

def predict_emotion(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(
        outputs.logits,
        dim=-1
    )[0]

    probs = probs.numpy()

    prediction = probs.argmax()

    emotion = LABELS[prediction]

    confidence = float(probs[prediction] * 100)

    scores = {
        LABELS[i]: float(probs[i] * 100)
        for i in range(len(LABELS))
    }

    return emotion, confidence, scores

# ---------------------------------------------------
# UI
# ---------------------------------------------------

st.title("🧠 EmoBERT")
st.subheader("Emotion-Aware Social Media Analyzer")

st.write(
    "Analyze emotions in social media text using "
    "fine-tuned BERT."
)

text = st.text_area(
    "Enter text:",
    height=150,
)

# ---------------------------------------------------
# HISTORY
# ---------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------------------------------
# ANALYZE BUTTON
# ---------------------------------------------------

if st.button("Analyze Emotion"):

    if text.strip() == "":
        st.warning("Please enter text.")
    else:

        emotion, confidence, scores = predict_emotion(text)

        emoji = EMOJIS[emotion]

        st.success(
            f"{emoji} Predicted Emotion: "
            f"{emotion.upper()} "
            f"({confidence:.2f}%)"
        )

        # -------------------------------------------
        # CHART
        # -------------------------------------------

        df = pd.DataFrame({
            "Emotion": list(scores.keys()),
            "Confidence": list(scores.values())
        })

        fig = px.bar(
            df,
            x="Emotion",
            y="Confidence",
            title="Emotion Confidence Scores",
        )

        st.plotly_chart(fig, use_container_width=True)

        # -------------------------------------------
        # SAVE HISTORY
        # -------------------------------------------

        st.session_state.history.append({
            "text": text,
            "emotion": emotion,
            "confidence": round(confidence, 2)
        })

# ---------------------------------------------------
# HISTORY DISPLAY
# ---------------------------------------------------

if st.session_state.history:

    st.subheader("🕘 Analysis History")

    for item in reversed(st.session_state.history):

        st.markdown(
            f"""
            **Text:** {item['text']}

            **Prediction:** {item['emotion']}
            ({item['confidence']}%)
            """
        )

