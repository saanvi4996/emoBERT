import gradio as gr
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
import re

nltk.download('stopwords')

# Load model
model = tf.keras.models.load_model("sentiment_model.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

stop_words = set(stopwords.words('english'))

max_len = 100

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

def predict_sentiment(text):
    processed = preprocess_text(text)
    
    sequence = tokenizer.texts_to_sequences([processed])
    padded = pad_sequences(sequence, maxlen=max_len)
    
    prediction = model.predict(padded)[0][0]
    
    if prediction > 0.5:
        return f"Positive Sentiment 😊 ({prediction:.2f})"
    else:
        return f"Negative Sentiment 😔 ({1-prediction:.2f})"

interface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=4, placeholder="Enter a tweet or sentence here..."),
    outputs="text",
    title="Sentiment Analysis using LSTM",
    description="Predict whether a tweet is positive or negative using an LSTM model."
)

interface.launch()
