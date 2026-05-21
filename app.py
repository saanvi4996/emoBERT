import streamlit as st
import tensorflow as tf
import pickle
import re
import nltk

from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download stopwords
nltk.download('stopwords')

# Load stopwords
stop_words = set(stopwords.words('english'))

# Load trained model
model = tf.keras.models.load_model("sentiment_model.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Text preprocessing function
def preprocess_text(text):

    # Remove links
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # Remove special characters
    text = re.sub(r'[^A-Za-z\s]', '', text)

    # Convert to lowercase
    text = text.lower()

    # Remove stopwords
    words = text.split()
    words = [word for word in words if word not in stop_words]

    return " ".join(words)

# Prediction function
def predict_sentiment(text):

    processed_text = preprocess_text(text)

    # Convert text to sequence
    sequence = tokenizer.texts_to_sequences([processed_text])

    # Pad sequence
    padded_sequence = pad_sequences(sequence, maxlen=100)

    # Predict
    prediction = model.predict(padded_sequence)

    score = prediction[0][0]

    if score >= 0.5:
        return "Positive 😊", score
    else:
        return "Negative 😔", score

# Streamlit UI
st.title("Sentiment Analysis using LSTM")

st.write("Enter a sentence and predict its sentiment.")

# User input
user_input = st.text_area("Enter text here")

# Button
if st.button("Analyze Sentiment"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")
    
    else:
        sentiment, confidence = predict_sentiment(user_input)

        st.subheader(f"Prediction: {sentiment}")

        st.write(f"Confidence Score: {confidence:.2f}")