# Sentiment Analysis Using LSTM

A web app that predicts whether a tweet is positive or negative, built with an LSTM neural network and deployed on Hugging Face Spaces.

## Author
**Saanvi Chauhan** — B.Tech CSE (IoT & IS)

## Tech Stack
Python, TensorFlow/Keras, Streamlit, NLTK, NumPy, Pandas, Scikit-learn

## Project Files
| File | Description |
|------|-------------|
| `app.py` | Streamlit web app for real-time sentiment prediction |
| `project.ipynb` | Model training, preprocessing, and evaluation |
| `sentiment_model.h5` | Trained LSTM model |
| `tokenizer.pkl` | Saved tokenizer for text preprocessing |
| `requirements.txt` | Python dependencies |

## Dataset
Trained on the [Sentiment140](http://help.sentiment140.com/) dataset — 1.6M labeled tweets.

## Getting Started

```bash
git clone https://github.com/saanvi4996/Sentiment-Analysis-LSTM.git
cd Sentiment-Analysis-LSTM
pip install -r requirements.txt
streamlit run app.py
```

The app will open in your browser. Type any tweet or piece of text and get an instant sentiment prediction.

## Deployment
Live on [Hugging Face Spaces](https://huggingface.co/spaces) — no setup needed to try it out.
