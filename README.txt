# Sentiment Analysis Using LSTM

A web app that predicts whether a tweet is positive or negative, built with a 
lightweight LSTM neural network and deployed on Hugging Face Spaces.

🔗 [Live Demo](https://saanvi4996-sentiment-analysis.hf.space)

## Author
Saanvi Chauhan

## Model
Embedding → LSTM → Dropout → Dense architecture trained on 1.6M tweets.  
Achieves ~88% validation accuracy on the Sentiment140 dataset.

## Tech Stack
Python, TensorFlow/Keras, Gradio, NLTK, NumPy, Pandas, Scikit-learn

## Project Files
`app.py` — Gradio web app for real-time sentiment prediction  
`project.ipynb` — Model training, preprocessing, and evaluation  
`sentiment_model.h5` — Trained LSTM model  
`tokenizer.pkl` — Saved tokenizer for text preprocessing  
`requirements.txt` — Python dependencies  

## Dataset
Trained on the [Sentiment140](http://help.sentiment140.com/) dataset with 1.6M labeled tweets.

## Getting Started
```bash
git clone https://github.com/saanvi4996/Sentiment-Analysis-LSTM.git
cd Sentiment-Analysis-LSTM
pip install -r requirements.txt
python app.py
```

## Deployment
Live on [Hugging Face Spaces](https://saanvi4996-sentiment-analysis.hf.space).
