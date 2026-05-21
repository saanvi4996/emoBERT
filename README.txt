# Sentiment Analysis Using LSTM

Predicts whether a tweet is positive or negative.  
Try it live → https://saanvi4996-sentiment-analysis.hf.space

## Stack
Python, TensorFlow/Keras, Gradio, NLTK, NumPy, Pandas, Scikit-learn

## Dataset
[Sentiment140](http://help.sentiment140.com/) — 1.6M labeled tweets. Achieves ~88% validation accuracy.

## Files
- `app.py` — Gradio app  
- `project.ipynb` — training notebook  
- `sentiment_model.h5` — saved model  
- `tokenizer.pkl` — saved tokenizer  
- `requirements.txt` — dependencies  

## Run locally
```bash
git clone https://github.com/saanvi4996/Sentiment-Analysis-LSTM.git
cd Sentiment-Analysis-LSTM
pip install -r requirements.txt
python app.py
```
