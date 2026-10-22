# EmoBERT — Emotion-Aware Social Media Analyzer

### Saanvi Chauhan

A Streamlit app that classifies the emotion in a short English text (sadness, joy, love, anger, fear, surprise) using `bert-base-uncased` fine-tuned on the `dair-ai/emotion` dataset.

**Test set (2,000 held-out samples):** accuracy 93.6% · macro F1 0.89

## Features

- Predicts one of six emotions with a confidence score
- Bar chart of the probability for every emotion
- History of the texts analysed in the current session
- The fine-tuned weights are hosted on the Hugging Face Hub ([`saanvi4996/emoBERT`](https://huggingface.co/saanvi4996/emoBERT)) and downloaded on first run, so the GitHub repo stays small

## Tech stack

- BERT (`bert-base-uncased`) via Hugging Face Transformers
- PyTorch
- Streamlit and Plotly
- scikit-learn (evaluation), Hugging Face Datasets
- Trained on Google Colab (T4 GPU)

## Dataset

[`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion): English Twitter messages labelled with six emotions. Split: 16,000 train, 2,000 validation, 2,000 test.

## Training

`train_model.ipynb` (Colab notebook):

- Tokenisation: `BertTokenizerFast`, max length 128
- 3 epochs, learning rate 2e-5, batch size 16, weight decay 0.01, warmup ratio 0.1, fp16, seed 42
- Model selection on the validation split by macro F1 (`load_best_model_at_end=True`)
- Final evaluation on the untouched test split

## Results (test split, from the notebook output)

| Emotion | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| sadness | 0.97 | 0.97 | 0.97 | 581 |
| joy | 0.95 | 0.96 | 0.96 | 695 |
| love | 0.87 | 0.83 | 0.85 | 159 |
| anger | 0.94 | 0.92 | 0.93 | 275 |
| fear | 0.87 | 0.94 | 0.90 | 224 |
| surprise | 0.81 | 0.67 | 0.73 | 66 |
| **macro avg** | 0.90 | 0.88 | **0.89** | 2000 |

Accuracy: 0.9355.

The dataset is imbalanced, so accuracy is higher than macro F1. **Surprise** (3% of test samples) and **love** are the weakest classes. In manual testing, surprise sentences are often predicted as fear. Class weighting or oversampling would be the next thing to try.

## Project structure

```
emoBERT/
├── app.py              # Streamlit application (loads the model from the Hub)
├── train_model.ipynb   # fine-tuning and evaluation notebook (Colab)
├── tests/test_app.py   # headless app tests with the real model
├── requirements.txt
└── README.md
```

## Run locally

```bash
git clone https://github.com/saanvi4996/emoBERT.git
cd emoBERT
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads about 440 MB of model weights from the Hugging Face Hub. To use a local copy instead, set `EMOBERT_MODEL` to the folder that contains `config.json` and `model.safetensors`.

`requirements.txt` installs the CPU-only PyTorch build. If you have a CUDA GPU and already have PyTorch installed, it is reused.

### Tests

```bash
pip install pytest
pytest
```

The tests load the real model and check predictions for a few clear examples, so they need the weights (downloaded or local via `EMOBERT_MODEL`).

## Deploy (Streamlit Community Cloud)

1. At [share.streamlit.io](https://share.streamlit.io) choose **New app**, select this repository and set the main file to `app.py`.
2. No secrets are needed because the model repository is public. The first start takes a few minutes while the weights download.
