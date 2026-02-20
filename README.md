# Sentiment Analysis using LSTM

## Overview

This project implements binary sentiment classification (Positive / Negative) using a Long Short-Term Memory (LSTM) neural network built with TensorFlow and Keras.

The model is trained on the IMDB movie review dataset and learns sequential dependencies in text to predict sentiment.

---

## Project Structure

sentiment-analysis-lstm/
│
├── data/
│   └── IMDB Dataset.csv
│
├── src/
│   ├── preprocess.py
│   ├── model.py
│   ├── train.py
│   └── predict.py
│
├── requirements.txt
└── README.md

---

## Dataset

Dataset: IMDB Movie Reviews  
Total Samples: 50,000  
Classes:
- Positive → 1
- Negative → 0

Preprocessing steps:
- Convert text labels to numeric
- Train-test split (80/20)
- Tokenization (top 5000 words)
- Padding sequences (max length = 200)

---

## Model Architecture

Embedding(input_dim=5000, output_dim=128, input_length=200)
→ LSTM(128 units, dropout=0.2)
→ Dense(1, activation="sigmoid")

Loss Function: Binary Crossentropy  
Optimizer: Adam  
Metric: Accuracy  

---

## Installation

Clone repository:

git clone https://github.com/jitupatel2406/sentiment-analysis-lstm.git
cd sentiment-analysis-lstm

Install dependencies:

pip install -r requirements.txt

---

## Training

Run:

python src/train.py

This will:
- Train the LSTM model
- Evaluate performance on test set
- Save model as model.h5

---

## Prediction

Run:

python src/predict.py

You can modify the input review inside predict.py to test custom sentences.

---

## Example Prediction

Input:
"This movie was fantastic. I loved it."

Output:
positive

---

## Results

Test Accuracy typically ranges between 85%–90% depending on environment and training configuration.

---

## Key Concepts Demonstrated

- Text preprocessing and tokenization
- Sequence padding
- LSTM for sequence modeling
- Binary classification with deep learning
- Modular ML project structure

---

## Author

Jitu Patel
Machine Learning / Deep Learning
