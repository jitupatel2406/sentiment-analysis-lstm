from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocess import load_and_preprocess

DATA_PATH = "data/IMDB Dataset.csv"

X_train, X_test, Y_train, Y_test, tokenizer = load_and_preprocess(DATA_PATH)

model = load_model("model.h5")

def predict_sentiment(review):
    sequence = tokenizer.texts_to_sequences([review])
    padded_sequence = pad_sequences(sequence, maxlen=200)
    prediction = model.predict(padded_sequence)
    return "positive" if prediction[0][0] > 0.5 else "negative"

print(predict_sentiment("This movie was amazing"))
