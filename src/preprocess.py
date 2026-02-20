import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_and_preprocess(data_path):
    data = pd.read_csv(data_path)

    data.replace({"sentiment": {"positive": 1, "negative": 0}}, inplace=True)

    train_data, test_data = train_test_split(
        data, test_size=0.2, random_state=42
    )

    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(train_data["review"])

    X_train = pad_sequences(
        tokenizer.texts_to_sequences(train_data["review"]),
        maxlen=200
    )

    X_test = pad_sequences(
        tokenizer.texts_to_sequences(test_data["review"]),
        maxlen=200
    )

    Y_train = train_data["sentiment"]
    Y_test = test_data["sentiment"]

    return X_train, X_test, Y_train, Y_test, tokenizer
