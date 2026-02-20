from preprocess import load_and_preprocess
from model import build_model

DATA_PATH = "data/IMDB Dataset.csv"

X_train, X_test, Y_train, Y_test, tokenizer = load_and_preprocess(DATA_PATH)

model = build_model()

model.fit(
    X_train,
    Y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2
)

loss, accuracy = model.evaluate(X_test, Y_test)

print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

model.save("model.h5")
