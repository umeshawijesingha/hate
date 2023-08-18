import pickle
import string
import re
from imp import load_module
from flask import Flask, render_template, request
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model('static/model/newhatemodel.h5')


# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
# tokenizer = Tokenizer()


# def preprocess_text(text):
#     text = text.lower()
#     text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
#     text = remove_punctuations(text)
#     text_tokens = text.split()
#     text_sequence = tokenizer.texts_to_sequences([text_tokens])
#     print(text_sequence)
#     padded_sequence = pad_sequences(
#         text_sequence, maxlen=20, padding='post', truncating='post')
#     predictions = model.predict(padded_sequence)
#     prediction = 'Hate ' if predictions[0][0] > 0.5 else 'Non hate'
#     return str(predictions[0][0])


# def remove_punctuations(text):
#     for punctuation in string.punctuation:
#         text = text.replace(punctuation, '')
#     return text


# # Preprocess the text
# text = ""
# preprocessed_text = preprocess_text(text)

# # Tokenize and pad the preprocessed text
# text_sequence = tokenizer.texts_to_sequences([preprocessed_text])
# padded_sequence = pad_sequences(
#     text_sequence, maxlen=20, padding='post', truncating='post')

# predictions = model.predict(padded_sequence)
# prediction = 'Hate ' if predictions[0][0] > 0.5 else 'Non hate'


# def my_function():
#     print("Hello")
#     print("Predicted label:", prediction)
#     print("Confidence:", predictions[0][0])
#     return str(predictions[0][0])

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    return text


def prepare_text(text):
    preprocessed_text = preprocess_text(text)
    text_sequence = tokenizer.texts_to_sequences([preprocessed_text])
    print(text_sequence)
    padded_sequence = pad_sequences(
        text_sequence, maxlen=20, padding='post', truncating='post')
    return padded_sequence


app = Flask(__name__)


# @app.route('/')
# def man():
#     return render_template('home.html')


@app.route('/', methods=['GET', 'POST'])
def index():
    input_data = ''
    result = ''

    if request.method == 'POST':
        input_data = request.form['data']
        if input_data:
            preprocessed_text = preprocess_text(input_data)
            text_sequence = tokenizer.texts_to_sequences([preprocessed_text])
            padded_sequence = pad_sequences(
                text_sequence, maxlen=20, padding='post', truncating='post')
            predictions = model.predict(padded_sequence)
            prediction_label = 'Hate' if predictions[0][0] > 0.5 else 'Non-hate'
            result = f"Predicted label: {prediction_label}, Confidence: {predictions[0][0]:.4f}"

    return render_template('index.html', input_data=input_data, result=result)


if __name__ == "__main__":
    app.run(debug=True)
