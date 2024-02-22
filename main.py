from flask import Flask, render_template, request, redirect
app = Flask(__name__)
from tensorflow.keras.models import load_model
import tensorflow as tf
from pathlib import Path

import numpy as np
max_length = 50
vocab_size = 1000


def setVeclayer():

    url = "https://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip"
    path = tf.keras.utils.get_file("spa-eng.zip",origin=url,cache_dir="datasets",extract=True)
    text = (Path(path).with_name('spa-eng')/'spa.txt').read_text()
    text = text.replace("¡", "").replace("¿", "")
    pairs = [line.split("\t") for line in text.splitlines()]
    np.random.shuffle(pairs) 
    sentences_en, sentences_es = zip(*pairs)
    text_vec_layer_en = tf.keras.layers.TextVectorization(
     vocab_size,output_sequence_length=max_length
    )
    text_vec_layer_es = tf.keras.layers.TextVectorization(
         vocab_size, output_sequence_length = max_length
    )
    text_vec_layer_en.adapt(sentences_en)
    text_vec_layer_es.adapt([f"startofseq {s} endofseq" for s in sentences_es])
    return text_vec_layer_es.get_vocabulary()
model = load_model("eng_to_germ")
outputVocab = setVeclayer()
def translation(sentence_en,model):
    translation = ''
    for word_idx in range(max_length):
        X = np.array([sentence_en]) # encoder input
        X_dec = np.array(['startofseq'+translation])# decoder input
        y_proba = model.predict((X,X_dec))[0,word_idx] # last token's probas
        predicted_word_id = np.argmax(y_proba)
        predicted_word = outputVocab.get_vocabulary()[predicted_word_id]
        if predicted_word == 'endofseq':
            break
        translation += ' '+predicted_word
    return translation.strip()

@app.route("/")
def home():
    return render_template("index.html",translated_text="")

@app.route("/translate", methods=["POST"])
def translate():
    text = str(request.form["text"])
    translated_text = translation(text,model)
    return render_template("index.html", translated_text=translated_text)



if __name__ == "__main__":
    app.run(debug=True)
