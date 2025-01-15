from flask import Flask, request, jsonify
import tensorflow as tf
import pickle

# Inicializa o aplicativo Flask
app = Flask(__name__)

# Função para carregar recursos do modelo
def carregar_recursos():
    # Carregando o modelo TFLite
    interpreter = tf.lite.Interpreter(model_path="files/sentiment_model.tflite")
    interpreter.allocate_tensors()

    # Carregando tokenizer e label encoder
    with open("files/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("files/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    return interpreter, tokenizer, label_encoder

# Carrega os recursos uma vez no início
interpreter, tokenizer, label_encoder = carregar_recursos()

# Rota para a API
@app.route("/")
def main():
    return "<p>Hello</p>"

