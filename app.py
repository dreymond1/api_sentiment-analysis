from flask import Flask, request, jsonify
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import logging

# Inicializa o aplicativo Flask
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

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

def prever_sentimento(texto):
    try:
        # Tokenização e padding
        sequencia = tokenizer.texts_to_sequences([texto])
        sequencia_padded = pad_sequences(sequencia, maxlen=50, padding='post')

        # Preparação para o modelo TFLite
        sequencia_array = np.array(sequencia_padded, dtype=np.float32)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Inferência com TFLite
        interpreter.set_tensor(input_details[0]['index'], sequencia_array)
        interpreter.invoke()
        predicoes = interpreter.get_tensor(output_details[0]['index'])

        # Decodificação do resultado
        classe_predita = np.argmax(predicoes, axis=1)
        sentimento = label_encoder.inverse_transform(classe_predita)
        return sentimento[0]
    except Exception as e:
        raise ValueError(f"Erro ao prever sentimento: {str(e)}")

# Rota para a API
@app.route("/", methods=["GET"])
def main():
    try:
        # Carrega os recursos uma vez no início
        interpreter, tokenizer, label_encoder = carregar_recursos()

        texto = request.json.get("texto", "")

        if not texto:
            logging.error("Texto não fornecido")
            return jsonify({"error": "Texto não fornecido"}), 400

        #sentimento = prever_sentimento(texto)
        #return jsonify({"sentimento": sentimento})
    #except Exception as e:
        #return jsonify({"error": f"Erro interno: {str(e)}"}), 500
    return jsonify({"status": "Modelo carregado com sucesso!"})


