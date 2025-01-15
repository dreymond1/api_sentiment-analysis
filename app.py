import numpy as np
import pickle
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

# Inicializa o aplicativo Flask
app = Flask(__name__)

# Função para carregar recursos do modelo
def carregar_recursos():
    # Caminhos dos arquivos
    modelo_path = "files/sentiment_model.tflite"
    tokenizer_path = "files/tokenizer.pkl"
    label_encoder_path = "files/label_encoder.pkl"

    # Carregando o modelo TFLite
    interpreter = tf.lite.Interpreter(model_path=modelo_path)
    interpreter.allocate_tensors()

    # Carregando tokenizer e label encoder
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    with open(label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    return interpreter, tokenizer, label_encoder

# Carrega os recursos uma vez no início
interpreter, tokenizer, label_encoder = carregar_recursos()

# Função para prever o sentimento
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
@app.route("/predict", methods=["POST"])
def predict():
    dados = request.json
    texto = dados.get("texto", "")
    if not texto:
        return jsonify({"error": "Texto não fornecido"}), 400

    try:
        sentimento = prever_sentimento(texto)
        return jsonify({"sentimento": sentimento})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
