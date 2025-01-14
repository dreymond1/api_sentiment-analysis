import os
import numpy as np
import pickle
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import gc

app = Flask(__name__)

# Carregando o modelo TFLite
tflite_model_path = "files/sentiment_model.tflite"
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Obtendo detalhes de entrada e saída do modelo
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Carregando o tokenizer e o label encoder
with open("files/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("files/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Função para prever o sentimento usando TFLite
def prever_sentimento_tflite(texto, interpreter, tokenizer, max_len_contexto=50):
    # Tokenização
    X_novos_comentarios = tokenizer.texts_to_sequences([texto])
    
    # Padding para garantir o mesmo tamanho
    X_novos_comentarios = pad_sequences(X_novos_comentarios, maxlen=max_len_contexto, padding='post')
    
    # Preparando os dados para o modelo TFLite
    X_novos_comentarios = np.array(X_novos_comentarios, dtype=np.float32)
    
    # Configurando os tensores de entrada do TFLite
    interpreter.set_tensor(input_details[0]['index'], X_novos_comentarios)
    
    # Executando a inferência
    interpreter.invoke()
    
    # Obtendo os resultados
    predicoes = interpreter.get_tensor(output_details[0]['index'])
    
    # Decodificando as previsões
    y_pred = np.argmax(predicoes, axis=1)  # Pegando a classe com maior probabilidade
    sentimento_predito = label_encoder.inverse_transform(y_pred)

    return sentimento_predito[0]  # Retorna o primeiro sentimento

@app.route("/", methods=["POST"])
def predict():
    try:
        dados = request.json
        texto = dados.get("texto", "")
        
        if not texto:
            return jsonify({"error": "Texto não fornecido"}), 400

        # Previsão de sentimento usando o modelo TFLite
        sentimento = prever_sentimento_tflite(texto, interpreter, tokenizer)

        # Libere memória
        gc.collect()

        return jsonify({"sentimento": sentimento})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
