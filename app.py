import os
from flask import Flask, request, jsonify
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gc

app = Flask(__name__)

# Carregando o modelo
model = load_model("files/sentiment_model.h5")

# Carregando o tokenizer e o label encoder
with open("files/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("files/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Função para prever o sentimento do texto
def prever_sentimento(texto, model, tokenizer, max_len_contexto=50):
    # Tokenização
    X_novos_comentarios = tokenizer.texts_to_sequences([texto])
    
    # Padding para garantir o mesmo tamanho
    X_novos_comentarios = pad_sequences(X_novos_comentarios, maxlen=max_len_contexto, padding='post')
    
    # Previsão
    predicoes = model.predict(X_novos_comentarios)
    
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

        # Processamento: prever o sentimento
        sentimento = prever_sentimento(texto, model, tokenizer)

        # Libere memória
        gc.collect()

        return jsonify({"sentimento": sentimento})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = os.getenv('PORT', 5000)  # Usando a variável de ambiente PORT
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)

