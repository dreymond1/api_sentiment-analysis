from flask import Flask, request, jsonify
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)

# Inicializa o aplicativo Flask
app = Flask(__name__)

# Variáveis globais
interpreter = None
tokenizer = None
label_encoder = None

# Função para carregar recursos do modelo
def carregar_recursos():
    global interpreter, tokenizer, label_encoder
    try:
        # Carregando o modelo TFLite
        logging.debug("Carregando o modelo TFLite.")
        interpreter = tf.lite.Interpreter(model_path="files/sentiment_model.tflite")
        interpreter.allocate_tensors()
        logging.debug("Modelo TFLite carregado com sucesso.")

        # Carregando tokenizer
        logging.debug("Carregando o tokenizer.")
        with open("files/tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)

        # Carregando label encoder
        logging.debug("Carregando o label encoder.")
        with open("files/label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)

        return interpreter, tokenizer, label_encoder
    except FileNotFoundError as e:
        logging.error(f"Arquivo não encontrado: {e}")
        raise FileNotFoundError(f"Erro ao carregar recursos: {str(e)}")
    except Exception as e:
        logging.error(f"Erro ao carregar recursos: {e}")
        raise ValueError(f"Erro ao carregar recursos: {str(e)}")


def prever_sentimento(texto):
    try:
        logging.debug(f"Texto recebido para previsão: {texto}")

        # Tokenização e padding
        sequencia = tokenizer.texts_to_sequences([texto])
        logging.debug(f"Sequência tokenizada: {sequencia}")

        sequencia_padded = pad_sequences(sequencia, maxlen=50, padding='post')
        logging.debug(f"Sequência com padding: {sequencia_padded}")

        # Preparação para o modelo TFLite
        sequencia_array = np.array(sequencia_padded, dtype=np.float32)
        logging.debug(f"Array numpy preparado para TFLite: {sequencia_array}")

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        logging.debug(f"Detalhes de entrada: {input_details}")
        logging.debug(f"Detalhes de saída: {output_details}")

        # Inferência com TFLite
        interpreter.set_tensor(input_details[0]['index'], sequencia_array)
        interpreter.invoke()
        predicoes = interpreter.get_tensor(output_details[0]['index'])
        logging.debug(f"Predições do modelo: {predicoes}")

        # Decodificação do resultado
        classe_predita = np.argmax(predicoes, axis=1)
        logging.debug(f"Classe prevista: {classe_predita}")

        sentimento = label_encoder.inverse_transform(classe_predita)
        logging.debug(f"Sentimento previsto: {sentimento[0]}")
        return sentimento[0]
    except Exception as e:
        logging.error(f"Erro ao prever sentimento: {str(e)}")
        raise ValueError(f"Erro ao prever sentimento: {str(e)}")

# Rota para a API
@app.route("/", methods=["GET", "POST"])
def main():
    if request.method == "GET":
        return jsonify({"message": "Servidor funcionando. Use POST para enviar dados."}), 200
    elif request.method == "POST":
        try:
            # Carregar recursos uma vez
            if not interpreter or not tokenizer or not label_encoder:
                carregar_recursos()

            dados = request.get_json()
            if not dados or "texto" not in dados:
                return jsonify({"error": "Texto não fornecido"}), 400

            texto = dados["texto"]
            sentimento = prever_sentimento(texto)
            return jsonify({"sentimento": sentimento})
        except Exception as e:
            return jsonify({"error": f"Erro interno: {str(e)}"}), 500

@app.route("/teste", methods=["GET"])
def teste():
    try:
        # Texto de teste
        texto_teste = "eu amo esse produto"
        logging.debug("Iniciando teste com texto fixo.")

        # Carregue os recursos
        if not interpreter or not tokenizer or not label_encoder:
            carregar_recursos()
        logging.debug("Recursos carregados com sucesso.")

        # Preveja o sentimento
        sentimento = prever_sentimento(texto_teste)
        logging.debug(f"Sentimento previsto com sucesso: {sentimento}")

        return jsonify({"texto": texto_teste, "sentimento": sentimento}), 200
    except Exception as e:
        logging.error(f"Erro ao executar teste: {e}")
        return jsonify({"error": f"Erro ao executar teste: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
