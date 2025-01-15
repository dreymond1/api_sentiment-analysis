from flask import Flask, request, jsonify

# Inicializa o aplicativo Flask
app = Flask(__name__)

# Rota para a API
@app.route("/", methods=["POST"])
def main():
    return "<p>Hello</p>"

