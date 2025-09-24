import os
from flask import Flask, request, jsonify
from bot import Bot

app = Flask(__name__)

# Inizializza il bot
bot = Bot()

# Caricamento iniziale del documento
def load_initial_pdf():
    file_to_load = "documento.txt"
    if os.path.exists(file_to_load):
        bot.carica_documento(file_to_load)
        print(f"{file_to_load} caricato all'avvio.")
    else:
        print(f"{file_to_load} non trovato nella directory {os.getcwd()}.")
        print("File presenti:", os.listdir('.'))

# Route principale
@app.route("/")
def home():
    return "Bot Sanità Toscana attivo 🚀"

# Forza il caricamento del documento
@app.route('/force-load', methods=['GET'])
def force_load():
    file_to_load = "documento.txt"
    if os.path.exists(file_to_load):
        chunks = bot.carica_documento(file_to_load)
        return jsonify({
            'status': 'success',
            'message': f'{file_to_load} caricato con successo!',
            'chunks': chunks
        })
    else:
        return jsonify({
            'status': 'error',
            'message': f'{file_to_load} non trovato nella directory {os.getcwd()}',
            'files': os.listdir('.')
        })

# Endpoint di chat
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "")
    if not user_message:
        return jsonify({"error": "Messaggio vuoto"}), 400

    response = bot.rispondi(user_message)
    return jsonify({"response": response})

# Debug: lista dei file presenti
@app.route("/debug", methods=["GET"])
def debug():
    return jsonify({
        "cwd": os.getcwd(),
        "files": os.listdir('.')
    })

if __name__ == "__main__":
    load_initial_pdf()
    app.run(host="0.0.0.0", port=5000)
