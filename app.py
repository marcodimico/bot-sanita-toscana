from flask import Flask, request, jsonify, render_template_string
import requests
import os
import chromadb
import pypdf
from datetime import datetime
import re

app = Flask(__name__)


class Bot:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="db")
        self.collection = self.client.get_or_create_collection(
            "documenti_toscana",
            metadata={"hnsw:space": "cosine"}
        )
        self.chat_history = []
        # Stato per la gestione del ticket
        self.awaiting_ticket_field = None
        self.ticket_data = {}
        self.ticket_fields = [
            "nome e cognome",
            "reparto",
            "ubicazione",
            "telefono",
            "problema"
        ]

    def carica_documento(self, file_path):
        try:
            if file_path.endswith('.txt'):
                return self._carica_txt(file_path)
            elif file_path.endswith('.csv'):
                return self._carica_csv(file_path)
            elif file_path.endswith('.pdf'):
                return self._carica_pdf_original(file_path)
            else:
                raise Exception(f"Formato non supportato: {file_path}")
        except Exception as e:
            raise Exception(f"Errore durante il caricamento: {str(e)}")

    def _carica_txt(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            contenuto = f.read()

        existing_ids = self.collection.get(where={"source": os.path.basename(file_path)})["ids"]
        if existing_ids:
            self.collection.delete(ids=existing_ids)

        chunks = self._split_text(contenuto, chunk_size=1000, overlap=200)

        documents = []
        metadatas = []
        ids = []

        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 50:
                continue
            documents.append(chunk.strip())
            metadatas.append({
                "source": os.path.basename(file_path),
                "full_path": file_path,
                "chunk_id": i,
                "chunk_length": len(chunk),
                "upload_date": datetime.now().isoformat(),
                "first_words": chunk[:100]
            })
            ids.append(f"{os.path.basename(file_path)}_{i}")

        batch_size = 20
        for i in range(0, len(documents), batch_size):
            try:
                self.collection.add(
                    documents=documents[i:i + batch_size],
                    metadatas=metadatas[i:i + batch_size],
                    ids=ids[i:i + batch_size]
                )
            except Exception as e:
                print(f"Errore batch {i}: {e}")

        return len(documents)

    def _carica_csv(self, file_path):
        import csv
        existing_ids = self.collection.get(where={"source": os.path.basename(file_path)})["ids"]
        if existing_ids:
            self.collection.delete(ids=existing_ids)

        documents = []
        metadatas = []
        ids = []

        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                testo = " | ".join([f"{k}: {v}" for k, v in row.items() if v])
                if len(testo.strip()) < 20:
                    continue
                documents.append(testo)
                metadatas.append({
                    "source": os.path.basename(file_path),
                    "full_path": file_path,
                    "chunk_id": i,
                    "chunk_length": len(testo),
                    "upload_date": datetime.now().isoformat(),
                    "row_number": i + 1
                })
                ids.append(f"{os.path.basename(file_path)}_row_{i}")

        batch_size = 50
        for i in range(0, len(documents), batch_size):
            self.collection.add(
                documents=documents[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size],
                ids=ids[i:i + batch_size]
            )

        return len(documents)

    def _carica_pdf_original(self, file_path):
        with open(file_path, 'rb') as f:
            reader = pypdf.PdfReader(f)
            full_text = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n\n"

        existing_ids = self.collection.get(where={"source": os.path.basename(file_path)})["ids"]
        if existing_ids:
            self.collection.delete(ids=existing_ids)

        chunks = self._split_text(full_text, chunk_size=1000, overlap=200)

        documents = []
        metadatas = []
        ids = []

        for i, chunk in enumerate(chunks):
            chunk_text = chunk.strip()
            if len(chunk_text) < 50:
                continue
            documents.append(chunk_text)
            metadatas.append({
                "source": os.path.basename(file_path),
                "full_path": file_path,
                "chunk_id": i,
                "chunk_length": len(chunk_text),
                "upload_date": datetime.now().isoformat(),
                "first_words": chunk_text[:100]
            })
            ids.append(f"{os.path.basename(file_path)}_{i}")

        batch_size = 20
        for i in range(0, len(documents), batch_size):
            try:
                self.collection.add(
                    documents=documents[i:i + batch_size],
                    metadatas=metadatas[i:i + batch_size],
                    ids=ids[i:i + batch_size]
                )
            except Exception as e:
                print(f"Errore batch {i}: {e}")

        return len(documents)

    def _split_text(self, text, chunk_size=1000, overlap=200):
        if not text or len(text) < chunk_size:
            return [text] if text else []

        section_headers = re.findall(r'\n\d+\s+[A-Z]', text)
        if section_headers:
            sections = re.split(r'(\n\d+\s+[A-Z])', text)
            reconstructed = []
            for i in range(1, len(sections), 2):
                if i + 1 < len(sections):
                    reconstructed.append(sections[i] + sections[i + 1].lstrip())
                else:
                    reconstructed.append(sections[i])
            if len(sections) % 2 == 1 and sections[-1].strip():
                reconstructed.append(sections[-1])

            chunks = []
            current = ""
            for sec in reconstructed:
                if len(current) + len(sec) <= chunk_size:
                    current += sec
                else:
                    if current:
                        chunks.append(current)
                    if len(sec) > chunk_size:
                        chunks.extend(self._split_text_simple(sec, chunk_size, overlap))
                    else:
                        current = sec
            if current:
                chunks.append(current)
            return [c.strip() for c in chunks if c.strip() and len(c.strip()) >= 50]
        else:
            return self._split_text_simple(text, chunk_size, overlap)

    def _split_text_simple(self, text, chunk_size=1000, overlap=200):
        if not text:
            return []
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            if len(chunk.strip()) >= 50:
                chunks.append(chunk)
            start += (chunk_size - overlap)
        return chunks

    def query_con_groq(self, domanda, n_results=10):
        try:
            results = self.collection.query(
                query_texts=[domanda],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )

            if not results["documents"] or not results["documents"][0]:
                return "🔍 Nessun documento trovato nel database."

            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]

            if distances:
                min_distance = min(distances)
                threshold = min_distance + 0.4
            else:
                threshold = 1.5

            relevant_docs = []
            for i, (doc, distance) in enumerate(zip(documents, distances)):
                if distance <= threshold and len(doc.strip()) > 50:
                    relevant_docs.append({
                        'content': doc,
                        'metadata': metadatas[i],
                        'distance': distance
                    })

            if not relevant_docs:
                return "🤔 Non ho trovato informazioni sufficientemente rilevanti nei documenti.\n\n📧 Per assistenza personalizzata, scrivi: **Apertura ticket**"

            relevant_docs.sort(key=lambda x: x['distance'])
            top_docs = relevant_docs[:7]
            contesto = "\n\n---\n\n".join([doc['content'] for doc in top_docs])

            history_context = ""
            for turn in self.chat_history[-3:]:
                history_context += f"UTENTE: {turn['domanda']}\nASSISTENTE: {turn['risposta']}\n\n"

            prompt = f"""Ciao! 🤖 Sono il tuo assistente per la Sanità Toscana.
Ecco la nostra conversazione recente:
{history_context}
Basandomi SOLO sulle informazioni NEI DOCUMENTI QUI SOTTO, ti risponderò in modo chiaro, preciso e amichevole.

DOMANDA: {domanda}

DOCUMENTI RILEVANTI:
{contesto}

REGOLE FONDAMENTALI (SEGUI ALLA LETTERA):
1. RISPONDI SOLO CON LE INFORMAZIONI PRESENTI NEI DOCUMENTI SOPRA. NON INVENTARE NULLA.
2. LEGGI ATTENTAMENTE TUTTO IL TESTO DEI DOCUMENTI. NON FERMARTI ALLA PRIMA RIGA.
3. SE LA RISPOSTA È NEL DOCUMENTO, COPIALA TAL QUALE, PAROLA PER PAROLA, SENZA MODIFICHE, OMISSIONI O RISCRITTURE.
4. ELENCA SEMPRE TUTTE LE INFORMAZIONI RICHIESTE. NON OMETTERE NULLA, NEANCHE DETTAGLI CHE SEMBRANO MINORI (ES. TELEFONO, CODICI, NOTE).
5. SE NON TROVI LA RISPOSTA, DILLO CHIARAMENTE: "Non ho trovato questa informazione nei documenti."
6. CITA SEMPRE la fonte (es. "Secondo il documento X, sezione Y...").
7. Usa elenchi puntati se serve per chiarezza.
8. Mantieni un tono amichevole ma professionale.
9. SE LA DOMANDA RIGUARDA "CUP 2.0", CERCA ESPLICITAMENTE LA SEZIONE "13. CUP 2.0" E COPIA TUTTO IL CONTENUTO DI QUELLA SEZIONE CHE RISPONDE ALLA DOMANDA.
10. SE LA DOMANDA RIGUARDA "CIS CARDIOLOGIA", CERCA ESPLICITAMENTE LA SEZIONE "14. CIS CARDIOLOGIA" E COPIA TUTTO IL CONTENUTO DI QUELLA SEZIONE CHE RISPONDE ALLA DOMANDA.

Ecco la mia risposta:
"""

            groq_api_key = os.environ.get('GROQ_API_KEY')
            if not groq_api_key:
                return "❌ Errore: API Key Groq non configurata."

            headers = {
                'Authorization': f'Bearer {groq_api_key}',
                'Content-Type': 'application/json'
            }

            payload = {
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "top_p": 0.9,
                "max_tokens": 1000
            }

            response = requests.post(
                'https://api.groq.com/openai/v1/chat/completions',
                headers=headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                risposta = response.json()['choices'][0]['message']['content'].strip()
                self.chat_history.append({"domanda": domanda, "risposta": risposta})
                if len(self.chat_history) > 10:
                    self.chat_history.pop(0)
                return risposta
            else:
                return f"❌ Errore API Groq: {response.status_code}"

        except Exception as e:
            return f"❌ Errore durante la ricerca: {str(e)}"

    def get_stats(self):
        try:
            collection_data = self.collection.get()
            if not collection_data["ids"]:
                return "📊 Database vuoto."
            total_docs = len(collection_data["ids"])
            sources = set()
            for metadata in collection_data["metadatas"]:
                if metadata and "source" in metadata:
                    sources.add(metadata["source"])
            return f"📊 Database: {total_docs} chunks da {len(sources)} file(s): {', '.join(sources)}"
        except Exception as e:
            return f"❌ Errore nel recuperare le statistiche: {str(e)}"

    def cancella_cronologia(self):
        self.chat_history = []
        return "🧹 Cronologia della chat cancellata!"

    def invia_email_ticket(self, dati_ticket):
        """Invia un'email con i dati del ticket (opzionale)"""
        try:
            smtp_user = os.environ.get("SMTP_USER")
            smtp_password = os.environ.get("SMTP_PASSWORD")
            destinatario = os.environ.get("SUPPORT_EMAIL", "supporto.sanita@toscana.it")

            if not smtp_user or not smtp_password:
                print("📧 Email non configurata: mancano SMTP_USER o SMTP_PASSWORD")
                return False

            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart

            msg = MIMEMultipart()
            msg["From"] = smtp_user
            msg["To"] = destinatario
            msg["Subject"] = "🆕 Ticket aperto dal bot - Sanità Toscana"

            corpo = "Un utente ha aperto un ticket tramite il bot:\n\n"
            for campo, valore in dati_ticket.items():
                corpo += f"{campo.capitalize()}: {valore}\n"

            msg.attach(MIMEText(corpo, "plain"))

            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.starttls()
                server.login(smtp_user, smtp_password)
                server.sendmail(smtp_user, destinatario, msg.as_string())

            print("✅ Email inviata con successo")
            return True
        except Exception as e:
            print(f"❌ Errore invio email: {e}")
            return False


bot = Bot()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>🤖 Assistente Documentale Sanità Toscana</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #FF8C00 0%, #FFA500 100%);
            min-height: 100vh; display: flex; align-items: center; justify-content: center;
            padding: 20px;
        }
        .container {
            background: white;
            border: 4px solid #8A2BE2;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(138, 43, 226, 0.3);
            width: 100%; max-width: 900px; height: 85vh; display: flex; flex-direction: column;
        }
        .header {
            background: linear-gradient(45deg, #8A2BE2, #9370DB);
            color: white;
            padding: 20px; border-radius: 15px 15px 0 0; text-align: center;
        }
        .header h1 { font-size: 1.5em; margin-bottom: 5px; }
        .header p { font-size: 0.9em; opacity: 0.9; }
        .stats {
            background: rgba(255,255,255,0.2); padding: 10px; border-radius: 10px; margin-top: 10px;
            font-size: 0.8em; text-align: center;
        }
        #chat {
            flex: 1; padding: 20px; overflow-y: auto; scroll-behavior: smooth;
            background: #FFF8F0; border-left: 3px solid #FF8C00;
        }
        .message {
            margin: 15px 0; padding: 15px; border-radius: 15px; max-width: 85%;
            animation: slideIn 0.3s ease-out; line-height: 1.5;
        }
        .user { 
            background: linear-gradient(45deg, #8A2BE2, #9370DB); color: white;
            margin-left: auto; text-align: right; 
            border-bottom-right-radius: 5px;
        }
        .bot { 
            background: white; border: 2px solid #8A2BE2; box-shadow: 0 2px 5px rgba(138, 43, 226, 0.2);
            border-bottom-left-radius: 5px;
        }
        .input-area {
            padding: 20px; border-top: 2px solid #8A2BE2; display: flex; gap: 10px;
            align-items: center; flex-wrap: wrap; background: #FFF8F0;
        }
        #message {
            flex: 1; padding: 12px 16px; border: 2px solid #8A2BE2; border-radius: 25px;
            outline: none; font-size: 14px; transition: all 0.3s; min-width: 300px;
            background: white;
        }
        #message:focus { 
            border-color: #FF8C00; box-shadow: 0 0 0 3px rgba(255, 140, 0, 0.3); 
        }
        button {
            background: linear-gradient(45deg, #8A2BE2, #9370DB); color: white;
            border: none; padding: 12px 15px; border-radius: 20px; cursor: pointer;
            font-weight: 600; transition: transform 0.2s; white-space: nowrap;
            font-size: 12px;
        }
        button:hover { transform: scale(1.05); }
        button:disabled { opacity: 0.6; transform: none; cursor: not-allowed; }
        .loading { opacity: 0.7; }
        .stats-btn {
            background: rgba(255,255,255,0.3); padding: 5px 10px; border-radius: 15px;
            font-size: 0.8em; margin-left: 10px;
        }
        @keyframes slideIn { 
            from { opacity: 0; transform: translateY(20px); } 
            to { opacity: 1; transform: translateY(0); } 
        }
        .welcome {
            background: linear-gradient(45deg, #FFF8F0, #FFEBCD); 
            border-left: 4px solid #FF8C00; padding: 20px; margin: 10px 0;
            border-radius: 10px;
        }
        .separator { border-top: 2px solid #8A2BE2; margin: 20px 0; opacity: 0.3; }

        @media (max-width: 768px) {
            .container { height: 95vh; margin: 10px; }
            .input-area { flex-direction: column; gap: 8px; }
            #message { min-width: 100%; }
            button { width: 100%; margin: 2px 0; }
            .header h1 { font-size: 1.2em; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Assistente Documentale Sanità Toscana</h1>
            <p>Powered by ChromaDB + Groq API</p>
            <div class="stats" id="stats">Caricamento statistiche...</div>
            <button class="stats-btn" onclick="loadStats()">📊 Aggiorna</button>
        </div>
        <div id="chat">
            <div class="welcome">
                <h3>👋 Benvenuto nell'Assistente Documentale!</h3>
                <p>Chiedimi qualsiasi cosa sui sistemi informatici, le procedure o le applicazioni aziendali della Sanità Toscana.</p>
                <p>💡 <strong>Se alla fine della nostra conversazione non hai risolto il tuo problema, possiamo aprire direttamente un ticket.</strong><br>
                Per farlo, scrivi semplicemente: <code>Apertura ticket</code></p>
                <p>Sono qui per aiutarti in modo chiaro, veloce e... con un sorriso! 😊</p>
            </div>
        </div>
        <div class="input-area">
            <input type="text" id="message" placeholder="💬 Fai una domanda sui documenti..." onkeypress="handleEnter(event)">
            <button id="sendBtn" onclick="sendMessage()">🚀 Invia</button>
            <button onclick="clearChat()">🧹 Pulisci</button>
            <button onclick="clearHistory()">🗑️ Cronologia</button>
        </div>
    </div>

    <script>
    window.onload = function() {
        loadStats();
    };

    function handleEnter(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            sendMessage();
        }
    }

    async function sendMessage() {
        const messageInput = document.getElementById('message');
        const sendBtn = document.getElementById('sendBtn');
        const chat = document.getElementById('chat');
        const message = messageInput.value.trim();

        if (!message) return;

        chat.innerHTML += '<div class="separator"></div>';
        chat.innerHTML += `<div class="message user">🙋‍♂️ TU: ${message}</div>`;
        messageInput.value = '';
        sendBtn.disabled = true;
        sendBtn.innerHTML = '⏳ Pensando...';
        chat.scrollTop = chat.scrollHeight;

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({message: message})
            });
            const data = await response.json();
            const saluti = ["Ciao!", "Ehilà!", "Buongiorno!", "Salve!", "Ciao, eccomi qui! 🤖"];
            const saluto = saluti[Math.floor(Math.random() * saluti.length)];
            chat.innerHTML += `<div class="message bot">🤖 ${saluto} ${data.response}<br><br>Spero di esserti stato utile! 😊</div>`;
        } catch (error) {
            chat.innerHTML += `<div class="message bot" style="color: red;">❌ Errore: ${error.message}</div>`;
        } finally {
            sendBtn.disabled = false;
            sendBtn.innerHTML = '🚀 Invia';
            chat.scrollTop = chat.scrollHeight;
        }
    }

    async function loadStats() {
        try {
            const response = await fetch('/stats');
            const data = await response.json();
            document.getElementById('stats').textContent = data.stats;
        } catch (error) {
            document.getElementById('stats').textContent = '❌ Errore nel caricamento stats';
        }
    }

    function clearChat() {
        const chat = document.getElementById('chat');
        chat.innerHTML = `
            <div class="welcome">
                <h3>👋 Chat pulita!</h3>
                <p>Puoi iniziare una nuova conversazione.</p>
            </div>
        `;
    }

    async function clearHistory() {
        try {
            const response = await fetch('/clear-history', {method: 'POST'});
            const data = await response.json();
            const chat = document.getElementById('chat');
            chat.innerHTML += `<div class="message bot">🗑️ ${data.message}</div>`;
            chat.scrollTop = chat.scrollHeight;
        } catch (error) {
            console.error('Errore:', error);
        }
    }
    </script>
</body>
</html>
"""


@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)


@app.route('/chat', methods=['POST'])
def chat():
    try:
        query = request.json.get('message', '').strip()
        if not query:
            return jsonify({'response': 'Per favore, scrivi una domanda.'})

        # --- Gestione speciale: Apertura ticket ---
        if query.lower() == "apertura ticket":
            bot.awaiting_ticket_field = 0
            bot.ticket_data = {}
            first_field = bot.ticket_fields[0]
            return jsonify({
                'response': f"📬 Apertura ticket in corso.\nPer favore, indicami il tuo **{first_field}**:"
            })

        if bot.awaiting_ticket_field is not None:
            current_index = bot.awaiting_ticket_field
            field_name = bot.ticket_fields[current_index]
            bot.ticket_data[field_name] = query

            if current_index + 1 < len(bot.ticket_fields):
                bot.awaiting_ticket_field += 1
                next_field = bot.ticket_fields[bot.awaiting_ticket_field]
                return jsonify({
                    'response': f"✅ {field_name.capitalize()} registrato.\nOra, per favore, indicami: **{next_field}**"
                })
            else:
                summary = "\n".join([f"• **{k.capitalize()}**: {v}" for k, v in bot.ticket_data.items()])
                # ✅ Invio email con gestione errori
                try:
                    bot.invia_email_ticket(bot.ticket_data)
                except Exception as e:
                    print(f"📧 Errore invio email (ma il ticket è comunque salvato): {e}")
                    # Non bloccare la risposta se l'email fallisce
                bot.awaiting_ticket_field = None
                bot.ticket_data = {}
                return jsonify({
                    'response': (
                        "✅ **Ticket compilato con successo!**\n\n"
                        "Ecco i dati che hai fornito:\n\n"
                        f"{summary}\n\n"
                        "📬 Il team di supporto è stato notificato via email. "
                        "Riceverai assistenza al più presto!\n\n"
                        "Puoi continuare a chiedermi informazioni sui documenti."
                    )
                })

        risposta = bot.query_con_groq(query)
        return jsonify({'response': risposta})

    except Exception as e:
        # ✅ Questo cattura TUTTI gli errori, inclusi quelli dell'email
        print(f"🚨 Errore critico in /chat: {e}")
        return jsonify({'response': f"❌ Errore imprevisto: {str(e)}"}), 500


@app.route('/stats')
def stats():
    try:
        stats_info = bot.get_stats()
        return jsonify({'stats': stats_info})
    except Exception as e:
        return jsonify({'stats': f'❌ Errore: {str(e)}'})


@app.route('/clear-history', methods=['POST'])
def clear_history():
    try:
        message = bot.cancella_cronologia()
        return jsonify({'message': message})
    except Exception as e:
        return jsonify({'message': f'❌ Errore: {str(e)}'})


@app.route('/health')
def health():
    return jsonify({'status': 'OK', 'message': 'Bot Sanità Toscana funzionante!'})


@app.route('/force-load')
def force_load():
    target_file = "documento.txt"
    try:
        if os.path.exists(target_file):
            chunks = bot.carica_documento(target_file)
            return jsonify({
                'status': 'success',
                'message': f'{target_file} caricato con successo!',
                'chunks': chunks,
                'tipo': 'TXT'
            })
        else:
            files = [f for f in os.listdir('.') if f.endswith(('.txt', '.csv', '.pdf'))]
            return jsonify({
                'status': 'error',
                'message': f"'{target_file}' non trovato nella directory.",
                'available_files': files,
                'current_dir': os.getcwd()
            })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Errore durante il caricamento: {str(e)}'
        }), 500


@app.route('/debug')
def debug():
    try:
        files = os.listdir('.')
        txt_files = [f for f in files if f.endswith('.txt')]
        csv_files = [f for f in files if f.endswith('.csv')]
        pdf_files = [f for f in files if f.endswith('.pdf')]
        stats = bot.get_stats()
        return jsonify({
            'tutti_file': files,
            'txt_trovati': txt_files,
            'csv_trovati': csv_files,
            'pdf_trovati': pdf_files,
            'statistiche_db': stats,
            'directory_corrente': os.getcwd(),
            'documento_txt_esiste': os.path.exists('documento.txt')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # ❌ NON caricare all'avvio su Render
    # load_initial_document_if_needed()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)