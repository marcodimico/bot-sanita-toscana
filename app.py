# ───────────────────────────────
# Import delle librerie necessarie
# ───────────────────────────────
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import os

# Librerie per elaborazione testo
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ───────────────────────────────
# Creazione dell’app FastAPI
# ───────────────────────────────
app = FastAPI()

# Monta la cartella "static" per servire HTML, CSS, JS
app.mount("/static", StaticFiles(directory="static"), name="static")

# ───────────────────────────────
# Rotta principale: serve la pagina HTML
# ───────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

# ───────────────────────────────
# Rotta per controllare lo stato del bot
# ───────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "message": "Bot attivo"}

# ───────────────────────────────
# Rotta per gestire le richieste di chat
# ───────────────────────────────
@app.post("/query")
async def query(req: Request):
    try:
        # Riceve il messaggio dell’utente
        data = await req.json()
        user_message = data.get("message", "").lower().strip()

        # ───────────────────────────────
        # Risposte di cortesia
        # Qui puoi aggiungere frasi fisse per alcune parole chiave
        # ───────────────────────────────
        if "ciao" in user_message:
            return {"original_message": "Ciao 😊 io sono il tuo assistente, fammi pure delle richieste!"}

        if "come stai" in user_message:
            return {"original_message": "Sto bene, grazie! 😊 E tu?"}

        if "buongiorno" in user_message:
            return {"original_message": "Buongiorno 🌞! Come posso aiutarti oggi?"}

        if "grazie" in user_message:
            return {"original_message": "Prego 😊, è un piacere aiutarti!"}

        if user_message == "":
            return {"original_message": "Scrivi qualcosa per iniziare la conversazione 😉"}

        # ───────────────────────────────
        # Verifica se il documento esiste
        # ───────────────────────────────
        if not os.path.exists("documento.txt"):
            return {"original_message": "⚠ documento.txt non trovato."}

        # Legge il documento
        with open("documento.txt", "r", encoding="utf-8") as f:
            testo = f.read()

        # ───────────────────────────────
        # Logica di ricerca semantica
        # ───────────────────────────────
        chunk_size = 500  # dimensione dei blocchi di testo
        chunks = [testo[i:i+chunk_size] for i in range(0, len(testo), chunk_size)]

        # TF-IDF per rappresentare i blocchi di testo e la query
        vectorizer = TfidfVectorizer(stop_words="italian")
        corpus = chunks + [user_message]
        X = vectorizer.fit_transform(corpus)

        # Calcola la similarità tra la query e i blocchi
        cosine_similarities = cosine_similarity(X[-1], X[:-1])
        best_match_idx = cosine_similarities.argsort()[0][-1]

        risposta = chunks[best_match_idx]

        return {"original_message": risposta}

    except Exception as e:
        # Se qualcosa va storto, restituisce l’errore
        return {"error": str(e)}

