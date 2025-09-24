
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/health")
async def health():
    return {"status": "ok", "message": "Bot attivo"}

@app.post("/query")
async def query(req: Request):
    try:
        data = await req.json()
        user_message = data.get("message", "").lower().strip()

        # Risposte di cortesia
        if "ciao" in user_message:
            return {"original_message": "Ciao 😊 io sono il tuo assistente, fammi pure delle richieste!"}
        if "come stai" in user_message:
            return {"original_message": "Sto bene, grazie! 😊 E tu?"}
        if user_message == "":
            return {"original_message": "Scrivi qualcosa per iniziare la conversazione 😉"}

        if not os.path.exists("documento.txt"):
            return {"original_message": "⚠ documento.txt non trovato."}

        with open("documento.txt", "r", encoding="utf-8") as f:
            testo = f.read()

        # Divide il documento in chunk di testo
        chunk_size = 500
        chunks = [testo[i:i+chunk_size] for i in range(0, len(testo), chunk_size)]

        # Usa TF-IDF per rappresentare query e document chunks
        vectorizer = TfidfVectorizer(stop_words="italian")
        corpus = chunks + [user_message]
        X = vectorizer.fit_transform(corpus)

        # Calcola similarità
        cosine_similarities = cosine_similarity(X[-1], X[:-1])
        best_match_idx = cosine_similarities.argsort()[0][-1]

        risposta = chunks[best_match_idx]

        return {"original_message": risposta}

    except Exception as e:
        return {"error": str(e)}
