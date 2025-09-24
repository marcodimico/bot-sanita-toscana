from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import os

app = FastAPI()

# Monta la cartella static (HTML, CSS, JS)
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

@app.get("/health")
async def health():
    return {"status": "ok", "message": "Bot attivo"}

@app.post("/query")
async def query(req: Request):
    try:
        data = await req.json()
        user_message = data.get("message", "").lower()

        # Legge documento.txt
        if not os.path.exists("documento.txt"):
            return {"original_message": "⚠ documento.txt non trovato."}

        with open("documento.txt", "r", encoding="utf-8") as f:
            testo = f.read().lower()

        # Cerca la query nel documento
        if user_message in testo:
            risposta = f"Trovato riferimento a '{user_message}' nel documento."
        else:
            risposta = f"Nessun riferimento trovato per '{user_message}'."

        return {"original_message": risposta}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
