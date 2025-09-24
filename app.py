
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import os
import tiktoken

app = FastAPI()

static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

@app.get("/health")
async def health():
    return {"status": "ok", "message": "Bot attivo"}

@app.post("/query")
async def query(req: Request):
    try:
        data = await req.json()
        user_message = data.get("message", "")
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(user_message)

        return {
            "original_message": user_message,
            "token_count": len(tokens),
            "tokens": tokens
        }
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
