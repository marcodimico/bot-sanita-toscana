import os
import uvicorn
from fastapi import FastAPI
from bot import create_bot_app

app = create_bot_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Render imposta PORT automaticamente
    uvicorn.run("app:app", host="0.0.0.0", port=port)
