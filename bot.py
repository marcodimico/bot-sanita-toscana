from fastapi import FastAPI
from pydantic import BaseModel
import tiktoken

class Query(BaseModel):
    message: str

def create_bot_app() -> FastAPI:
    app = FastAPI()

    @app.get("/")
    def root():
        return {"message": "Bot attivo"}

    @app.post("/query")
    def process_query(query: Query):
        # Tokenizza il messaggio usando tiktoken
        tokens = tiktoken.get_encoding("cl100k_base").encode(query.message)
        return {
            "original_message": query.message,
            "token_count": len(tokens),
            "tokens": tokens
        }

    return app