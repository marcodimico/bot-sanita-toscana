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
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = encoding.encode(query.message)
            return {
                "original_message": query.message,
                "token_count": len(tokens),
                "tokens": tokens
            }
        except Exception as e:
            return {"error": str(e)}

    return app
