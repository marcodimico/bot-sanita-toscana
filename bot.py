from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import tiktoken, os

class Query(BaseModel):
    message: str

def create_bot_app() -> FastAPI:
    app = FastAPI()
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

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
