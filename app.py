from bot import create_bot_app

# Creiamo direttamente l'istanza FastAPI per Gunicorn
app = create_bot_app()
