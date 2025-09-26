# precarga.py
from app import bot
print("Caricamento documento.txt...")
chunks = bot.carica_documento("documento.txt")
print(f"✅ Precaricati {chunks} chunks nella cartella 'db/'")