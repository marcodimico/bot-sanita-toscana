import os
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document

class Bot:
    def __init__(self):
        self.chunks = []

    def carica_documento(self, file_path):
        """Carica il documento e lo divide in chunk"""
        if not os.path.exists(file_path):
            return ["File non trovato"]

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.split_text(text)

        # Salva come Document (compatibile con langchain)
        self.chunks = [Document(page_content=doc) for doc in docs]
        return [doc.page_content for doc in self.chunks]

    def rispondi(self, query):
        """Risponde cercando il chunk più simile alla query"""
        if not self.chunks:
            return "Nessun documento caricato."

        query = query.lower()
        for doc in self.chunks:
            if query in doc.page_content.lower():
                return doc.page_content

        return "Non ho trovato nulla nel documento su questo argomento."
