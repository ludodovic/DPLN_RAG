import os
import json
from pathlib import Path
from pymongo import MongoClient
from typing_extensions import List, TypedDict
from langchain_core.documents import Document

# Charger les clés depuis config.json
config_file = Path(__file__).parent / "config.json"
if config_file.exists():
    with open(config_file, "r", encoding="utf-8") as f:
        config_data = json.load(f)[0]  # Prendre le premier élément du tableau
    
    langsmith_key = config_data.get("LANGSMITH_API_KEY")
    mistral_key = config_data.get("MISTRAL_API_KEY")
    hg_token = config_data.get("HF_TOKEN")
    atlas_mongodb = config_data.get("Atlas_MongoDB")
else:
    # Valeurs par défaut si le fichier n'existe pas
    langsmith_key = None
    mistral_key = None
    hg_token = None
    atlas_mongodb = None

# Configuration des variables d'environnement
os.environ["LANGSMITH_TRACING"] = "true"
if not os.environ.get("LANGSMITH_API_KEY") and langsmith_key:
    os.environ["LANGSMITH_API_KEY"] = langsmith_key
if not os.environ.get("MISTRAL_API_KEY") and mistral_key:
    os.environ["MISTRAL_API_KEY"] = mistral_key
if not os.environ.get("HF_TOKEN") and hg_token:
    os.environ["HF_TOKEN"] = hg_token
    
def connect_to_local_mongodb_db():
    db = None
    try:
        client = MongoClient("mongodb://localhost:51739/?directConnection=true&serverSelectionTimeoutMS=2000&appName=mongosh+2.5.5")
        db = client["DPLN"]
        print("Connected to local MongoDB")
    except Exception as e:
        print(f"Error connecting to local MongoDB: {e}")
    
    return db

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str