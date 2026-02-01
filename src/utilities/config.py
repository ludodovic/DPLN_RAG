import os
import json
from pymongo import MongoClient

# Configuration des variables d'environnement

    
def setup_PATH_and_connect_to_local_mongodb_db(config_file: dict):
    with open(config_file, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = config_data.get("LANGSMITH_API_KEY")
    os.environ["MISTRAL_API_KEY"] = config_data.get("MISTRAL_API_KEY")
    os.environ["HF_TOKEN"] = config_data.get("HF_TOKEN")
    db = None
    try:
        client = MongoClient(config_data["Atlas_MongoDB"]["address"] + config_data["Atlas_MongoDB"]["pass"] + config_data["Atlas_MongoDB"]["db_name"])
        db = client["DPLN_RAG"]
        print("Connected to local MongoDB")
    except Exception as e:
        print(f"Error connecting to local MongoDB: {e}")
    
    return db