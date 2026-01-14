"""Exemple d'utilisation de l'agent RAG."""

from agent import RAGAgent, create_agent
from DocumentLoader import DocumentProcessor
import json

# Créer un agent RAG
# agent = create_agent()

# Exemple d'utilisation synchrone
# question = "Qu'est-ce que le dofus ocre et comment l'obtenir ?"
# response = agent.invoke(question)
# print("Question:", question)
# print("Réponse:", response)

def process_documents_from_config(
    folder_path: str,
    collection_name: str,
    config_file: str = "../config.json",
    pattern: str = "*.html",
    db_name: str = "DPLN_RAG",
    skip_errors: bool = False,
    **kwargs
) -> dict:
    """Fonction helper pour traiter les documents en utilisant la configuration.
    
    Args:
        config_file: Chemin du fichier de configuration JSON.
        folder_path: Chemin du dossier contenant les fichiers à traiter.
        pattern: Pattern de fichiers à traiter.
        db_name: Nom de la base de données MongoDB.
        collection_name: Nom de la collection.
        **kwargs: Arguments supplémentaires pour DocumentProcessor.
        
    Returns:
        Dictionnaire avec les statistiques du traitement.
    """
    # Charger la configuration
    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    db_password = config[0]["Atlas_MongoDB"]
    mongo_connection_string = f"mongodb+srv://Ludzu:{db_password}@dofus-cluster.dm8hqcb.mongodb.net/"
    
    # Créer le processeur
    processor = DocumentProcessor(
        mongo_connection_string=mongo_connection_string,
        db_name=db_name,
        collection_name=collection_name,
        **kwargs
    )
    
    # Traiter le dossier
    try:
        stats = processor.process_folder(folder_path, pattern=pattern, skip_errors=skip_errors)
    finally:
        processor.close()
    
    return stats

process_documents_from_config(
    folder_path="F:/Work/Coding/DPLN_RAG/rawData/Dungeons",
    collection_name="Vec_Dungeons",
    skip_errors=True
),

# Exemple d'utilisation asynchrone
# import asyncio

# async def async_example():
#     question = "Donne-moi des informations sur les monstres de niveau 50."
#     response = await agent.ainvoke(question)
#     print("Question:", question)
#     print("Réponse:", response)

# # Décommentez pour exécuter l'exemple asynchrone
# # asyncio.run(async_example())

# # Exemple de streaming
# def stream_example():
#     question = "Explique-moi comment fonctionne le système de quêtes."
#     print("Question:", question)
#     print("Réponse (streaming):")
#     for chunk in agent.stream(question):
#         print(chunk, end="", flush=True)
#     print()

# Décommentez pour exécuter l'exemple de streaming
# stream_example()

