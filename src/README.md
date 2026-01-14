# Agent LangChain Core avec RAG

Ce module contient un agent LangChain Core capable d'utiliser un système RAG (Retrieval-Augmented Generation) pour répondre aux questions sur Dofus en interrogeant une base de connaissances vectorielle.

## 📁 Structure

- **`rag_tool.py`** : Outil RAG qui encapsule la recherche vectorielle et la génération de réponses
- **`agent.py`** : Agent LangChain Core (`RAGAgent`) qui utilise l'outil RAG
- **`example_usage.py`** : Exemples d'utilisation de l'agent
- **`__init__.py`** : Export du module principal

## 🚀 Utilisation rapide

### Utilisation basique (synchrone)

```python
from src.agent import RAGAgent

# Créer un agent RAG
agent = RAGAgent()

# Poser une question
question = "Qu'est-ce que le dofus ocre et comment l'obtenir ?"
response = agent.invoke(question)
print(response)
```

### Utilisation asynchrone

```python
import asyncio
from src.agent import RAGAgent

async def main():
    agent = RAGAgent()
    question = "Donne-moi des informations sur les monstres de niveau 50."
    response = await agent.ainvoke(question)
    print(response)

asyncio.run(main())
```

### Utilisation avec streaming

```python
from src.agent import RAGAgent

agent = RAGAgent()
question = "Explique-moi comment fonctionne le système de quêtes."

for chunk in agent.stream(question):
    print(chunk, end="", flush=True)
```

### Utilisation avec la fonction helper

```python
from src.agent import create_agent

agent = create_agent()
response = agent.invoke("Quel est le meilleur équipement pour un niveau 100 ?")
```

## ⚙️ Configuration

L'agent utilise automatiquement les configurations définies dans `config.py` :

- **Connexion MongoDB** : Connexion locale à la base de données `DPLN`
- **Vector Store** : Collection `Vector_store` avec index `vector_index`
- **Embeddings** : Modèle Mistral Embed (`mistral-embed`)
- **LLM** : Modèle Mistral (`magistral-small-latest`)
- **Prompt RAG** : Prompt depuis LangChain Hub (`rlm/rag-prompt-mistral`)

### Variables d'environnement requises

Les clés API sont configurées dans `config.py` :
- `LANGSMITH_API_KEY` : Pour le tracing LangSmith
- `MISTRAL_API_KEY` : Pour les embeddings et le LLM Mistral
- `HF_TOKEN` : Token HuggingFace (optionnel)

## 🛠️ Personnalisation

### Personnaliser le LLM

```python
from src.agent import RAGAgent
from langchain.chat_models import init_chat_model

# Créer un LLM personnalisé
custom_llm = init_chat_model("autre-modele", model_provider="mistralai")

# Créer un agent avec le LLM personnalisé
agent = RAGAgent(llm=custom_llm)
```

### Personnaliser le prompt système

```python
from src.agent import RAGAgent

custom_prompt = """Tu es un expert en Dofus spécialisé dans les quêtes.
Réponds toujours de manière détaillée et précise."""

agent = RAGAgent(system_prompt=custom_prompt)
```

### Ajouter des outils supplémentaires

```python
from src.agent import RAGAgent
from langchain_core.tools import tool

@tool
def calculer_damage(attaque: int, defense: int) -> str:
    """Calcule les dégâts infligés."""
    damage = max(1, attaque - defense)
    return f"Dégâts infligés: {damage}"

agent = RAGAgent(tools=[calculer_damage])
# L'outil RAG est automatiquement ajouté
```

### Personnaliser l'outil RAG

```python
from src.rag_tool import RAGTool, get_rag_tool
from src.agent import RAGAgent

# Personnaliser l'outil RAG (plus de documents récupérés)
rag_tool_instance = RAGTool(k=8)  # Récupère 8 documents au lieu de 4

# Note: L'agent utilise automatiquement l'instance globale via get_rag_tool()
# Pour utiliser une instance personnalisée, vous devrez modifier rag_tool.py
```

## 📚 API de l'agent

### Classe `RAGAgent`

#### Constructeur

```python
RAGAgent(
    llm=None,                    # Modèle de langage (optionnel)
    tools=None,                  # Liste d'outils supplémentaires (optionnel)
    system_prompt=None           # Prompt système personnalisé (optionnel)
)
```

#### Méthodes

- **`invoke(message: str, **kwargs) -> str`** : Exécute l'agent de manière synchrone
- **`ainvoke(message: str, **kwargs) -> str`** : Exécute l'agent de manière asynchrone
- **`stream(message: str, **kwargs) -> Iterator`** : Stream la réponse de l'agent

## 🔍 Fonctionnement interne

1. **Réception de la question** : L'agent reçoit une question de l'utilisateur
2. **Appel du LLM** : Le LLM décide s'il doit utiliser l'outil RAG
3. **Recherche vectorielle** : Si nécessaire, l'outil RAG effectue une recherche dans MongoDB
4. **Génération** : Le LLM génère une réponse basée sur les documents récupérés
5. **Retour** : La réponse finale est retournée à l'utilisateur

### Flux de traitement des outils

L'agent gère automatiquement les appels d'outils multiples :
- Si le LLM décide d'appeler un outil, celui-ci est exécuté
- Les résultats sont ajoutés au contexte
- Le LLM est réinvité avec les résultats pour générer la réponse finale

## 📝 Exemples d'utilisation

Voir le fichier `example_usage.py` pour des exemples complets d'utilisation.

## 🔗 Dépendances

- `langchain` / `langchain-core`
- `langchain-mongodb`
- `langchain-mistralai`
- `pymongo`
- `config.py` (fichier de configuration du projet)

## ⚠️ Notes importantes

- L'agent nécessite une connexion MongoDB active
- Le vector store doit être préalablement peuplé avec des documents
- Les clés API doivent être configurées dans `config.py`
- L'outil RAG utilise une instance globale (singleton) pour optimiser les performances

## 🐛 Dépannage

### Erreur de connexion MongoDB

Assurez-vous que MongoDB est démarré et que la chaîne de connexion dans `config.py` est correcte.

### Erreur d'API

Vérifiez que les clés API dans `config.py` sont valides et que les variables d'environnement sont correctement définies.

### Aucun document trouvé

Vérifiez que le vector store contient des documents et que l'index `vector_index` existe dans MongoDB.

