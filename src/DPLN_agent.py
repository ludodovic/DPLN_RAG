import json
from pyexpat.errors import messages
import utilities.config as config
import pprint

#Langchain core
import langchain
import langchain_core
from langchain.tools import tool
import langchain.agents
## LLM and Embeddings
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from utilities.MyEmbeddings import MyEmbeddings



## Custom tools
import RAG_tool

class DPLNAgent:
    def __init__(self, model_name="magistral-small-latest", config_file="config/config.json", **kwargs):
        """Initialise l'agent RAG avec les outils et configurations spécifiés.
        
        Args:
            **kwargs: Arguments additionnels pour la création de l'agent.
        """
        self.mongo_connection = config.setup_PATH_and_connect_to_local_mongodb_db(config_file)
        self.mistral_model = ChatMistralAI(
            model="mistral-large-latest",
            temperature=0,
            max_retries=2,
            # other params...
        )
        self.retriver = RAG_tool.RAGTool(
            embedding_model="bert-base-nli-mean-tokens",
            database=self.mongo_connection
        )
        self.agent = langchain.agents.create_agent(
            model=self.mistral_model,
            tools=self.get_tools(),  # Ajouter d'autres outils si nécessaire
        )
    
    def invoke(self, message: str, **kwargs) -> str:
        """Invoke l'agent avec un message synchronement.
        
        Args:
            message: Le message de l'utilisateur.
            **kwargs: Arguments additionnels pour l'agent.
            
        Returns:
            La réponse de l'agent.
        """
        return self.agent.invoke(message, **kwargs)

    def get_tools(self):
        # ---- you must wrap in a closure like this ↓ ----
        @tool
        def retrieve_document(query: str, type: str, subject_name: str = "") -> str:
            """Récupère des informations sur les donjons depuis la base de données MongoDB.
            
            Args:
                query: La requête de l'utilisateur.
                type: Le type d'information à récupérer (dungeon ou quest).
                subject_name: (optionel) Le nom du donjon ou de la quête à filtrer si spécifié.
            Returns:
                Les informations récupérées sous forme de chaîne JSON.
            """
            documents = self.retriver.retrieve(store_name=type, question=query, subject_name=subject_name)

            return documents

        return [retrieve_document]

if __name__ == "__main__":
    agent = DPLNAgent()
    response = agent.invoke({
        "messages": [{ 
            "role": "user", 
            "content": "Comment battre anerice la shushesse dans son donjon le manoir de katrepat ?" 
        }],
    })
    with open("response.json", "w", encoding="utf-8") as f:
        try:
            json.dump(response, f, ensure_ascii=False, indent=4)
        except:
            f.write(str(response))