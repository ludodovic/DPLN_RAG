"""Outil RAG pour la recherche vectorielle et la génération de réponses."""

from typing import Optional, List
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mistralai import MistralAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain import hub
import config


class RAGTool:
    """Classe qui encapsule la fonctionnalité RAG."""
    
    def __init__(
        self,
        vector_store: Optional[MongoDBAtlasVectorSearch] = None,
        llm = None,
        prompt = None,
        k: int = 4
    ):
        """Initialise l'outil RAG.
        
        Args:
            vector_store: Le vector store MongoDB. Si None, sera créé automatiquement.
            llm: Le modèle de langage. Si None, sera créé automatiquement.
            prompt: Le prompt RAG. Si None, sera chargé depuis LangChain Hub.
            k: Nombre de documents à récupérer lors de la recherche.
        """
        # Initialisation des embeddings et du vector store
        if vector_store is None:
            embeddings = MistralAIEmbeddings(
                model="mistral-embed",
                mistral_api_key=config.mistral_key,
                api_key=config.hg_token
            )
            my_database = config.connect_to_local_mongodb_db()
            vector_store = MongoDBAtlasVectorSearch(
                embedding=embeddings,
                collection=my_database["Vector_store"],
                index_name="embedding",
                relevance_score_fn="cosine",
            )
        
        self.vector_store = vector_store
        self.k = k
        
        # Initialisation du LLM
        if llm is None:
            llm = init_chat_model("magistral-small-latest", model_provider="mistralai")
        self.llm = llm
        
        # Initialisation du prompt
        if prompt is None:
            prompt = hub.pull("rlm/rag-prompt-mistral")
        self.prompt = prompt
    
    def retrieve(self, question: str) -> List[Document]:
        """Récupère les documents pertinents pour une question.
        
        Args:
            question: La question à rechercher.
            
        Returns:
            Liste de documents pertinents.
        """
        return self.vector_store.similarity_search(question, k=self.k)
    
    def generate(self, question: str, context: List[Document]) -> str:
        """Génère une réponse à partir d'une question et d'un contexte.
        
        Args:
            question: La question à répondre.
            context: Les documents de contexte.
            
        Returns:
            La réponse générée.
        """
        docs_content = "\n\n".join(doc.page_content for doc in context)
        messages = self.prompt.invoke({"question": question, "context": docs_content})
        response = self.llm.invoke(messages)
        return response.content
    
    def query(self, question: str) -> str:
        """Effectue une recherche RAG complète : récupération + génération.
        
        Args:
            question: La question à traiter.
            
        Returns:
            La réponse générée basée sur les documents récupérés.
        """
        context = self.retrieve(question)
        answer = self.generate(question, context)
        return answer


# Instance globale de l'outil RAG
_rag_tool_instance: Optional[RAGTool] = None


def get_rag_tool() -> RAGTool:
    """Récupère ou crée l'instance globale de l'outil RAG."""
    global _rag_tool_instance
    if _rag_tool_instance is None:
        _rag_tool_instance = RAGTool()
    return _rag_tool_instance


@tool
def rag_query_tool(question: str) -> str:
    """Outil pour interroger la base de connaissances RAG.
    
    Utilise cet outil lorsque tu as besoin d'informations sur Dofus ou des contenus
    de la base de connaissances. Cet outil effectue une recherche dans les documents
    vectorisés et génère une réponse basée sur le contexte trouvé.
    
    Args:
        question: La question à poser à la base de connaissances.
        
    Returns:
        La réponse générée basée sur les documents pertinents.
    """
    rag = get_rag_tool()
    return rag.query(question)

