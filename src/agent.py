"""Agent LangChain Core avec capacité RAG."""

from typing import List, Optional, Any
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import init_chat_model
# import config

from rag_tool import rag_query_tool


class RAGAgent:
    """Agent LangChain Core qui peut utiliser le RAG pour répondre aux questions."""
    
    def __init__(
        self,
        llm = None,
        tools: Optional[List] = None,
        system_prompt: Optional[str] = None
    ):
        """Initialise l'agent RAG.
        
        Args:
            llm: Le modèle de langage. Si None, sera créé automatiquement.
            tools: Liste d'outils disponibles pour l'agent. Si None, seul l'outil RAG sera utilisé.
            system_prompt: Prompt système pour l'agent. Si None, un prompt par défaut sera utilisé.
        """
        # Initialisation du LLM
        if llm is None:
            llm = init_chat_model("magistral-small-latest", model_provider="mistralai")
        self.llm = llm
        
        # Initialisation des outils
        if tools is None:
            tools = [rag_query_tool]
        else:
            # S'assurer que l'outil RAG est présent
            if rag_query_tool not in tools:
                tools = [rag_query_tool] + tools
        
        self.tools = tools
        
        # Création d'un dictionnaire pour accéder rapidement aux outils par nom
        self.tool_map = {tool.name: tool for tool in self.tools}
        
        # Bind des outils au LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Initialisation du prompt système
        if system_prompt is None:
            system_prompt = """
                Tu es un assistant intelligent spécialisé dans le jeu Dofus.
                Tu as accès à une base de connaissances RAG qui contient des informations détaillées 
                sur les monstres, objets, quêtes et autres éléments du jeu.

                Lorsqu'un utilisateur te pose une question sur Dofus :
                1. Utilise l'outil rag_query_tool pour chercher des informations dans la base de connaissances
                2. Analyse les résultats retournés
                3. Fournis une réponse claire et précise basée sur les informations trouvées

                Si l'information n'est pas disponible dans la base de connaissances, dis-le clairement.
            """
        
        self.system_prompt = system_prompt
    
    def _process_tool_calls(self, messages: List) -> List:
        """Traite les appels d'outils dans les messages et exécute les outils."""
        while messages and hasattr(messages[-1], 'tool_calls') and messages[-1].tool_calls:
            last_message = messages[-1]
            
            # Exécuter chaque appel d'outil
            for tool_call in last_message.tool_calls:
                tool_name = tool_call.get("name", "")
                tool_input = tool_call.get("args", {})
                tool_call_id = tool_call.get("id", "")
                
                # Trouver et exécuter l'outil
                if tool_name in self.tool_map:
                    tool = self.tool_map[tool_name]
                    try:
                        tool_result = tool.invoke(tool_input)
                    except Exception as e:
                        tool_result = f"Erreur lors de l'exécution de l'outil: {str(e)}"
                    
                    # Ajouter le résultat de l'outil comme ToolMessage
                    messages.append(ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call_id
                    ))
            
            # Réinvoker le LLM avec les résultats des outils
            response = self.llm_with_tools.invoke(messages)
            messages.append(response)
        
        return messages
    
    def invoke(self, message: str, **kwargs) -> Any:
        """Invoke l'agent avec un message.
        
        Args:
            message: Le message de l'utilisateur.
            **kwargs: Arguments additionnels pour l'agent.
            
        Returns:
            La réponse de l'agent.
        """
        # Création des messages avec le prompt système
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=message)
        ]
        
        # Invocation du LLM avec les outils
        response = self.llm_with_tools.invoke(messages)
        messages.append(response)
        
        # Traiter les appels d'outils
        messages = self._process_tool_calls(messages)
        
        # Extraire la dernière réponse de l'agent
        if messages:
            last_message = messages[-1]
            if isinstance(last_message, AIMessage):
                return last_message.content
        
        return response.content if hasattr(response, 'content') else str(response)
    
    async def ainvoke(self, message: str, **kwargs) -> Any:
        """Invoke asynchrone l'agent avec un message.
        
        Args:
            message: Le message de l'utilisateur.
            **kwargs: Arguments additionnels pour l'agent.
            
        Returns:
            La réponse de l'agent.
        """
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=message)
        ]
        
        response = await self.llm_with_tools.ainvoke(messages)
        messages.append(response)
        
        # Traiter les appels d'outils de manière asynchrone
        while messages and hasattr(messages[-1], 'tool_calls') and messages[-1].tool_calls:
            last_message = messages[-1]
            
            for tool_call in last_message.tool_calls:
                tool_name = tool_call.get("name", "")
                tool_input = tool_call.get("args", {})
                tool_call_id = tool_call.get("id", "")
                
                if tool_name in self.tool_map:
                    tool = self.tool_map[tool_name]
                    try:
                        tool_result = await tool.ainvoke(tool_input)
                    except Exception as e:
                        tool_result = f"Erreur lors de l'exécution de l'outil: {str(e)}"
                    
                    messages.append(ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call_id
                    ))
            
            response = await self.llm_with_tools.ainvoke(messages)
            messages.append(response)
        
        if messages:
            last_message = messages[-1]
            if isinstance(last_message, AIMessage):
                return last_message.content
        
        return response.content if hasattr(response, 'content') else str(response)
    
    def stream(self, message: str, **kwargs):
        """Stream la réponse de l'agent.
        
        Args:
            message: Le message de l'utilisateur.
            **kwargs: Arguments additionnels pour l'agent.
            
        Yields:
            Les chunks de réponse de l'agent.
        """
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=message)
        ]
        
        for chunk in self.llm_with_tools.stream(messages, **kwargs):
            yield chunk


def create_agent(**kwargs) -> RAGAgent:
    """Fonction helper pour créer un agent RAG.
    
    Args:
        **kwargs: Arguments à passer au constructeur RAGAgent.
        
    Returns:
        Une instance de RAGAgent.
    """
    return RAGAgent(**kwargs)