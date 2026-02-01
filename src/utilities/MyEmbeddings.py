
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from typing import List

class MyEmbeddings(Embeddings):
    """Classe personnalisée pour les embeddings avec SentenceTransformer."""
    
    def __init__(self, model: str = 'bert-base-nli-mean-tokens'):
        self.model = SentenceTransformer(model, trust_remote_code=True)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Génère les embeddings pour une liste de textes."""
        return [self.model.encode(t).tolist() for t in texts]
    
    def embed_query(self, query: str) -> List[float]:
        """Génère l'embedding pour une requête."""
        return self.model.encode([query]).tolist()[0]