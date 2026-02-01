from typing import List
from rapidfuzz import fuzz

from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.documents import Document

from utilities.MyEmbeddings import MyEmbeddings


class RAGTool:
    def __init__(self, database, embedding_model: str = 'bert-base-nli-mean-tokens', k: int = 3):
        """Initialise l'outil RAG.
        
        Args:
            embedding_model: Le modèle d'embedding à utiliser.
            vector_store: Le magasin de vecteurs. Si None, un magasin par défaut sera créé.
            k: Le nombre de documents à récupérer.
        """
        self.k = k
        self.database = database

        self.embedding_model = MyEmbeddings(embedding_model)
        # Initialisation des magasins de vecteurs
        self.dungeon_vector_store = MongoDBAtlasVectorSearch(
                embedding=self.embedding_model,
                collection=database["Vec_Dungeons"],
                index_name="embedding",
                relevance_score_fn="cosine",
            )
        
        self.quest_vector_store = MongoDBAtlasVectorSearch(
                embedding=self.embedding_model,
                collection=database["Vec_Quests"],
                index_name="embedding",
                relevance_score_fn="cosine",
            )
    
    def get_best_name(self, store_name: str, subject_name: str) -> str:
        list = []
        best_score = 0
        best_match = None

        match store_name:
            case "dungeon":
                collection = self.database["List_Dungeons"]
            case "quest":
                collection = self.database["List_Quests"]
            case _:
                return None
        
        docs = collection.find({}, {"_id": 0, "title": 1})

        for d in docs:
            score = fuzz.ratio(d["title"], subject_name)
            if score > best_score:
                best_score = score
                best_match = d["title"]
        
        print(f"RAGTool: Best score for subject_name '{subject_name}' is {best_score} for match '{best_match}'")
        return None if best_score < 70 else best_match


    def retrieve(self, store_name: str, question: str, subject_name: str = "") -> List[Document]:
        """Récupère les documents pertinents pour une question.
        
        Args:
            store_name: Le nom du magasin de vecteurs à utiliser ("dungeon" ou "quest").
            question: La question à rechercher.
            subject_name: Le nom de la quête ou du donjon à filtrer.
            
        Returns:
            Liste de documents pertinents.
        """


        print(f"RAGTool: Retrieving from store '{store_name}' with question: {question} and subject_name: {subject_name}")

        v_store = None
        match store_name:
            case "dungeon":
                v_store = self.dungeon_vector_store
            case "quest":
                v_store = self.quest_vector_store
            case _:
                return [Document(page_content=f"Nom de magasin inconnu : {store_name}, veillez à utiliser 'dungeon' ou 'quest'", metadata={"error": True})]
        
        if subject_name != "":
            subject_name = self.get_best_name(store_name, subject_name)
            if subject_name is not None:
                print(f"RAGTool: Best match for subject_name is: {subject_name}")
                results = v_store.similarity_search(query=question, k=self.k, pre_filter={"title": subject_name})
            else:
                print(f"RAGTool: No match found for subject_name: {subject_name}")
                results = [Document(page_content=f"Nom de {store_name} inconnu : {subject_name}, veillez à utiliser le nom d'un donjon ou d'une quete existante", metadata={"error": True})]
        else:
            results = v_store.similarity_search(query=question, k=self.k, pre_filter=None)

        return results