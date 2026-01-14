
# ============================================================================
# DocumentProcessor - Traite tous les fichiers d'un dossier
# ============================================================================

# System import
import os
import re
from tkinter.filedialog import test
import bs4
import json
from pathlib import Path

# Utilities
from html_to_markdown import convert
from typing import List, Optional, Any

# Database import
from pymongo import MongoClient

# AI imports
# # from langchain_community.document_loaders import UnstructuredMarkdownLoader
# from langchain_community.document_loaders import BSHTMLLoader
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.documents import Document


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


class DocumentProcessor:
    """Classe pour traiter tous les fichiers d'un dossier et les stocker dans MongoDB Atlas."""
    
    def __init__(
        self,
        mongo_connection_string: str,
        db_name: str = "DPLN_RAG",
        collection_name: str = "Doc_Vectors_semantic",
        index_name: str = "vector_index",
        embedding_model: str = 'bert-base-nli-mean-tokens',
        breakpoint_threshold_type: str = 'percentile'
    ):
        """Initialise le processeur de documents.
        
        Args:
            mongo_connection_string: Chaîne de connexion MongoDB Atlas.
            db_name: Nom de la base de données MongoDB.
            collection_name: Nom de la collection pour les vecteurs.
            index_name: Nom de l'index vectoriel.
            embedding_model: Modèle d'embedding à utiliser.
            breakpoint_threshold_type: Type de seuil pour le semantic chunker.
        """
        self.mongo_connection_string = mongo_connection_string
        self.db_name = db_name
        self.collection_name = collection_name
        self.index_name = index_name
        
        # Connexion à MongoDB
        self.client = MongoClient(mongo_connection_string)
        self.db = self.client[db_name]
        
        # Initialisation des embeddings
        self.embeddings = MyEmbeddings(embedding_model)
        
        # Initialisation du vector store
        self.vector_store = MongoDBAtlasVectorSearch(
            embedding=self.embeddings,
            collection=self.db[collection_name],
            index_name=index_name,
            relevance_score_fn="cosine",
        )
        
        print(f"✓ DocumentProcessor initialisé avec succès")
        print(f"  - Base de données: {db_name}")
        print(f"  - Collection: {collection_name}")
        print(f"  - Modèle d'embedding: {embedding_model}")
    
    def _extract_title_from_html(self, filename: str) -> str:
        """Extrait le titre d'un fichier HTML.
        
        Args:
            filename: Chemin du fichier HTML.
            
        Returns:
            Le titre extrait ou le nom du fichier si extraction échoue.
        """
        try:
            with open(filename, "r", encoding="utf-8") as f:
                soup = bs4.BeautifulSoup(f.read(), 'html.parser')
                title_element = soup.find("h2", class_="wsite-content-title")
                if title_element:
                    return title_element.text.strip()
        except Exception as e:
            print(f"⚠ Erreur lors de l'extraction du titre de {filename}: {e}")
        
        return Path(filename).stem
    
    def _load_document(self, filename: str, title: str, origin: str) -> List[Document]:
        """Charge un document HTML et retourne les documents avec métadonnées.
        
        Args:
            filename: Chemin du fichier HTML à charger.
            
        Returns:
            Liste des documents chargés.
        """
        
        try:
            with open(filename, "r", encoding="utf-8") as f:
                text = f.read()
            
            doc = Document(
                page_content=text,
                metadata={
                    "title": str(title),
                    "source": str(origin),
                    "filename": str(Path(filename).stem),
                    "URL": "https://www.dofuspourlesnoobs.com/" + str(Path(filename).stem) + ".html"
                },
                id=1,
            )
        except Exception as e:
            print(f"└─ ✗ Erreur lors du chargement de {filename}: {e}")
        
        return [doc]
    
    def _html_to_splited_markdown_by_h3_headers(self, html_path, output_dir=None):
        """
        Découpe un fichier markdown en plusieurs sous-documents basés sur les titres "###".
        
        Args:
            html_path (str): Chemin du fichier HTML à convertir et découper
            output_dir (str, optional): Dossier de sortie. Si None, utilise le dossier du fichier source
            
        Returns:
            list: Liste des chemins des fichiers créés
        """
        # Lire le fichier markdown

        file_name_stem =Path(html_path).stem

        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()

        content = convert(html)
        
        # Obtenir le nom du fichier source (sans extension)
        source_title = self._extract_title_from_html(html_path)
        
        # Définir le dossier de sortie
        if output_dir is None:
            output_dir = Path("../markdownData").parent
        else:
            output_dir = Path(f"{output_dir}/{file_name_stem}")
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Pattern pour identifier les titres H3 (### )
        h3_pattern = r'^###\s+(.+)$'
        
        # Diviser le contenu en lignes
        lines = content.split('\n')
        lines.insert(0, f"### {file_name_stem}")
        
        # Liste pour stocker les sections
        sections = []
        current_section = []
        current_title = None
        
        for line in lines:
            # Vérifier si c'est un titre H3
            match = re.match(h3_pattern, line.strip())
            if match:
                # Sauvegarder la section précédente si elle existe
                if current_section and current_title:
                    sections.append({
                        'title': current_title,
                        'content': current_section
                    })
                
                # Démarrer une nouvelle section
                current_title = match.group(1).strip()
                current_section = [line]  # Inclure le titre dans le contenu
            else:
                # Ajouter la ligne à la section courante
                if current_title is not None:
                    current_section.append(line)
        
        # Ajouter la dernière section
        if current_section and current_title:
            sections.append({
                'title': current_title,
                'content': current_section
            })
        
        # Si aucune section H3 trouvée, créer un seul fichier avec tout le contenu
        if not sections:
            output_file = output_dir / f"full.md"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"Source: {source_title}\n\n")
                f.write(content)
            return [str(output_file)]
        
        # Créer les fichiers pour chaque section
        created_files = []
        for i, section in enumerate(sections, 1):
            # Nettoyer le titre pour le nom de fichier
            clean_title = re.sub(r'[^\w\s-]', '', section['title']).strip()
            clean_title = re.sub(r'\s+', '_', clean_title)
            
            # Nom du fichier de sortie
            output_filename = f"{clean_title}.md"
            output_file = output_dir / output_filename
            
            # Écrire le fichier
            with open(output_file, 'w', encoding='utf-8') as f:
                # Ajouter le nom du fichier source en première ligne
                f.write(f"Source: {source_title}\n\n")
                # Écrire le contenu de la section
                f.write('\n'.join(section['content']))
            
            created_files.append(str(output_file))
        ret = {source_title: {"file_list": created_files, "origin_file": html_path}}
        return ret

    def process_folder(
        self,
        folder_path: str,
        pattern: str = "*.html",
        skip_errors: bool = True
    ) -> dict:
        """Traite tous les fichiers d'un dossier.
        
        Args:
            folder_path: Chemin du dossier contenant les fichiers HTML.
            pattern: Pattern de fichiers à traiter (défaut: "*.html").
            skip_errors: Si True, continue même en cas d'erreur sur un fichier.
            
        Returns:
            Dictionnaire avec les statistiques du traitement.
        """
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            raise ValueError(f"Le dossier {folder_path} n'existe pas")
        
        if not folder_path.is_dir():
            raise ValueError(f"{folder_path} n'est pas un dossier")
        
        # Récupérer tous les fichiers correspondant au pattern
        html_files = list(folder_path.glob(pattern))
        
        if not html_files:
            print(f"⚠ Aucun fichier correspondant à '{pattern}' trouvé dans {folder_path}")
            return {
                "total_files": 0,
                "processed_files": 0,
                "failed_files": 0,
                "total_documents": 0,
                "total_doc_ids": 0
            }
        
        print(f"\n{'='*70}")
        print(f"Traitement des fichiers du dossier: {folder_path}")
        print(f"Convertissement en markdrown vers: {folder_path}")
        
        # split tous les fichier en markdown
        md_files = {}
        doc_count = 0
        for html_file in html_files:
            md_file = self._html_to_splited_markdown_by_h3_headers(html_file, str(html_file.parent).replace("rawData","markdownData"))
            md_files.update(md_file)
            doc_count += len(md_file[list(md_file.keys())[0]]["file_list"])

        print(f"Nombre de fichiers convertis en markdown: {doc_count}")
        print(f"{'='*70}\n")

        stats = {
            "total_files": doc_count,
            "processed_files": 0,
            "failed_files": 0,
            "total_documents": 0,
            "total_doc_ids": 0,
            "failed_files_list": []
        }
        i = 0
        doc_id = 1
        for page_title, content in md_files.items():
            file_list = content["file_list"]
            origin_file = content["origin_file"]
            page_docs = []
            i += 1
            
            # Charger tous les documents de la page
            print(f"[{i}/{len(list(md_files.items()))}] Chargement de [{page_title}]...")
            for file_path in file_list:
                try:
                    docs = self._load_document(filename=str(file_path), title=page_title, origin=origin_file)
                    
                    if docs:
                        page_docs.extend(docs)
                        stats["processed_files"] += 1
                        stats["total_documents"] += len(docs)
                    else:
                        stats["failed_files"] += 1
                        stats["failed_files_list"].append(file_path)
                        
                except Exception as e:
                    print(f"└─ ✗ Erreur critique pour {file_path}: {e}")
                    stats["failed_files"] += 1
                    stats["failed_files_list"].append(file_path)
                    
                    if not skip_errors:
                        raise
            if page_docs:
                try:
                    print(f"├─ Stockage des documents originant {origin_file} dans MongoDB Atlas...")
                    ids = [str(x) for x in list(range(doc_id, doc_id + len(page_docs)))]
                    doc_ids = self.vector_store.add_documents(documents=page_docs, ids=ids)
                    print(f"└─ ✓ Ajoutés avec les IDs: [{doc_ids}]")
                    doc_id += len(page_docs)
                    stats["total_doc_ids"] += len(doc_ids)
                except Exception as e:
                    print(f"└─ ✗ Erreur lors du stockage des documents: {e}")
                    if not skip_errors:
                        raise

            print(f"\n{'='*70}")
            print(f"Résumé du chargement: {stats['processed_files']}/{stats['total_files']} fichiers traités")
            print(f"Nombre total de documents: {stats['total_documents']}")
            print(f"{'='*70}\n")
        
        print(f"{'='*70}")
        print(f"STATISTIQUES FINALES")
        print(f"{'='*70}")
        print(f"Fichiers traités avec succès: {stats['processed_files']}/{stats['total_files']}")
        print(f"Documents chargés: {stats['total_documents']}")
        print(f"Documents stockés: {stats['total_doc_ids']}")
        
        if stats['failed_files_list']:
            print(f"\nFichiers échoués ({stats['failed_files']}):")
            for failed_file in stats['failed_files_list']:
                print(f"  - {failed_file}")
        
        with open("stats.json", 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=4)
        
        print(f"{'='*70}\n")
        
        return stats
    
    def close(self):
        """Ferme la connexion à MongoDB."""
        if self.client:
            self.client.close()
            print("✓ Connexion à MongoDB fermée")

