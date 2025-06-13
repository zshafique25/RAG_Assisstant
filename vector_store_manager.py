import os
import pickle
from typing import List, Optional
from langchain.vectorstores import Chroma, FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document
from document_loader import TravelDocumentLoader

class VectorStoreManager:
    """Manages vector database operations for the RAG system."""
    
    def __init__(self, store_type: str = "chroma", persist_directory: str = "vectorstore"):
        self.store_type = store_type.lower()
        self.persist_directory = persist_directory
        self.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = None
        self.document_loader = TravelDocumentLoader()
        
    def create_vector_store(self, force_recreate: bool = False) -> None:
        """Create or load vector store with travel documents."""
        
        if self.store_type == "chroma":
            self._create_chroma_store(force_recreate)
        elif self.store_type == "faiss":
            self._create_faiss_store(force_recreate)
        else:
            raise ValueError(f"Unsupported store type: {self.store_type}")
    
    def _create_chroma_store(self, force_recreate: bool = False) -> None:
        """Create or load Chroma vector store."""
        
        if os.path.exists(self.persist_directory) and not force_recreate:
            print("Loading existing Chroma vector store...")
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            print(f"Loaded vector store with {self.vector_store._collection.count()} documents")
        else:
            print("Creating new Chroma vector store...")
            
            # Load and process documents
            documents = self.document_loader.load_documents()
            documents = self.document_loader.add_metadata_to_documents(documents)
            
            # Create vector store
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            
            # Persist the store
            self.vector_store.persist()
            print(f"Created and persisted vector store with {len(documents)} documents")
    
    def _create_faiss_store(self, force_recreate: bool = False) -> None:
        """Create or load FAISS vector store."""
        
        faiss_index_path = os.path.join(self.persist_directory, "faiss_index.pkl")
        
        if os.path.exists(faiss_index_path) and not force_recreate:
            print("Loading existing FAISS vector store...")
            self.vector_store = FAISS.load_local(
                self.persist_directory, 
                self.embeddings,
                index_name="faiss_index"
            )
            print("Loaded FAISS vector store")
        else:
            print("Creating new FAISS vector store...")
            
            # Load and process documents
            documents = self.document_loader.load_documents()
            documents = self.document_loader.add_metadata_to_documents(documents)
            
            # Create vector store
            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            
            # Save the store
            os.makedirs(self.persist_directory, exist_ok=True)
            self.vector_store.save_local(self.persist_directory, index_name="faiss_index")
            print(f"Created and saved FAISS vector store with {len(documents)} documents")
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add new documents to the vector store."""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call create_vector_store() first.")
        
        # Add metadata to documents
        documents = self.document_loader.add_metadata_to_documents(documents)
        
        # Add to vector store
        self.vector_store.add_documents(documents)
        
        # Persist changes
        if self.store_type == "chroma":
            self.vector_store.persist()
        elif self.store_type == "faiss":
            self.vector_store.save_local(self.persist_directory, index_name="faiss_index")
        
        print(f"Added {len(documents)} documents to vector store")
    
    def search_similar_documents(self, query: str, k: int = 5) -> List[Document]:
        """Search for similar documents in the vector store."""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call create_vector_store() first.")
        
        # Perform similarity search
        results = self.vector_store.similarity_search(query, k=k)
        return results
    
    def search_with_score(self, query: str, k: int = 5) -> List[tuple]:
        """Search for similar documents with similarity scores."""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call create_vector_store() first.")
        
        # Perform similarity search with scores
        results = self.vector_store.similarity_search_with_score(query, k=k)
        return results
    
    def get_retriever(self, search_type: str = "similarity", k: int = 5, score_threshold: float = 0.7):
        """Get a retriever object for the vector store."""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call create_vector_store() first.")
        
        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k}
        )
    
    def delete_vector_store(self) -> None:
        """Delete the vector store and its files."""
        import shutil
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
            print(f"Deleted vector store at {self.persist_directory}")
        self.vector_store = None
    
    def get_store_info(self) -> dict:
        """Get information about the vector store."""
        if self.vector_store is None:
            return {"status": "not_initialized"}
        
        info = {
            "store_type": self.store_type,
            "persist_directory": self.persist_directory,
            "embedding_model": "all-MiniLM-L6-v2"
        }
        
        if self.store_type == "chroma":
            try:
                count = self.vector_store._collection.count()
                info["document_count"] = count
            except:
                info["document_count"] = "unknown"
        
        return info