import os
import requests
from bs4 import BeautifulSoup
from typing import List, Dict
import json
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

class TravelDocumentLoader:
    """Loads and processes travel documents for the RAG system."""
    
    def __init__(self):
        self.documents_dir = "travel_documents"
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,  # Reduced from 1000
            chunk_overlap=100,  # Reduced from 200
            length_function=len,
        )
    
    def load_documents(self) -> List[Document]:
        """Load all travel documents from the directory."""
        # Verify documents directory exists
        if not os.path.exists(self.documents_dir):
            raise FileNotFoundError(
                f"Travel documents directory '{self.documents_dir}' not found. "
                "Please create the directory and add your travel documents."
            )
        
        # Check if directory is empty
        if not os.listdir(self.documents_dir):
            raise ValueError(
                f"No documents found in '{self.documents_dir}'. "
                "Please add travel documents in .txt format."
            )
        
        # Load documents
        loader = DirectoryLoader(
            self.documents_dir,
            glob="*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )
        
        documents = loader.load()
        print(f"Loaded {len(documents)} documents from {self.documents_dir}")
        
        # Split documents into chunks
        split_documents = self.text_splitter.split_documents(documents)
        print(f"Split into {len(split_documents)} chunks")
        
        return split_documents
    
    def add_metadata_to_documents(self, documents: List[Document]) -> List[Document]:
        """Add metadata to documents for better retrieval."""
        for doc in documents:
            # Extract filename from source
            filename = os.path.basename(doc.metadata.get('source', ''))
            
            # Add metadata based on filename
            if 'lahore' in filename:
                doc.metadata.update({
                    'city': 'Lahore',
                    'province': 'Punjab',
                    'type': 'city_guide'
                })
            elif 'islamabad' in filename:
                doc.metadata.update({
                    'city': 'Islamabad',
                    'province': 'Federal Capital',
                    'type': 'city_guide'
                })
            elif 'karachi' in filename:
                doc.metadata.update({
                    'city': 'Karachi',
                    'province': 'Sindh',
                    'type': 'city_guide'
                })
            elif 'peshawar' in filename:
                doc.metadata.update({
                    'city': 'Peshawar',
                    'province': 'KPK',
                    'type': 'city_guide'
                })
            elif 'quetta' in filename:
                doc.metadata.update({
                    'city': 'Quetta',
                    'province': 'Balochistan',
                    'type': 'city_guide'
                })
            elif 'hunza' in filename:
                doc.metadata.update({
                    'city': 'Hunza',
                    'province': 'KPK',
                    'type': 'city_guide'
                })
            elif 'skardu' in filename:
                doc.metadata.update({
                    'city': 'Skardu',
                    'province': 'KPK',
                    'type': 'city_guide'
                })
            elif 'northern' in filename:
                doc.metadata.update({
                    'region': 'Northern Areas',
                    'type': 'regional_guide'
                })
            elif 'punjab' in filename:
                doc.metadata.update({
                    'province': 'Punjab',
                    'type': 'cultural_info'
                })
            elif 'tips' in filename:
                doc.metadata.update({
                    'type': 'general_tips'
                })
        
        return documents