import os
from typing import List, Dict, Optional
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM 
from langchain.callbacks.manager import CallbackManagerForLLMRun
import cohere 
import google.generativeai as genai
from vector_store_manager import VectorStoreManager
from langchain.schema import Document
from langchain.pydantic_v1 import BaseModel, Field
from typing import Any, Optional

class CohereWrapper(LLM):
    """Wrapper for Cohere LLM to work with LangChain."""
    
    client: Any = None
    model: str = "command"
    
    def __init__(self, api_key: str, model: str = "command"):
        super().__init__()
        self.client = cohere.Client(api_key)
        self.model = model
    
    @property
    def _llm_type(self) -> str:
        return "cohere"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                max_tokens=2000,
                temperature=0.7,
                stop_sequences=stop or []
            )
            return response.generations[0].text.strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"

class GeminiWrapper(LLM):
    """Wrapper for Google Gemini LLM to work with LangChain."""
    
    def __init__(self, api_key: str, model: str = "gemini-pro"):
        super().__init__()
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
    
    @property
    def _llm_type(self) -> str:
        return "gemini"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"

class TravelRAGSystem:
    """Main RAG system for travel itinerary generation."""
    
    def __init__(self, llm_provider: str = "cohere"):
        self.llm_provider = llm_provider
        self.llm = None
        self.vector_store_manager = VectorStoreManager(store_type="chroma")
        self.retrieval_chain = None
        self.setup_system()
    
    def setup_system(self):
        """Initialize the RAG system components."""
        # Setup LLM
        self._setup_llm()
        
        # Setup vector store
        self.vector_store_manager.create_vector_store()
        
        # Setup retrieval chain
        self._setup_retrieval_chain()
    
    def _setup_llm(self):
        """Setup the language model."""
        if self.llm_provider == "cohere":
            api_key = os.getenv("COHERE_API_KEY")
            if not api_key:
                raise ValueError("COHERE_API_KEY environment variable not set")
            self.llm = CohereWrapper(api_key=api_key)
        
        elif self.llm_provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")
            self.llm = GeminiWrapper(api_key=api_key)
        
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
    
    def _setup_retrieval_chain(self):
        """Setup the retrieval QA chain with token optimization."""
        # Create optimized prompt template
        prompt_template = """Create Pakistan travel itinerary using context:
    
Context Information:
{context}

User Query: {question}

Instructions:
1. Create day-by-day itinerary based on requirements
2. Include key attractions from context
3. Add practical info: travel times, costs
4. Suggest accommodation and dining
5. Include cultural etiquette and safety tips
6. Keep concise and realistic

Generate travel itinerary:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
    
        # Create optimized retrieval chain
        self.retrieval_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store_manager.get_retriever(
                k=5,  # Reduced from 8
                search_type="mmr",  # Maximal Marginal Relevance for better diversity
                score_threshold=0.7  # Filter lower quality matches
            ),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
    
    def generate_itinerary(self, user_input: Dict) -> Dict:
        """Generate travel itinerary using RAG with token limit handling."""
    
        # Format user input into a query
        query = self._format_user_query(user_input)
    
        # Generate response using retrieval chain
        try:
            # Create a new prompt with summarized context
            result = self.retrieval_chain({"query": query})
        
            response = {
                "success": True,
                "itinerary": result["result"],
                "source_documents": [
                    {
                        "content": doc.page_content[:300] + "...",
                        "metadata": doc.metadata
                    } for doc in result["source_documents"]  
                ],
                "query_used": query
            }
        
        except Exception as e:
            response = {
                "success": False,
                "error": str(e),
                "itinerary": "Sorry, I encountered an error generating your itinerary. Please try again.",
                "source_documents": [],
                "query_used": query
            }
    
        return response
    
    
    def _format_user_query(self, user_input: Dict) -> str:
        """Format user input into a structured query."""
        
        locations = user_input.get("locations", [])
        start_location = user_input.get("start_location", "")
        start_date = user_input.get("start_date", "")
        nights = user_input.get("nights", 0)
        accommodation = user_input.get("accommodation", "")
        trip_type = user_input.get("trip_type", "")
        group_size = user_input.get("group_size", 1)
        budget = user_input.get("budget", "")
        interests = user_input.get("interests", [])
        
        query_parts = []
        
        if locations:
            query_parts.append(f"Travel destinations: {', '.join(locations)}")
        
        if start_location:
            query_parts.append(f"Starting from: {start_location}")
        
        if nights:
            query_parts.append(f"Duration: {nights} nights")
        
        if accommodation:
            query_parts.append(f"Accommodation preference: {accommodation}")
        
        if trip_type:
            query_parts.append(f"Trip type: {trip_type}")
        
        if group_size:
            query_parts.append(f"Group size: {group_size} people")
        
        if budget:
            query_parts.append(f"Budget: {budget}")
        
        if interests:
            query_parts.append(f"Interests: {', '.join(interests)}")
        
        query_parts.append("Create a detailed day-by-day travel itinerary with attractions, activities, accommodation, dining, and practical tips.")
        
        return " | ".join(query_parts)
    
    def search_destinations(self, query: str, k: int = 5) -> List[Dict]:
        """Search for destination information."""
        docs = self.vector_store_manager.search_with_score(query, k=k)
        
        results = []
        for doc, score in docs:
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "relevance_score": score,
                "source": doc.metadata.get("source", "Unknown")
            })
        
        return results
    
    def add_custom_documents(self, documents: List[str], metadata_list: List[Dict] = None):
        """Add custom travel documents to the vector store."""
        
        if metadata_list is None:
            metadata_list = [{}] * len(documents)
        
        # Create Document objects
        doc_objects = []
        for i, (content, metadata) in enumerate(zip(documents, metadata_list)):
            doc = Document(
                page_content=content,
                metadata=metadata
            )
            doc_objects.append(doc)
        
        # Add to vector store
        self.vector_store_manager.add_documents(doc_objects)
        
        print(f"Added {len(doc_objects)} custom documents to the knowledge base")
    
    def get_system_info(self) -> Dict:
        """Get information about the RAG system."""
        return {
            "llm_provider": self.llm_provider,
            "vector_store_info": self.vector_store_manager.get_store_info(),
            "system_status": "initialized" if self.retrieval_chain else "not_initialized"
        }
    
    def update_knowledge_base(self, force_recreate: bool = False):
        """Update the knowledge base with latest documents."""
        self.vector_store_manager.create_vector_store(force_recreate=force_recreate)
        self._setup_retrieval_chain()
        print("Knowledge base updated successfully")

# Utility functions for the RAG system
def create_rag_system(llm_provider: str = "cohere") -> TravelRAGSystem:
    """Factory function to create and initialize RAG system."""
    return TravelRAGSystem(llm_provider=llm_provider)

def test_rag_system():
    """Test function for the RAG system."""
    try:
        # Create RAG system
        rag = create_rag_system("cohere")
        
        # Test query
        test_input = {
            "locations": ["Lahore", "Islamabad"],
            "start_location": "Karachi",
            "nights": 5,
            "accommodation": "mid-range",
            "trip_type": "cultural",
            "group_size": 2
        }
        
        result = rag.generate_itinerary(test_input)
        
        if result["success"]:
            print("RAG System Test Successful!")
            print(f"Generated itinerary length: {len(result['itinerary'])} characters")
            print(f"Sources used: {len(result['source_documents'])}")
        else:
            print(f"RAG System Test Failed: {result['error']}")
        
        return result
        
    except Exception as e:
        print(f"RAG System Test Error: {str(e)}")
        return None