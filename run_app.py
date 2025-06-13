import os
import sys
import subprocess
import importlib.util
from pathlib import Path
from dotenv import load_dotenv
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class MusafirAIRunner:
    """Main runner class for the MusafirAI application."""
    
    def __init__(self):
        self.app_name = "MusafirAI"
        self.required_packages = [
            'streamlit',
            'langchain',
            'chromadb',
            'sentence-transformers',
            'cohere',
            'google-generativeai',
            'reportlab',
            'python-dotenv'
        ]
        self.app_file = "MusafirAI.py"
        
    def print_banner(self):
        """Print application banner."""
        banner = """
        ╔══════════════════════════════════════════════════════════════╗
        ║                                                              ║
        ║    🧳 MusafirAI - RAG-Powered Travel Assistant 🧳           ║
        ║                                                              ║
        ║    🤖 Powered by Retrieval-Augmented Generation (RAG)       ║
        ║    📚 Vector Database + AI Language Models                   ║
        ║    🎯 Personalized Travel Itineraries for Pakistan          ║
        ║                                                              ║
        ╚══════════════════════════════════════════════════════════════╝
        """
        print(banner)
    
    def check_python_version(self):
        """Check if Python version is compatible."""
        print("🐍 Checking Python version...")
        
        if sys.version_info < (3, 8):
            print("❌ Error: Python 3.8 or higher is required!")
            print(f"   Current version: {sys.version}")
            sys.exit(1)
        else:
            print(f"✅ Python version: {sys.version.split()[0]}")
    
    def check_dependencies(self):
        """Check if all required packages are installed."""
        print("\n📦 Checking dependencies...")
        
        missing_packages = []
        
        for package in self.required_packages:
            try:
                # Special handling for packages with different import names
                if package == 'python-dotenv':
                    import dotenv
                elif package == 'google-generativeai':
                    import google.generativeai
                elif package == 'sentence-transformers':
                    import sentence_transformers
                else:
                    importlib.import_module(package)
                
                print(f"✅ {package}")
                
            except ImportError:
                print(f"❌ {package} - Not installed")
                missing_packages.append(package)
        
        if missing_packages:
            print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
            print("📥 Install missing packages with:")
            print(f"   pip install {' '.join(missing_packages)}")
            print("\n💡 Or install all requirements with:")
            print("   pip install -r requirements.txt")
            
            install_choice = input("\n🤔 Would you like to install missing packages now? (y/n): ").lower()
            if install_choice in ['y', 'yes']:
                self.install_dependencies(missing_packages)
            else:
                print("❌ Cannot proceed without required dependencies.")
                sys.exit(1)
    
    def install_dependencies(self, packages):
        """Install missing dependencies."""
        print(f"\n📥 Installing {len(packages)} missing packages...")
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--upgrade", "pip"
            ])
            
            subprocess.check_call([
                sys.executable, "-m", "pip", "install"
            ] + packages)
            
            print("✅ All packages installed successfully!")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error installing packages: {e}")
            print("💡 Try installing manually with: pip install -r requirements.txt")
            sys.exit(1)
    
    def check_environment_variables(self):
        """Check for required environment variables."""
        print("\n🔐 Checking environment variables...")
        
        # Load environment variables
        load_dotenv()
        
        cohere_key = os.getenv('COHERE_API_KEY')
        gemini_key = os.getenv('GEMINI_API_KEY')
        
        if not cohere_key or cohere_key == 'your_cohere_api_key_here':
            print("⚠️  COHERE_API_KEY not found or not set")
            print("📋 Get your API key from: https://dashboard.cohere.ai/")
            
        if not gemini_key or gemini_key == 'your_gemini_api_key_here':
            print("⚠️  GEMINI_API_KEY not found or not set")
            print("📋 Get your API key from: https://makersuite.google.com/app/apikey")
        
        if (not cohere_key or cohere_key == 'your_cohere_api_key_here') and \
           (not gemini_key or gemini_key == 'your_gemini_api_key_here'):
            print("\n❌ No valid API keys found!")
            print("💡 Please set up your API keys in the .env file")
            
            setup_choice = input("🤔 Would you like to set up API keys now? (y/n): ").lower()
            if setup_choice in ['y', 'yes']:
                self.setup_api_keys()
            else:
                print("⚠️  Warning: Application may not work without API keys")
        else:
            print("✅ API keys configured")
    
    def setup_api_keys(self):
        """Interactive setup for API keys."""
        print("\n🔧 Setting up API keys...")
        
        env_file = Path('.env')
        
        # Read existing .env file or create template
        if env_file.exists():
            with open(env_file, 'r') as f:
                content = f.read()
        else:
            content = """# API Keys for the RAG-powered Travel Assistant
# Get your Cohere API key from: https://dashboard.cohere.ai/
COHERE_API_KEY=your_cohere_api_key_h

# Get your Gemini API key from: https://makersuite.google.com/app/apikey
GEMINI_API_KEY="AVaHvSiqrmLl235Ox7u551hiuhEECbQqK0XRMhlD"

# Optional: OpenAI API key (if you want to add OpenAI support)
# OPENAI_API_KEY="AIzaSyDIDO1gJQ4050A6kvj_iZirlrFLnvA7Nmw"
"""
        
        print("📝 Enter your API keys (press Enter to skip):")
        
        # Get Cohere API key
        cohere_key = input("🔑 Cohere API Key: ").strip()
        if cohere_key:
            content = content.replace('COHERE_API_KEY=your_cohere_api_key_here', 
                                    f'COHERE_API_KEY={cohere_key}')
        
        # Get Gemini API key
        gemini_key = input("🔑 Gemini API Key: ").strip()
        if gemini_key:
            content = content.replace('GEMINI_API_KEY=your_gemini_api_key_here', 
                                    f'GEMINI_API_KEY={gemini_key}')
        
        # Write updated .env file
        with open(env_file, 'w') as f:
            f.write(content)
        
        print("✅ API keys saved to .env file")
        
        # Reload environment variables
        load_dotenv()
    
    def check_project_structure(self):
        """Check if all required files exist."""
        print("\n📁 Checking project structure...")
        
        required_files = [
            'MusafirAI.py',
            'rag_system.py',
            'vector_store_manager.py',
            'document_loader.py',
            'requirements.txt'
        ]
        
        missing_files = []
        
        for file in required_files:
            if os.path.exists(file):
                print(f"✅ {file}")
            else:
                print(f"❌ {file} - Missing")
                missing_files.append(file)
        
        if missing_files:
            print(f"\n⚠️  Missing files: {', '.join(missing_files)}")
            print("💡 Make sure all project files are in the current directory")
            
            continue_choice = input("🤔 Continue anyway? (y/n): ").lower()
            if continue_choice not in ['y', 'yes']:
                sys.exit(1)
    
    def initialize_rag_system(self):
    """Initialize the RAG system components with proper embeddings."""
    print("\n🤖 Initializing RAG system...")
    
    try:
        # Initialize embeddings with fallback
        try:
            from langchain.embeddings import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            print("✅ Using HuggingFace embeddings")
        except ImportError:
            try:
                from langchain.embeddings import CohereEmbeddings
                embeddings = CohereEmbeddings(cohere_api_key=os.getenv("COHERE_API_KEY"))
                print("⚠️ Using Cohere embeddings fallback")
            except ImportError:
                print("❌ Embeddings system failed to initialize")
                raise

        print("📚 Setting up vector database...")
        print("🔍 Loading travel documents...")
        
        # Initialize vector store (FAISS example)
        from langchain.vectorstores import FAISS
        vector_store_path = "faiss_index"
        
        if os.path.exists(vector_store_path):
            print("💾 Loading existing vector store")
            vector_store = FAISS.load_local(vector_store_path, embeddings)
        else:
            print("🆕 Creating new vector store")
            # Add your document loading and processing logic here
            # Example:
            from langchain.document_loaders import DirectoryLoader
            loader = DirectoryLoader('travel_docs/', glob="*.pdf")
            documents = loader.load()
            vector_store = FAISS.from_documents(documents, embeddings)
            vector_store.save_local(vector_store_path)
            # For now we'll create an empty index
            # from langchain.schema import Document
            # vector_store = FAISS.from_documents([Document(page_content="")], embeddings)
            # vector_store.save_local(vector_store_path)
        
        print("⚡ Initializing LangChain components...")
        self.retriever = vector_store.as_retriever()
        print("✅ RAG system initialized successfully")
        
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        # Add specific error handling
        if "Could not import sentence_transformers" in str(e):
            print("💡 Please ensure langchain and transformers are installed")
        raise
    
    def run_streamlit_app(self):
        """Launch the Streamlit application."""
        print(f"\n🚀 Starting {self.app_name}...")
        print("🌐 The application will open in your default browser")
        print("🔗 If it doesn't open automatically, go to: http://localhost:8501")
        print("\n💡 To stop the application, press Ctrl+C in this terminal")
        
        try:
            # Run the Streamlit app
            subprocess.run([
                sys.executable, "-m", "streamlit", "run", self.app_file,
                "--server.address", "localhost",
                "--server.port", "8501",
                "--server.headless", "false",
                "--browser.gatherUsageStats", "false"
            ])
            
        except KeyboardInterrupt:
            print("\n\n👋 Application stopped by user")
        except Exception as e:
            print(f"\n❌ Error running application: {e}")
            print("💡 Try running manually with: streamlit run MusafirAI.py")
    
    def run(self):
        """Main application runner."""
        self.print_banner()
        
        try:
            # System checks
            self.check_python_version()
            self.check_dependencies()
            self.check_project_structure()
            self.check_environment_variables()
            self.initialize_rag_system()
            
            print("\n" + "="*60)
            print("🎉 All system checks passed!")
            print("="*60)
            
            # Start the application
            self.run_streamlit_app()
            
        except KeyboardInterrupt:
            print("\n\n👋 Setup interrupted by user")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            print("💡 Please check the error message and try again")
            sys.exit(1)

def main():
    """Main entry point."""
    runner = MusafirAIRunner()
    runner.run()

if __name__ == "__main__":
    main()
