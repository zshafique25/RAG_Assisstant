# 🧳 MusafirAI - RAG-Powered Travel Assistant

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-00A67E?style=flat)](https://python.langchain.com/)
[![Cohere](https://img.shields.io/badge/Cohere-FFFFFF?style=flat&logo=cohere&logoColor=black)](https://cohere.com/)
[![Gemini](https://img.shields.io/badge/Gemini-4285F4?style=flat&logo=google&logoColor=white)](https://gemini.google.com/)

MusafirAI is a Retrieval-Augmented Generation (RAG) powered travel assistant that helps users create personalized travel itineraries for Pakistan. It combines the power of vector databases with large language models to provide contextually relevant travel recommendations.

![Capture1](https://github.com/user-attachments/assets/d3e8bbf4-43aa-45ef-a3de-c6d37636049d)

## System Architecture

#### Component Architecture
![deepseek_mermaid_20250701_b400eb](https://github.com/user-attachments/assets/600a0fbe-3af6-4b6f-829e-3cbde742708a)

#### Workflow Sequence
![deepseek_mermaid_20250701_8eb6b7](https://github.com/user-attachments/assets/8ff53895-1f03-476d-8263-45bd81a25d90)

## Features

- 🗺️ **Intelligent Itinerary Generation**: Create day-by-day travel plans based on user preferences
- 🔍 **Document Retrieval**: Retrieve relevant travel information from curated knowledge base
- 🤖 **Multiple LLM Support**: Currently supports Cohere and Gemini (with easy extension to others)
- 🧠 **Vector Database**: Uses Chroma DB or FAISS for efficient document retrieval
- 💻 **Streamlit Web Interface**: User-friendly web application for itinerary planning
- 📥 **Document Upload**: Add custom travel guides to enhance the knowledge base
- 📄 **PDF Export**: Download itineraries as professional PDF documents

## Technology Stack

- **Python**: Primary programming language
- **LangChain**: Framework for developing applications powered by language models
- **Chroma DB/FAISS**: Vector databases for document storage and retrieval
- **Cohere/Gemini**: Large language models for itinerary generation
- **Streamlit**: For building and deploying the web application
- **Sentence Transformers**: For document embeddings
- **ReportLab**: For PDF generation

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/zshafique25/RAG_Assisstant.git
   cd RAG_Assisstant
   ```

2. **Create virtual environment (recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   
   Create `.env` file in project root:
   ```text
   COHERE_API_KEY=your_cohere_api_key_here
   GEMINI_API_KEY=your_gemini_api_key_here
   ```
   
   Get API keys from:
   - [Cohere](https://cohere.com/)
   - [Gemini](https://gemini.google.com/)

5. **Prepare travel documents**:
   
   Place your travel guide text files in `travel_documents` directory
   
   Example files should include:
   - `lahore_guide.txt`
   - `islamabad_guide.txt`
   - `northern_areas.txt`

## Usage

1. **Run the application**:
   ```bash
   python run_app.py
   ```
   This will perform system checks and launch the Streamlit application

2. **Access the web interface**:
   - After running the command, the app will open in your browser at `http://localhost:8501`
   - If it doesn't open automatically, manually navigate to the URL

3. **Generate an itinerary**:
   - Fill in travel preferences (destinations, dates, budget, etc.)
   - Click "Generate RAG-Powered Itinerary"
   - View and download your personalized travel plan

## Project Structure

```
RAG_Assisstant/
├── travel_documents/          # Directory for travel guide text files
├── vector_store/              # Vector database storage (auto-generated)
├── .gitignore
├── requirements.txt
├── runtime.txt                # Python version specification
├── run_app.py                 # Main runner script
├── MusafirAI.py               # Streamlit application
├── rag_system.py              # RAG system implementation
├── vector_store_manager.py    # Vector database management
├── document_loader.py         # Document loading and processing
└── README.md
```

## Key Components

### Vector Store Manager (`vector_store_manager.py`)
- Manages Chroma/FAISS vector databases
- Handles document storage and retrieval
- Supports adding new travel documents

### RAG System (`rag_system.py`)
- Orchestrates retrieval and generation process
- Integrates with Cohere/Gemini LLMs
- Generates personalized itineraries
- Implements prompt engineering for structured outputs

### Streamlit App (`MusafirAI.py`)
- User-friendly web interface
- PDF itinerary generation
- Source document inspection
- Vector store management controls

### Document Loader (`document_loader.py`)
- Processes travel documents
- Splits text into optimized chunks
- Adds metadata based on filenames

## Customization

### Add new destinations
Add text files to `travel_documents` with naming convention: `[destination]_guide.txt`

### Change LLM provider
Modify `create_rag_system()` call in `MusafirAI.py`:
```python
# Change "cohere" to "gemini"
rag_system = create_rag_system("cohere") 
```

### Modify itinerary style
Edit prompt template in `rag_system.py` to change output format

## License
Distributed under the MIT License. See `LICENSE` for more information.
