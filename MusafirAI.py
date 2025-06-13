import streamlit as st
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from rag_system import TravelRAGSystem, create_rag_system
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.write("Environment Variables:")
st.write(f"COHERE_API_KEY exists: {os.getenv('COHERE_API_KEY') is not None}")
st.write(f"GEMINI_API_KEY exists: {os.getenv('GEMINI_API_KEY') is not None}")

# Page configuration
st.set_page_config(
    page_title="MusafirAI - RAG-Powered Travel Planner",
    page_icon="🧳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #2c5f2d;
        margin: 1rem 0;
    }
    
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    .rag-info {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

class StreamlitTravelApp:
    """Streamlit application for RAG-powered travel planning."""
    
    def __init__(self):
        self.rag_system = None
        self.initialize_session_state()
        self.setup_rag_system()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        default_values = {
            'current_step': 0,
            'responses': {},
            'itinerary_generated': False,
            'generated_itinerary': '',
            'source_documents': []
        }
        
        for key, value in default_values.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    @st.cache_resource
    def setup_rag_system(_self):
        """Setup and cache the RAG system."""
        try:
            with st.spinner("🔧 Initializing RAG-powered AI system..."):
                rag_system = create_rag_system("cohere")
                st.success("✅ RAG system initialized successfully!")
                return rag_system
        except Exception as e:
            st.error(f"❌ Failed to initialize RAG system: {str(e)}")
            st.info("Please check your API keys in the .env file")
            return None
    
    def render_header(self):
        """Render the application header."""
        st.markdown('<h1 class="main-header">🧳 MusafirAI</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">RAG-Powered AI Travel Itinerary Generator for Pakistan</p>', unsafe_allow_html=True)
        
        # RAG System Info
        st.markdown("""
        <div class="rag-info">
            <h3>🤖 Powered by RAG Technology</h3>
            <p>This application uses Retrieval-Augmented Generation (RAG) with:</p>
            <ul>
                <li>📚 Vector database with travel documents</li>
                <li>🔍 Intelligent document retrieval</li>
                <li>🧠 AI-powered itinerary generation</li>
                <li>⚡ LangChain integration</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the sidebar with system information and controls."""
        st.sidebar.title("🔧 System Information")
        
        if self.rag_system:
            system_info = self.rag_system.get_system_info()
            st.sidebar.success("✅ RAG System Active")
            st.sidebar.write(f"**LLM Provider:** {system_info['llm_provider'].title()}")
            st.sidebar.write(f"**Vector Store:** {system_info['vector_store_info']['store_type'].title()}")
            
            if 'document_count' in system_info['vector_store_info']:
                st.sidebar.write(f"**Documents:** {system_info['vector_store_info']['document_count']}")
        else:
            st.sidebar.error("❌ RAG System Not Available")
        
        # Add system controls
        st.sidebar.title("🛠️ System Controls")
        
        if st.sidebar.button("🔄 Refresh Knowledge Base"):
            if self.rag_system:
                with st.spinner("Updating knowledge base..."):
                    self.rag_system.update_knowledge_base(force_recreate=True)
                st.sidebar.success("Knowledge base updated!")
        
        if st.sidebar.button("🧪 Test RAG System"):
            self.test_rag_system()
            
        # NEW: Document upload section
        st.sidebar.title("📤 Upload Travel Documents")
        st.sidebar.info("Add custom travel guides to enhance the AI's knowledge")

        # Create a unique container for the file uploader
        upload_container = st.sidebar.container()
    
        # Display uploaded files in session state
        if 'uploaded_files' in st.session_state and st.session_state.uploaded_files:
            st.sidebar.write("**Selected files:**")
            for file in st.session_state.uploaded_files:
                st.sidebar.markdown(f"- `{file.name}`")
        
        uploaded_files = st.sidebar.file_uploader(
            "Choose text files to add to the knowledge base",
            type=["txt"],
            accept_multiple_files=True,
            key="file_uploader"  # Use unique key
        )

        # Store uploaded files in session state
        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files
        
        if st.sidebar.button("📥 Process Uploaded Files"):
            if 'uploaded_files' in st.session_state and st.session_state.uploaded_files:
                self.process_uploaded_files(st.session_state.uploaded_files)

                # Clear files after processing
                st.session_state.pop('uploaded_files', None)

                # Recreate the file uploader container to reset it
                upload_container.empty()
                st.rerun()
            else:
                st.sidebar.warning("Please upload at least one text file")
    
    def process_uploaded_files(self, uploaded_files):
        """Process and add uploaded files to the knowledge base."""
        if not self.rag_system:
            st.sidebar.error("RAG system not available. Cannot add documents.")
            return
            
        documents = []
        metadata_list = []
        processed_files = []

        # Create travel_documents directory if missing
        os.makedirs("travel_documents", exist_ok=True)
        
        for uploaded_file in uploaded_files:
            try:
                # Read file content
                content = uploaded_file.read().decode("utf-8")
                
                # NEW: Save file to travel_documents
                file_path = os.path.join("travel_documents", uploaded_file.name)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)

                # Create metadata
                filename = uploaded_file.name
                metadata = {
                    "source": f"travel_documents/{filename}",
                    "type": "user_uploaded"
                }
                
                # Add to processing lists
                documents.append(content)
                metadata_list.append(metadata)
                processed_files.append(filename)
                
            except Exception as e:
                st.sidebar.error(f"Error processing {uploaded_file.name}: {str(e)}")
        
        if documents:
            try:
                with st.spinner(f"Adding {len(documents)} documents to knowledge base..."):
                    self.rag_system.add_custom_documents(documents, metadata_list)
                
                st.sidebar.success(f"✅ Added {len(documents)} documents to knowledge base:")
                for filename in processed_files:
                    st.sidebar.markdown(f"- `{filename}`")
                
            except Exception as e:
                st.sidebar.error(f"❌ Failed to add documents: {str(e)}")
    
    def test_rag_system(self):
        """Test the RAG system functionality."""
        if not self.rag_system:
            st.sidebar.error("RAG system not available")
            return
        
        with st.sidebar:
            with st.spinner("Testing RAG system..."):
                # Test search functionality
                test_query = "best places to visit in Lahore"
                results = self.rag_system.search_destinations(test_query, k=3)
                
                if results:
                    st.success(f"✅ Found {len(results)} relevant documents")
                    with st.expander("View Test Results"):
                        for i, result in enumerate(results[:2]):
                            st.write(f"**Result {i+1}:**")
                            st.write(result['content'][:200] + "...")
                            st.write(f"*Relevance Score: {result['relevance_score']:.3f}*")
                else:
                    st.error("❌ No results found")
    
    def render_travel_form(self):
        """Render the travel planning form."""
        st.markdown('<h2 class="sub-header">📋 Plan Your Journey</h2>', unsafe_allow_html=True)
        
        with st.form("travel_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Destination Details")
                
                # Locations with suggestions
                st.write("**Select destinations to visit:**")
                pakistan_cities = [
                    "Lahore", "Islamabad", "Karachi", "Multan", "Faisalabad",
                    "Peshawar", "Quetta", "Rawalpindi", "Sialkot", "Gujranwala",
                    "Hunza", "Skardu", "Gilgit", "Murree", "Nathia Gali"
                ]
                
                selected_locations = st.multiselect(
                    "Choose your destinations:",
                    pakistan_cities,
                    default=st.session_state.responses.get('locations', [])
                )
                
                start_location = st.selectbox(
                    "Starting location:",
                    [""] + pakistan_cities,
                    index=0 if not st.session_state.responses.get('start_location') 
                    else pakistan_cities.index(st.session_state.responses['start_location']) + 1
                )
                
                start_date = st.date_input(
                    "Start date:",
                    value=st.session_state.responses.get('start_date', datetime.now().date())
                )
                
                nights = st.slider(
                    "Number of nights:",
                    min_value=1, max_value=30,
                    value=st.session_state.responses.get('nights', 7)
                )
            
            with col2:
                st.subheader("🏨 Preferences")
                
                accommodation = st.selectbox(
                    "Accommodation type:",
                    ["Budget", "Mid-range", "Luxury", "Mixed"],
                    index=["Budget", "Mid-range", "Luxury", "Mixed"].index(
                        st.session_state.responses.get('accommodation', 'Mid-range')
                    )
                )
                
                trip_type = st.selectbox(
                    "Trip style:",
                    ["Cultural", "Adventure", "Relaxed", "Family", "Business", "Photography"],
                    index=["Cultural", "Adventure", "Relaxed", "Family", "Business", "Photography"].index(
                        st.session_state.responses.get('trip_type', 'Cultural')
                    )
                )
                
                group_size = st.number_input(
                    "Group size:",
                    min_value=1, max_value=50,
                    value=st.session_state.responses.get('group_size', 2)
                )
                
                budget = st.selectbox(
                    "Budget range (per person):",
                    ["Under PKR 50,000", "PKR 50,000 - 100,000", 
                     "PKR 100,000 - 200,000", "Above PKR 200,000"],
                    index=0
                )
                
                interests = st.multiselect(
                    "Interests:",
                    ["History", "Food", "Shopping", "Nature", "Museums", 
                     "Religious Sites", "Adventure Sports", "Photography"],
                    default=st.session_state.responses.get('interests', [])
                )
            
            # Submit button
            submitted = st.form_submit_button(
                "🚀 Generate RAG-Powered Itinerary",
                use_container_width=True
            )
            
            if submitted:
                if not selected_locations:
                    st.error("Please select at least one destination!")
                elif not start_location:
                    st.error("Please select a starting location!")
                else:
                    # Store responses
                    st.session_state.responses = {
                        'locations': selected_locations,
                        'start_location': start_location,
                        'start_date': start_date,
                        'nights': nights,
                        'accommodation': accommodation,
                        'trip_type': trip_type,
                        'group_size': group_size,
                        'budget': budget,
                        'interests': interests
                    }
                    
                    # Generate itinerary
                    self.generate_itinerary()
    
    def generate_itinerary(self):
        """Generate travel itinerary using RAG system."""
        if not self.rag_system:
            st.error("❌ RAG system not available. Please check your API configuration.")
            return
        
        with st.spinner("🤖 RAG system is working... Retrieving relevant travel information and generating your personalized itinerary..."):
            try:
                # Generate itinerary using RAG
                result = self.rag_system.generate_itinerary(st.session_state.responses)
                
                if result['success']:
                    st.session_state.generated_itinerary = result['itinerary']
                    st.session_state.source_documents = result['source_documents']
                    st.session_state.itinerary_generated = True
                    
                    # Show success message
                    st.markdown("""
                    <div class="success-box">
                        <h3>🎉 Your RAG-Powered Itinerary is Ready!</h3>
                        <p>Generated using intelligent document retrieval and AI processing</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                else:
                    st.error(f"❌ Failed to generate itinerary: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                st.error(f"❌ Error generating itinerary: {str(e)}")
    
    def render_itinerary_results(self):
        """Render the generated itinerary and source information."""
        if not st.session_state.itinerary_generated:
            return
        
        st.markdown('<h2 class="sub-header">🗺️ Your Personalized Itinerary</h2>', unsafe_allow_html=True)
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📋 Itinerary", "📚 Sources Used", "📄 Download"])
        
        with tab1:
            st.markdown("### Generated Travel Plan")
            st.markdown(st.session_state.generated_itinerary)
        
        with tab2:
            st.markdown("### 🔍 RAG Sources Used")
            st.info("These are the travel documents that were retrieved and used to generate your itinerary:")
            
            for i, doc in enumerate(st.session_state.source_documents):
                with st.expander(f"Source {i+1}: {doc['metadata'].get('source', 'Unknown')}"):
                    st.write("**Content Preview:**")
                    st.write(doc['content'])
                    st.write("**Metadata:**")
                    st.json(doc['metadata'])
        
        with tab3:
            st.markdown("### 📥 Download Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📄 Download PDF", use_container_width=True):
                    pdf_buffer = self.generate_pdf()
                    if pdf_buffer:
                        st.download_button(
                            label="📁 Save PDF File",
                            data=pdf_buffer,
                            file_name=f"travel_itinerary_{datetime.now().strftime('%Y%m%d')}.pdf",
                            mime="application/pdf"
                        )
            
            with col2:
                if st.button("📝 Download Text", use_container_width=True):
                    text_content = f"""
TRAVEL ITINERARY
Generated by MusafirAI RAG System
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

{st.session_state.generated_itinerary}

---
This itinerary was generated using RAG (Retrieval-Augmented Generation) technology,
combining AI with real travel data from our knowledge base.
                    """
                    st.download_button(
                        label="📁 Save Text File",
                        data=text_content,
                        file_name=f"travel_itinerary_{datetime.now().strftime('%Y%m%d')}.txt",
                        mime="text/plain"
                    )
    
    def generate_pdf(self) -> bytes:
        """Generate PDF of the itinerary."""
        try:
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                textColor='#1f4e79'
            )
            story.append(Paragraph("🧳 MusafirAI Travel Itinerary", title_style))
            story.append(Spacer(1, 12))
            
            # Subtitle
            subtitle_style = ParagraphStyle(
                'CustomSubtitle',
                parent=styles['Heading2'],
                fontSize=14,
                spaceAfter=20,
                textColor='#666666'
            )
            story.append(Paragraph(f"Generated on {datetime.now().strftime('%B %d, %Y')}", subtitle_style))
            story.append(Paragraph("Powered by RAG (Retrieval-Augmented Generation)", subtitle_style))
            story.append(Spacer(1, 20))
            
            # Itinerary content
            content_style = ParagraphStyle(
                'Content',
                parent=styles['Normal'],
                fontSize=11,
                spaceAfter=12,
                leading=14
            )
            
            # Split content into paragraphs
            paragraphs = st.session_state.generated_itinerary.split('\n\n')
            for para in paragraphs:
                if para.strip():
                    story.append(Paragraph(para.strip(), content_style))
                    story.append(Spacer(1, 8))
            
            # Footer
            story.append(Spacer(1, 30))
            footer_style = ParagraphStyle(
                'Footer',
                parent=styles['Normal'],
                fontSize=9,
                textColor='#888888'
            )
            story.append(Paragraph(
                "This itinerary was generated using RAG technology, combining AI with curated travel data.",
                footer_style
            ))
            
            doc.build(story)
            buffer.seek(0)
            return buffer.getvalue()
            
        except Exception as e:
            st.error(f"Error generating PDF: {str(e)}")
            return None
    
    def render_footer(self):
        """Render the application footer."""
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; padding: 2rem;">
            <p><strong>MusafirAI</strong> - RAG-Powered Travel Planning Assistant</p>
            <p>Built with ❤️ using Streamlit, LangChain, Chroma DB, and Cohere AI</p>
            <p>🔧 <strong>RAG Technology:</strong> Retrieval-Augmented Generation for intelligent travel recommendations</p>
        </div>
        """, unsafe_allow_html=True)
    
    def run(self):
        """Run the Streamlit application."""
        self.rag_system = self.setup_rag_system()
        
        self.render_header()
        self.render_sidebar()
        self.render_travel_form()
        self.render_itinerary_results()
        self.render_footer()

# Main execution
if __name__ == "__main__":
    app = StreamlitTravelApp()
    app.run()
