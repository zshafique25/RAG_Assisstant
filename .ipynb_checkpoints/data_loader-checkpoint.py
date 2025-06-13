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
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
    def create_sample_documents(self):
        """Create sample travel documents for Pakistan."""
        os.makedirs(self.documents_dir, exist_ok=True)
        
        # Sample travel documents
        documents = {
            "lahore_guide.txt": """
            Lahore Travel Guide
            
            Lahore is the cultural capital of Pakistan and the second-largest city. Known for its rich history, magnificent architecture, and vibrant food scene.
            
            Top Attractions:
            - Badshahi Mosque: One of the largest mosques in the world, built in 1673
            - Lahore Fort (Shahi Qila): A UNESCO World Heritage site with stunning Mughal architecture
            - Shalimar Gardens: Beautiful Mughal gardens with fountains and pavilions
            - Wazir Khan Mosque: Famous for its intricate tile work and frescoes
            - Anarkali Bazaar: Historic market for shopping and local crafts
            
            Best Time to Visit: October to March when weather is pleasant
            
            Local Cuisine:
            - Nihari: Slow-cooked beef stew, best eaten for breakfast
            - Haleem: Lentil and meat porridge
            - Kulfi: Traditional ice cream
            - Lassi: Yogurt-based drink
            
            Accommodation Options:
            - Luxury: Pearl Continental Hotel, Avari Hotel
            - Mid-range: Hotel One, Shelton Hotel
            - Budget: YMCA, local guesthouses
            
            Transportation:
            - Metro Bus Service connects major areas
            - Rickshaws and taxis available
            - Orange Line Metro Train operational
            
            Travel Tips:
            - Dress modestly when visiting religious sites
            - Bargain in local markets
            - Try street food but choose busy stalls
            - Learn basic Urdu phrases
            """,
            
            "islamabad_guide.txt": """
            Islamabad Travel Guide
            
            Islamabad is the capital city of Pakistan, known for its modern architecture, clean environment, and natural beauty.
            
            Top Attractions:
            - Faisal Mosque: One of the largest mosques in the world with unique contemporary design
            - Pakistan Monument: National monument representing four provinces
            - Margalla Hills: Perfect for hiking and nature walks
            - Lok Virsa Museum: Showcases Pakistani culture and heritage
            - Centaurus Mall: Modern shopping and entertainment complex
            
            Best Time to Visit: March to May and September to November
            
            Local Cuisine:
            - Chapli Kebab: Spiced ground meat patties
            - Karahi: Wok-cooked curry with meat or vegetables
            - Biryani: Fragrant rice dish with meat
            - Falooda: Sweet dessert drink
            
            Accommodation Options:
            - Luxury: Serena Hotel, Marriott Hotel
            - Mid-range: Best Western, Envoy Continental
            - Budget: Backpackers Inn, local hotels
            
            Transportation:
            - Metro Bus connects to Rawalpindi
            - Uber and Careem available
            - Local bus service
            
            Travel Tips:
            - Carry water bottle for hiking
            - Visit Margalla Hills early morning
            - Respect local customs and dress codes
            - Keep emergency contacts handy
            """,
            
            "karachi_guide.txt": """
            Karachi Travel Guide
            
            Karachi is Pakistan's largest city and economic hub, known for its beaches, diverse culture, and vibrant nightlife.
            
            Top Attractions:
            - Clifton Beach: Popular beach destination
            - Quaid-e-Azam's Mausoleum: Tomb of Pakistan's founder
            - Mohatta Palace: Beautiful palace turned museum
            - Empress Market: Historic market building
            - Pakistan Maritime Museum: Naval history and artifacts
            
            Best Time to Visit: November to February for pleasant weather
            
            Local Cuisine:
            - Biryani: Karachi-style spiced rice with meat
            - Nihari: Slow-cooked curry
            - Seekh Kebab: Grilled meat skewers
            - Falooda: Cold dessert drink
            - Street food at Burns Road
            
            Accommodation Options:
            - Luxury: Pearl Continental, Marriott
            - Mid-range: Regent Plaza, Carlton Hotel
            - Budget: YMCA, local guesthouses
            
            Transportation:
            - Uber and Careem widely available
            - Local bus service (BRT under construction)
            - Rickshaws for short distances
            
            Travel Tips:
            - Be cautious with belongings in crowded areas
            - Try local street food but choose reputable vendors
            - Respect local customs and dress modestly
            - Stay hydrated in hot weather
            """,
            
            "northern_areas_guide.txt": """
            Northern Areas of Pakistan Travel Guide
            
            The northern regions include Gilgit-Baltistan and parts of KPK, famous for stunning mountains, glaciers, and valleys.
            
            Top Destinations:
            - Hunza Valley: Beautiful valley with apricot orchards and mountain views
            - Skardu: Gateway to K2 and other peaks
            - Fairy Meadows: Base camp for Nanga Parbat
            - Deosai Plains: High-altitude plateau known as 'Land of Giants'
            - Naltar Valley: Famous for colorful lakes
            
            Best Time to Visit: May to September for accessibility
            
            Adventure Activities:
            - Trekking to K2 Base Camp
            - Mountaineering
            - White water rafting
            - Paragliding
            - Photography tours
            
            Accommodation Options:
            - Hunza: Serena Hotel, Eagle's Nest Hotel
            - Skardu: Concordia Hotel, local guesthouses
            - Camping options available
            
            Transportation:
            - Karakoram Highway (KKH) connects major areas
            - Domestic flights to Skardu and Gilgit
            - 4WD vehicles recommended for rough terrain
            
            Travel Tips:
            - Carry warm clothing even in summer
            - Respect local culture and traditions
            - Carry cash as ATMs are limited
            - Book accommodation in advance during peak season
            - Get proper permits for restricted areas
            """,
            
            "punjab_cultural_info.txt": """
            Punjab Cultural Information
            
            Punjab is the most populous province of Pakistan, known for its rich culture, agriculture, and historical significance.
            
            Cultural Highlights:
            - Punjabi language and literature
            - Traditional folk music and dance (Bhangra, Giddha)
            - Sufi poetry and shrines
            - Handicrafts and textiles
            - Agricultural festivals
            
            Major Cities:
            - Lahore: Cultural capital
            - Faisalabad: Textile hub
            - Multan: City of Saints
            - Rawalpindi: Twin city of Islamabad
            - Sialkot: Sports goods manufacturing
            
            Festivals and Events:
            - Basant (Kite Flying Festival)
            - Shalimar Festival
            - Lok Virsa Folk Festival
            - Urs celebrations at Sufi shrines
            
            Traditional Cuisine:
            - Saag and Makki di Roti
            - Punjabi Karahi
            - Lassi and Buttermilk
            - Kulfi and Falooda
            - Traditional sweets like Gulab Jamun
            
            Best Time to Visit Punjab:
            - October to March for pleasant weather
            - April to June can be very hot
            - Monsoon season: July to September
            
            Transportation:
            - Well-connected by roads and railways
            - Domestic flights to major cities
            - Public transport and private vehicles
            
            Shopping:
            - Traditional textiles and fabrics
            - Handicrafts and pottery
            - Leather goods
            - Sports equipment (especially in Sialkot)
            """,
            
            "travel_tips_pakistan.txt": """
            General Travel Tips for Pakistan
            
            Visa and Documentation:
            - Tourist visa required for most countries
            - Passport valid for at least 6 months
            - Some areas require special permits
            
            Health and Safety:
            - Recommended vaccinations: Hepatitis A/B, Typhoid, routine vaccines
            - Drink bottled or filtered water
            - Be cautious with street food initially
            - Travel insurance recommended
            
            Cultural Etiquette:
            - Dress modestly, especially when visiting religious sites
            - Remove shoes when entering mosques
            - Respect local customs and traditions
            - Ask permission before photographing people
            - Use right hand for eating and greeting
            
            Money and Payments:
            - Pakistani Rupee (PKR) is the currency
            - Cash is preferred in most places
            - Credit cards accepted in major hotels and restaurants
            - ATMs available in cities but limited in remote areas
            
            Communication:
            - Urdu is the national language
            - English widely spoken in cities
            - Local languages vary by region
            - Internet and mobile coverage good in cities
            
            Transportation:
            - Domestic flights connect major cities
            - Train network covers most of the country
            - Buses and coaches for intercity travel
            - Uber and Careem available in major cities
            - Rent a car with driver for long trips
            
            Weather:
            - Hot summers (April to June)
            - Monsoon season (July to September)
            - Pleasant winters (October to March)
            - Northern areas have different climate patterns
            
            Shopping:
            - Bargaining is common in markets
            - Fixed prices in malls and branded stores
            - Best buys: textiles, handicrafts, carpets, leather goods
            - VAT refund available for tourists in some cases
            """
        }
        
        # Write documents to files
        for filename, content in documents.items():
            with open(os.path.join(self.documents_dir, filename), 'w', encoding='utf-8') as f:
                f.write(content)
        
        print(f"Created {len(documents)} sample travel documents")
    
    def load_documents(self) -> List[Document]:
        """Load all travel documents from the directory."""
        if not os.path.exists(self.documents_dir):
            print("Creating sample documents...")
            self.create_sample_documents()
        
        # Load documents
        loader = DirectoryLoader(
            self.documents_dir,
            glob="*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )
        
        documents = loader.load()
        print(f"Loaded {len(documents)} documents")
        
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