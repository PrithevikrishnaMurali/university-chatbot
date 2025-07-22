# University Recommendation Chatbot Backend
# Complete FastAPI + MongoDB + NLP Pipeline

import os
import json
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from enum import Enum
import uuid
import logging
from dataclasses import dataclass, asdict
import asyncio
from contextlib import asynccontextmanager
import urllib.parse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# FastAPI and dependencies
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# MongoDB
from motor.motor_asyncio import AsyncIOMotorClient
import pymongo

# NLP and ML
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Configuration
class Settings:
    MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    DATABASE_NAME = os.getenv("DATABASE_NAME", "University_data")
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"

settings = Settings()

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic Models
class UserPreferences(BaseModel):
    field_of_study: Optional[str] = None
    location_preference: Optional[str] = None
    budget_range: Optional[str] = None
    degree_level: Optional[str] = None
    gpa: Optional[float] = None
    test_scores: Optional[Dict[str, int]] = None
    international_student: Optional[bool] = None
    scholarship_needed: Optional[bool] = None
    preferred_size: Optional[str] = None
    preferred_setting: Optional[str] = None

class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    recommendations: Optional[List[Dict]] = None
    follow_up_questions: Optional[List[str]] = None
    conversation_stage: str

class University(BaseModel):
    name: str
    location: str
    type: str
    ranking: Optional[int] = None
    tuition_fees: Optional[Dict[str, float]] = None
    programs: List[str]
    admission_requirements: Dict[str, Any]
    scholarships: List[Dict[str, Any]]
    campus_size: Optional[str] = None
    setting: Optional[str] = None
    website: Optional[str] = None

class UserSession(BaseModel):
    session_id: str
    user_id: Optional[str] = None
    preferences: UserPreferences
    conversation_history: List[Dict[str, str]]
    current_stage: str
    created_at: datetime
    updated_at: datetime

# Conversation Stages
class ConversationStage(Enum):
    GREETING = "greeting"
    COLLECTING_PREFERENCES = "collecting_preferences"
    FIELD_OF_STUDY = "field_of_study"
    LOCATION = "location"
    BUDGET = "budget"
    DEGREE_LEVEL = "degree_level"
    ACADEMIC_PROFILE = "academic_profile"
    INTERNATIONAL_STATUS = "international_status"
    PROVIDING_RECOMMENDATIONS = "providing_recommendations"
    FOLLOW_UP = "follow_up"
    COMPLETED = "completed"

# Enhanced NLP Engine
class NLPEngine:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Enhanced field of study patterns - more comprehensive
        self.field_patterns = {
            'computer science': ['computer science', 'cs', 'computing', 'software engineering', 'programming', 'coding', 'software development', 'it', 'information technology'],
            'engineering': ['engineering', 'mechanical engineering', 'electrical engineering', 'civil engineering', 'chemical engineering', 'aerospace engineering', 'biomedical engineering'],
            'business': ['business', 'management', 'mba', 'business administration', 'commerce', 'marketing', 'finance', 'accounting', 'economics'],
            'medicine': ['medicine', 'medical', 'doctor', 'physician', 'healthcare', 'nursing', 'pharmacy', 'dentistry', 'veterinary'],
            'law': ['law', 'legal studies', 'jurisprudence', 'legal', 'lawyer', 'attorney'],
            'psychology': ['psychology', 'psychologist', 'mental health', 'behavioral science'],
            'biology': ['biology', 'life sciences', 'biotechnology', 'biochemistry', 'microbiology'],
            'chemistry': ['chemistry', 'chemical sciences'],
            'physics': ['physics', 'physical sciences', 'astronomy', 'astrophysics'],
            'mathematics': ['mathematics', 'math', 'statistics', 'applied mathematics'],
            'education': ['education', 'teaching', 'teacher', 'pedagogy'],
            'art': ['art', 'fine arts', 'visual arts', 'design', 'graphic design'],
            'literature': ['literature', 'english literature', 'creative writing', 'linguistics'],
            'history': ['history', 'historical studies', 'archaeology'],
            'political science': ['political science', 'politics', 'international relations', 'public policy']
        }
        
        # Canadian provinces for location matching
        self.canadian_provinces = [
            'british columbia', 'bc', 'new brunswick', 'nb', 'newfoundland', 'nl',
            'nova scotia', 'ns', 'ontario', 'on', 'prince edward island', 'pei',
            'quebec', 'qc', 'saskatchewan', 'sk', 'alberta', 'ab', 'manitoba', 'mb'
        ]
        
        # Enhanced intent patterns
        self.intent_patterns = {
            'greeting': [
                r'\b(hi|hello|hey|good morning|good afternoon|good evening|start|begin|help)\b'
            ],
            'field_of_study': [
                r'\b(study|major|field|course|program|subject|degree in)\b',
                r'\b(want to study|interested in|planning to study|looking for)\b'
            ],
            'location': [
                r'\b(location|place|city|province|country|where|in)\b',
                r'\b(prefer|want|like)\s+.*\b(location|place|city|province)\b'
            ],
            'budget': [
                r'\b(budget|cost|money|expensive|cheap|afford|financial|tuition|fees)\b',
                r'\b(how much|price|scholarship|funding)\b'
            ],
            'degree_level': [
                r'\b(bachelor|master|phd|doctorate|undergraduate|graduate|degree)\b'
            ],
            'yes': [r'\b(yes|yeah|yep|sure|ok|okay|definitely|absolutely)\b'],
            'no': [r'\b(no|nope|not|never|none)\b']
        }
    
    def extract_field_of_study(self, text: str) -> Optional[str]:
        """Enhanced field of study extraction"""
        text_lower = text.lower()
        
        # Check for direct matches first
        for field, patterns in self.field_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    return field
        
        # Check for partial matches or synonyms
        words = text_lower.split()
        for field, patterns in self.field_patterns.items():
            for pattern in patterns:
                pattern_words = pattern.split()
                if any(word in words for word in pattern_words):
                    return field
        
        return None
    
    def extract_location(self, text: str) -> Optional[str]:
        """Extract Canadian province from text"""
        text_lower = text.lower()
        
        for province in self.canadian_provinces:
            if province in text_lower:
                # Map abbreviations to full names
                province_map = {
                    'bc': 'British Columbia',
                    'nb': 'New Brunswick',
                    'nl': 'Newfoundland',
                    'ns': 'Nova Scotia',
                    'on': 'Ontario',
                    'pei': 'Prince Edward Island',
                    'qc': 'Quebec',
                    'sk': 'Saskatchewan',
                    'ab': 'Alberta',
                    'mb': 'Manitoba'
                }
                return province_map.get(province, province.title())
        
        # Check for "canada" or general location mentions
        if 'canada' in text_lower:
            return 'Canada'
        
        return None
    
    def extract_intent(self, text: str) -> str:
        """Extract user intent from text"""
        text = text.lower()
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return intent
        
        return 'general'
    
    def generate_response(self, intent: str, message: str, stage: str, preferences: UserPreferences, 
                         session_history: List[Dict[str, str]]) -> Dict:
        """Generate appropriate response based on intent, message, and context"""
        
        # Check if this is the very first interaction (empty history)
        if len(session_history) == 0:
            return {
                'response': "Hello! I'm here to help you find the perfect university for your studies in Canada. Let's start by learning more about what you're looking for. What field would you like to study?",
                'stage': ConversationStage.FIELD_OF_STUDY.value,
                'questions': []
            }
        
        # Handle field of study stage
        elif stage == ConversationStage.FIELD_OF_STUDY.value:
            field = self.extract_field_of_study(message)
            if field:
                return {
                    'response': f"Great! You're interested in {field}. Which Canadian province would you prefer to study in? For example: Ontario, British Columbia, Quebec, Alberta, etc.",
                    'stage': ConversationStage.LOCATION.value,
                    'questions': []
                }
            else:
                return {
                    'response': "I'd love to help you find the right field! Could you be more specific? For example: computer science, engineering, business, medicine, law, psychology, etc.",
                    'stage': ConversationStage.FIELD_OF_STUDY.value,
                    'questions': []
                }
        
        # Handle location stage
        elif stage == ConversationStage.LOCATION.value:
            location = self.extract_location(message)
            if location:
                return {
                    'response': f"Perfect! You're interested in studying in {location}. What's your budget range for tuition? (e.g., under $20,000, under $30,000, under $50,000, or no specific limit)",
                    'stage': ConversationStage.BUDGET.value,
                    'questions': []
                }
            else:
                return {
                    'response': "No worries about the location! What's your budget range for tuition? (e.g., under $20,000, under $30,000, under $50,000, or no specific limit)",
                    'stage': ConversationStage.BUDGET.value,
                    'questions': []
                }
        
        # Handle budget stage
        elif stage == ConversationStage.BUDGET.value:
            return {
                'response': "Thanks for that information! What level of degree are you looking for? Bachelor's, Master's, or PhD?",
                'stage': ConversationStage.DEGREE_LEVEL.value,
                'questions': []
            }
        
        # Handle degree level stage
        elif stage == ConversationStage.DEGREE_LEVEL.value:
            return {
                'response': "Perfect! Are you an international student? This helps me find universities with good international support and appropriate programs.",
                'stage': ConversationStage.INTERNATIONAL_STATUS.value,
                'questions': []
            }
        
        # Handle international status stage
        elif stage == ConversationStage.INTERNATIONAL_STATUS.value:
            return {
                'response': "Excellent! Based on your preferences, let me find some great university recommendations for you!",
                'stage': ConversationStage.PROVIDING_RECOMMENDATIONS.value,
                'questions': []
            }
        
        # Handle providing recommendations stage
        elif stage == ConversationStage.PROVIDING_RECOMMENDATIONS.value:
            return {
                'response': "Here are some university recommendations based on your preferences. Would you like more information about any of these universities or would you like to modify your search criteria?",
                'stage': ConversationStage.FOLLOW_UP.value,
                'questions': []
            }
        
        # Default response for other cases
        else:
            return {
                'response': "I understand. Let me help you find the best universities based on your preferences!",
                'stage': ConversationStage.PROVIDING_RECOMMENDATIONS.value,
                'questions': []
            }

# Enhanced Recommendation Engine
class RecommendationEngine:
    def __init__(self):
        self.universities = []
        
    async def load_universities(self, db):
        """Load universities from all province collections"""
        self.universities = []
        
        # List of province collections based on your actual database structure
        provinces = [
            'British columbia', 'New Brunswick', 'Newfoundland', 'Nova scotia',
            'Ontario', 'Prince Edward island', 'Quebec', 'Saskatchewan', 
            'alberta', 'manitoba'
        ]
        
        for province in provinces:
            try:
                # Use the collection name exactly as it appears in MongoDB
                cursor = db[province].find({})
                province_universities = await cursor.to_list(length=None)
                
                # Add province information to each university
                for uni in province_universities:
                    uni['province'] = province
                    self.universities.append(uni)
                
                logger.info(f"Loaded {len(province_universities)} universities from {province}")
            except Exception as e:
                logger.warning(f"Could not load universities from {province}: {e}")
        
        logger.info(f"Total universities loaded: {len(self.universities)}")
        
    def calculate_match_score(self, university: Dict, preferences: UserPreferences) -> float:
        """Calculate match score between university and user preferences"""
        score = 0.0
        max_score = 0.0
        
        # Field of study matching (40% weight)
        if preferences.field_of_study:
            max_score += 40
            # Check if university has programs field, otherwise use name for matching
            programs = university.get('programs', [])
            if not programs:
                # If no programs field, check the university name for field matches
                programs = [university.get('name', '')]
            
            if isinstance(programs, list):
                program_text = ' '.join(str(p) for p in programs).lower()
            else:
                program_text = str(programs).lower()
            
            if preferences.field_of_study.lower() in program_text:
                score += 40
            # Partial matching for common field keywords
            elif any(keyword in program_text for keyword in ['engineering', 'computer', 'business', 'science']):
                score += 20
        
        # Location matching (30% weight)
        if preferences.location_preference:
            max_score += 30
            # Check both province and city fields
            uni_province = university.get('province', '').lower()
            uni_city = university.get('city', '').lower()
            pref_location = preferences.location_preference.lower()
            
            if pref_location in uni_province or uni_province in pref_location:
                score += 30
            elif pref_location in uni_city or uni_city in pref_location:
                score += 20
        
        # University size preference (15% weight)
        if preferences.preferred_size:
            max_score += 15
            student_count = university.get('number_of_students', 0)
            if student_count:
                if preferences.preferred_size == 'small' and student_count < 10000:
                    score += 15
                elif preferences.preferred_size == 'medium' and 10000 <= student_count < 25000:
                    score += 15
                elif preferences.preferred_size == 'large' and student_count >= 25000:
                    score += 15
        
        # International student support (15% weight)
        if preferences.international_student is not None:
            max_score += 15
            # Assume larger universities have better international support
            student_count = university.get('number_of_students', 0)
            if preferences.international_student and student_count > 15000:
                score += 15
            elif not preferences.international_student:
                score += 15
        
        return (score / max_score * 100) if max_score > 0 else 60  # Default 60% if no specific criteria
    
    def _is_within_budget(self, tuition: float, budget_range: str) -> bool:
        """Check if tuition is within budget range"""
        if not tuition or tuition == 0:
            return True  # If no tuition info, assume it's within budget
        
        budget_limits = {
            'under 20000': 20000,
            'under 30000': 30000,
            'under 50000': 50000,
            'no limit': float('inf'),
            'no specific limit': float('inf')
        }
        
        limit = budget_limits.get(budget_range.lower(), float('inf'))
        return tuition <= limit
    
    async def get_recommendations(self, preferences: UserPreferences, limit: int = 5) -> List[Dict]:
        """Get university recommendations based on preferences"""
        if not self.universities:
            return []
        
        # Calculate match scores
        scored_universities = []
        for university in self.universities:
            score = self.calculate_match_score(university, preferences)
            university_with_score = university.copy()
            university_with_score['match_score'] = score
            
            # Clean the document for JSON serialization - remove ObjectId and other non-serializable fields
            university_with_score = self._clean_document(university_with_score)
            
            scored_universities.append(university_with_score)
        
        # Sort by score and return top recommendations
        scored_universities.sort(key=lambda x: x['match_score'], reverse=True)
        return scored_universities[:limit]
    
    def _clean_document(self, doc: Dict) -> Dict:
        """Clean MongoDB document for JSON serialization"""
        cleaned = {}
        for key, value in doc.items():
            if key == '_id':
                # Convert ObjectId to string or skip it
                continue
            elif hasattr(value, '__dict__') and hasattr(value, '__class__'):
                # Skip complex objects that can't be serialized
                continue
            elif isinstance(value, list):
                # Clean list items
                cleaned[key] = [self._clean_value(item) for item in value]
            else:
                cleaned[key] = self._clean_value(value)
        return cleaned
    
    def _clean_value(self, value):
        """Clean individual values for JSON serialization"""
        if hasattr(value, '__dict__') and hasattr(value, '__class__'):
            # Convert complex objects to string
            return str(value)
        return value

# Database Manager
class DatabaseManager:
    def __init__(self):
        self.client = None
        self.db = None
    
    async def connect(self):
        """Connect to MongoDB"""
        try:
            self.client = AsyncIOMotorClient(settings.MONGODB_URL)
            self.db = self.client[settings.DATABASE_NAME]
            # Test connection
            await self.client.admin.command('ping')
            logger.info(f"Connected to MongoDB successfully. Database: {settings.DATABASE_NAME}")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    async def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
    
    async def create_indexes(self):
        """Create database indexes for session management"""
        try:
            await self.db.sessions.create_index([("session_id", 1)])
            await self.db.sessions.create_index([("created_at", 1)], expireAfterSeconds=3600*24)  # 24 hours
            logger.info("Session indexes created successfully")
        except Exception as e:
            logger.warning(f"Could not create indexes: {e}")
    
    async def save_session(self, session: UserSession):
        """Save user session to database"""
        session_dict = {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "preferences": session.preferences.model_dump(),  # Updated from .dict()
            "conversation_history": session.conversation_history,
            "current_stage": session.current_stage,
            "created_at": session.created_at,
            "updated_at": session.updated_at
        }
        await self.db.sessions.replace_one(
            {"session_id": session.session_id},
            session_dict,
            upsert=True
        )
    
    async def get_session(self, session_id: str) -> Optional[UserSession]:
        """Get user session from database"""
        session_data = await self.db.sessions.find_one({"session_id": session_id})
        if session_data:
            return UserSession(
                session_id=session_data["session_id"],
                user_id=session_data.get("user_id"),
                preferences=UserPreferences(**session_data["preferences"]),
                conversation_history=session_data["conversation_history"],
                current_stage=session_data["current_stage"],
                created_at=session_data["created_at"],
                updated_at=session_data["updated_at"]
            )
        return None

# Chatbot Controller
class ChatbotController:
    def __init__(self, db_manager: DatabaseManager, recommendation_engine: RecommendationEngine):
        self.db_manager = db_manager
        self.recommendation_engine = recommendation_engine
        self.nlp_engine = NLPEngine()
    
    async def process_message(self, message: str, session_id: str = None) -> ChatResponse:
        """Process user message and generate response"""
        
        try:
            # Get or create session
            if session_id:
                session = await self.db_manager.get_session(session_id)
            else:
                session = None
            
            if not session:
                session_id = str(uuid.uuid4())
                session = UserSession(
                    session_id=session_id,
                    preferences=UserPreferences(),
                    conversation_history=[],
                    current_stage=ConversationStage.GREETING.value,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
            
            # Extract intent
            intent = self.nlp_engine.extract_intent(message)
            
            # Update preferences based on message and current stage
            await self._update_preferences(session.preferences, message, session.current_stage)
            
            # Generate response
            response_data = self.nlp_engine.generate_response(
                intent, message, session.current_stage, session.preferences, session.conversation_history
            )
            
            # Update session with user message first
            session.conversation_history.append({
                "user": message,
                "assistant": response_data['response'],
                "timestamp": datetime.utcnow().isoformat()
            })
            session.current_stage = response_data['stage']
            session.updated_at = datetime.utcnow()
            
            # Get recommendations if ready
            recommendations = []
            if session.current_stage == ConversationStage.PROVIDING_RECOMMENDATIONS.value:
                try:
                    recommendations = await self.recommendation_engine.get_recommendations(session.preferences)
                    logger.info(f"Generated {len(recommendations)} recommendations")
                except Exception as e:
                    logger.error(f"Error getting recommendations: {e}")
                    recommendations = []
            
            # Save session
            await self.db_manager.save_session(session)
            
            return ChatResponse(
                response=response_data['response'],
                session_id=session.session_id,
                recommendations=recommendations,
                follow_up_questions=response_data.get('questions', []),
                conversation_stage=session.current_stage
            )
            
        except Exception as e:
            logger.error(f"Error in process_message: {e}")
            # Return a safe error response
            return ChatResponse(
                response="I apologize, but I encountered an error. Let's try again. What would you like to know about universities?",
                session_id=session_id or str(uuid.uuid4()),
                recommendations=[],
                follow_up_questions=[],
                conversation_stage=ConversationStage.GREETING.value
            )
    
    async def _update_preferences(self, preferences: UserPreferences, message: str, current_stage: str):
        """Update user preferences based on message and conversation stage"""
        
        message_lower = message.lower().strip()
        
        try:
            # Update field of study
            if current_stage == ConversationStage.FIELD_OF_STUDY.value:
                field = self.nlp_engine.extract_field_of_study(message)
                if field:
                    preferences.field_of_study = field
                    logger.info(f"Updated field of study: {field}")
            
            # Update location
            elif current_stage == ConversationStage.LOCATION.value:
                location = self.nlp_engine.extract_location(message)
                if location:
                    preferences.location_preference = location
                    logger.info(f"Updated location: {location}")
            
            # Update budget
            elif current_stage == ConversationStage.BUDGET.value:
                budget_patterns = {
                    r'under.*?20': 'under 20000',
                    r'under.*?30': 'under 30000',
                    r'under.*?50': 'under 50000',
                    r'no.*?limit': 'no limit',
                    r'unlimited': 'no limit',
                    r'any.*?amount': 'no limit'
                }
                
                for pattern, budget_range in budget_patterns.items():
                    if re.search(pattern, message_lower):
                        preferences.budget_range = budget_range
                        logger.info(f"Updated budget: {budget_range}")
                        break
            
            # Update degree level
            elif current_stage == ConversationStage.DEGREE_LEVEL.value:
                if any(word in message_lower for word in ['bachelor', 'undergraduate']):
                    preferences.degree_level = 'bachelor'
                elif any(word in message_lower for word in ['master', 'graduate', 'msc', 'ma']):
                    preferences.degree_level = 'master'
                elif any(word in message_lower for word in ['phd', 'doctorate', 'doctoral']):
                    preferences.degree_level = 'phd'
                logger.info(f"Updated degree level: {preferences.degree_level}")
            
            # Update international status
            elif current_stage == ConversationStage.INTERNATIONAL_STATUS.value:
                if any(word in message_lower for word in ['yes', 'international', 'foreign', 'abroad', 'outside']):
                    preferences.international_student = True
                    logger.info("Updated international status: True")
                elif any(word in message_lower for word in ['no', 'domestic', 'local', 'canadian']):
                    preferences.international_student = False
                    logger.info("Updated international status: False")
                    
        except Exception as e:
            logger.error(f"Error updating preferences: {e}")
            # Continue without crashing

# Initialize components
db_manager = DatabaseManager()
recommendation_engine = RecommendationEngine()
chatbot_controller = ChatbotController(db_manager, recommendation_engine)

# Lifespan event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application lifespan events"""
    # Startup
    try:
        await db_manager.connect()
        await db_manager.create_indexes()
        await recommendation_engine.load_universities(db_manager.db)
        logger.info("Application started successfully")
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
    
    yield
    
    # Shutdown
    await db_manager.close()
    logger.info("Application shutdown complete")

# FastAPI App
app = FastAPI(
    title="University Recommendation Chatbot API",
    description="A comprehensive chatbot for university and course recommendations",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Routes
@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "University Recommendation Chatbot API", "version": "1.0.0"}

@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """Main chat endpoint"""
    try:
        response = await chatbot_controller.process_message(
            message.message, 
            message.session_id
        )
        
        # Ensure all data is JSON serializable
        if response.recommendations:
            # Clean recommendations data
            cleaned_recommendations = []
            for rec in response.recommendations:
                cleaned_rec = {}
                for key, value in rec.items():
                    if key == '_id':
                        continue  # Skip ObjectId
                    elif hasattr(value, '__dict__'):
                        cleaned_rec[key] = str(value)  # Convert complex objects to string
                    else:
                        cleaned_rec[key] = value
                cleaned_recommendations.append(cleaned_rec)
            response.recommendations = cleaned_recommendations
        
        return response
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        import traceback
        traceback.print_exc()
        
        # Return a safe error response
        return ChatResponse(
            response="I apologize, but I encountered an error while processing your request. Please try again.",
            session_id=message.session_id or str(uuid.uuid4()),
            recommendations=[],
            follow_up_questions=[],
            conversation_stage="greeting"
        )

@app.get("/recommendations/{session_id}")
async def get_recommendations(session_id: str):
    """Get recommendations for a session"""
    try:
        session = await db_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        recommendations = await recommendation_engine.get_recommendations(session.preferences)
        return {"recommendations": recommendations}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get session information"""
    try:
        session = await db_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {
            "session_id": session.session_id,
            "preferences": session.preferences.model_dump(),  # Updated from .dict()
            "conversation_history": session.conversation_history,
            "current_stage": session.current_stage
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/universities")
async def get_universities(limit: int = Query(50, ge=1, le=100)):
    """Get list of universities from all provinces"""
    try:
        universities = []
        provinces = [
            'British columbia', 'New Brunswick', 'Newfoundland', 'Nova scotia',
            'Ontario', 'Prince Edward island', 'Quebec', 'Saskatchewan', 
            'alberta', 'manitoba'
        ]
        
        count = 0
        for province in provinces:
            if count >= limit:
                break
            try:
                # Use collection name as-is
                cursor = db_manager.db[province].find({}).limit(limit - count)
                province_universities = await cursor.to_list(length=limit - count)
                
                for uni in province_universities:
                    uni['province'] = province
                    universities.append(uni)
                    count += 1
                    if count >= limit:
                        break
            except Exception as e:
                logger.warning(f"Could not fetch from {province}: {e}")
        
        return {"universities": universities, "total": count}
    except Exception as e:
        logger.error(f"Error getting universities: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        await db_manager.client.admin.command('ping')
        return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e), "timestamp": datetime.utcnow().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)