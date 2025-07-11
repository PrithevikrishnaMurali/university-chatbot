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
    DATABASE_NAME = os.getenv("DATABASE_NAME", "university_chatbot")
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

# NLP Engine
class NLPEngine:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Intent patterns
        self.intent_patterns = {
            'greeting': [
                r'\b(hi|hello|hey|good morning|good afternoon|good evening)\b',
                r'\b(start|begin|help)\b'
            ],
            'field_of_study': [
                r'\b(study|major|field|course|program|subject|engineering|computer science|business|medicine|law)\b',
                r'\b(want to study|interested in|planning to study)\b'
            ],
            'location': [
                r'\b(location|place|city|state|country|where|geographical)\b',
                r'\b(prefer|want|like)\s+.*\b(location|place|city|state)\b'
            ],
            'budget': [
                r'\b(budget|cost|money|expensive|cheap|afford|financial|tuition|fees)\b',
                r'\b(how much|price|scholarship|funding)\b'
            ],
            'degree_level': [
                r'\b(bachelor|master|phd|doctorate|undergraduate|graduate|degree)\b',
                r'\b(bachelors|masters|doctoral)\b'
            ],
            'academic_profile': [
                r'\b(gpa|grade|score|sat|act|gre|gmat|toefl|ielts)\b',
                r'\b(academic|performance|achievement)\b'
            ],
            'international': [
                r'\b(international|foreign|visa|overseas|abroad)\b',
                r'\b(not from|outside|different country)\b'
            ],
            'recommendation_request': [
                r'\b(recommend|suggest|show|find|best|top|good)\b.*\b(university|college|school)\b',
                r'\b(what.*university|which.*college|where.*study)\b'
            ],
            'yes': [r'\b(yes|yeah|yep|sure|ok|okay|definitely|absolutely)\b'],
            'no': [r'\b(no|nope|not|never|none)\b'],
            'more_info': [r'\b(more|tell me|information|details|about)\b']
        }
        
        # Entity extraction patterns
        self.entity_patterns = {
            'field_of_study': [
                'computer science', 'engineering', 'business', 'medicine', 'law',
                'psychology', 'biology', 'chemistry', 'physics', 'mathematics',
                'economics', 'political science', 'history', 'literature', 'art'
            ],
            'location': [
                'usa', 'united states', 'uk', 'united kingdom', 'canada', 'australia',
                'germany', 'france', 'california', 'new york', 'texas', 'florida'
            ],
            'degree_level': [
                'bachelor', 'master', 'phd', 'doctorate', 'undergraduate', 'graduate'
            ],
            'budget_range': [
                'under 10000', 'under 20000', 'under 30000', 'under 50000',
                'no limit', 'high budget', 'low budget', 'medium budget'
            ]
        }
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for NLP analysis"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        return ' '.join(tokens)
    
    def extract_intent(self, text: str) -> str:
        """Extract user intent from text"""
        text = text.lower()
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return intent
        
        return 'general'
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from text"""
        entities = {}
        text = text.lower()
        
        for entity_type, values in self.entity_patterns.items():
            found_entities = []
            for value in values:
                if value in text:
                    found_entities.append(value)
            if found_entities:
                entities[entity_type] = found_entities
        
        return entities
    
    def generate_response(self, intent: str, entities: Dict, stage: str, preferences: UserPreferences) -> Dict:
        """Generate appropriate response based on intent and context"""
        
        if intent == 'greeting' or stage == ConversationStage.GREETING.value:
            return {
                'response': "Hello! I'm here to help you find the perfect university for your studies. Let's start by learning more about what you're looking for. What field would you like to study?",
                'stage': ConversationStage.FIELD_OF_STUDY.value,
                'questions': ['What field of study are you interested in?']
            }
        
        elif stage == ConversationStage.FIELD_OF_STUDY.value:
            if 'field_of_study' in entities:
                field = entities['field_of_study'][0]
                return {
                    'response': f"Great! You're interested in {field}. Where would you prefer to study? Any specific country or region?",
                    'stage': ConversationStage.LOCATION.value,
                    'questions': ['Which country or region would you prefer?']
                }
            else:
                return {
                    'response': "I'd love to help you find the right field! Could you tell me more about your interests? For example: engineering, business, medicine, computer science, etc.",
                    'stage': ConversationStage.FIELD_OF_STUDY.value,
                    'questions': ['What subjects or careers interest you most?']
                }
        
        elif stage == ConversationStage.LOCATION.value:
            if 'location' in entities:
                location = entities['location'][0]
                return {
                    'response': f"Excellent! You're interested in studying in {location}. What's your budget range for tuition and living expenses?",
                    'stage': ConversationStage.BUDGET.value,
                    'questions': ['What is your budget range?', 'Do you need financial aid or scholarships?']
                }
            else:
                return {
                    'response': "No worries! Location can be flexible. What about your budget? Are you looking for affordable options or do you have a specific budget range in mind?",
                    'stage': ConversationStage.BUDGET.value,
                    'questions': ['What is your budget range?']
                }
        
        elif stage == ConversationStage.BUDGET.value:
            return {
                'response': "Thanks for that information! What level of degree are you looking for? Bachelor's, Master's, or PhD?",
                'stage': ConversationStage.DEGREE_LEVEL.value,
                'questions': ['What degree level are you seeking?']
            }
        
        elif stage == ConversationStage.DEGREE_LEVEL.value:
            return {
                'response': "Perfect! Could you share your academic profile? What's your GPA and any test scores (SAT, ACT, GRE, etc.)?",
                'stage': ConversationStage.ACADEMIC_PROFILE.value,
                'questions': ['What is your GPA?', 'Do you have any standardized test scores?']
            }
        
        elif stage == ConversationStage.ACADEMIC_PROFILE.value:
            return {
                'response': "Thanks! One more question: Are you an international student? This helps me find universities with good international support.",
                'stage': ConversationStage.INTERNATIONAL_STATUS.value,
                'questions': ['Are you an international student?']
            }
        
        else:
            return {
                'response': "Based on your preferences, let me find some great university recommendations for you!",
                'stage': ConversationStage.PROVIDING_RECOMMENDATIONS.value,
                'questions': []
            }

# Recommendation Engine
class RecommendationEngine:
    def __init__(self):
        self.universities = []
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
    async def load_universities(self, db):
        """Load universities from database"""
        cursor = db.universities.find({})
        self.universities = await cursor.to_list(length=None)
        
    def calculate_match_score(self, university: Dict, preferences: UserPreferences) -> float:
        """Calculate match score between university and user preferences"""
        score = 0.0
        max_score = 0.0
        
        # Field of study matching
        if preferences.field_of_study:
            max_score += 30
            if any(preferences.field_of_study.lower() in program.lower() 
                  for program in university.get('programs', [])):
                score += 30
        
        # Location matching
        if preferences.location_preference:
            max_score += 20
            if preferences.location_preference.lower() in university.get('location', '').lower():
                score += 20
        
        # Budget matching
        if preferences.budget_range and university.get('tuition_fees'):
            max_score += 25
            tuition = university['tuition_fees'].get('international', 
                     university['tuition_fees'].get('domestic', 0))
            if self._is_within_budget(tuition, preferences.budget_range):
                score += 25
        
        # Degree level matching
        if preferences.degree_level:
            max_score += 15
            if preferences.degree_level.lower() in str(university.get('programs', [])).lower():
                score += 15
        
        # International student support
        if preferences.international_student:
            max_score += 10
            if university.get('international_support', False):
                score += 10
        
        return (score / max_score * 100) if max_score > 0 else 0
    
    def _is_within_budget(self, tuition: float, budget_range: str) -> bool:
        """Check if tuition is within budget range"""
        budget_map = {
            'under 10000': 10000,
            'under 20000': 20000,
            'under 30000': 30000,
            'under 50000': 50000,
            'no limit': float('inf')
        }
        return tuition <= budget_map.get(budget_range, float('inf'))
    
    async def get_recommendations(self, preferences: UserPreferences, limit: int = 5) -> List[Dict]:
        """Get university recommendations based on preferences"""
        if not self.universities:
            return []
        
        # Calculate match scores
        scored_universities = []
        for university in self.universities:
            score = self.calculate_match_score(university, preferences)
            if score > 0:
                university_with_score = university.copy()
                university_with_score['match_score'] = score
                scored_universities.append(university_with_score)
        
        # Sort by score and return top recommendations
        scored_universities.sort(key=lambda x: x['match_score'], reverse=True)
        return scored_universities[:limit]

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
            logger.info("Connected to MongoDB successfully")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    async def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
    
    async def create_indexes(self):
        """Create database indexes"""
        await self.db.universities.create_index([("name", 1)])
        await self.db.universities.create_index([("location", 1)])
        await self.db.universities.create_index([("programs", 1)])
        await self.db.sessions.create_index([("session_id", 1)])
        await self.db.sessions.create_index([("created_at", 1)], expireAfterSeconds=3600*24)  # 24 hours
    
    async def save_session(self, session: UserSession):
        """Save user session to database"""
        session_dict = {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "preferences": session.preferences.dict(),
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
    
    async def load_universities_from_json(self, json_file_path: str):
        """Load universities from JSON file to database"""
        try:
            with open(json_file_path, 'r', encoding='utf-8') as file:
                universities_data = json.load(file)
            
            if isinstance(universities_data, list):
                await self.db.universities.delete_many({})  # Clear existing data
                await self.db.universities.insert_many(universities_data)
                logger.info(f"Loaded {len(universities_data)} universities from JSON")
            else:
                logger.error("JSON file should contain a list of universities")
        except Exception as e:
            logger.error(f"Error loading universities from JSON: {e}")

# Chatbot Controller
class ChatbotController:
    def __init__(self, db_manager: DatabaseManager, recommendation_engine: RecommendationEngine):
        self.db_manager = db_manager
        self.recommendation_engine = recommendation_engine
        self.nlp_engine = NLPEngine()
    
    async def process_message(self, message: str, session_id: str = None) -> ChatResponse:
        """Process user message and generate response"""
        
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
        
        # Extract intent and entities
        intent = self.nlp_engine.extract_intent(message)
        entities = self.nlp_engine.extract_entities(message)
        
        # Update preferences based on entities
        await self._update_preferences(session.preferences, entities, message)
        
        # Generate response
        response_data = self.nlp_engine.generate_response(
            intent, entities, session.current_stage, session.preferences
        )
        
        # Update session
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
            recommendations = await self.recommendation_engine.get_recommendations(session.preferences)
        
        # Save session
        await self.db_manager.save_session(session)
        
        return ChatResponse(
            response=response_data['response'],
            session_id=session.session_id,
            recommendations=recommendations,
            follow_up_questions=response_data.get('questions', []),
            conversation_stage=session.current_stage
        )
    
    async def _update_preferences(self, preferences: UserPreferences, entities: Dict, message: str):
        """Update user preferences based on extracted entities and message"""
        
        # Update field of study
        if 'field_of_study' in entities:
            preferences.field_of_study = entities['field_of_study'][0]
        
        # Update location
        if 'location' in entities:
            preferences.location_preference = entities['location'][0]
        
        # Update budget (extract from message)
        budget_patterns = {
            r'under.*?(\d+)': lambda x: f"under {x}",
            r'less.*?(\d+)': lambda x: f"under {x}",
            r'no.*?limit': lambda x: "no limit",
            r'unlimited': lambda x: "no limit"
        }
        
        for pattern, formatter in budget_patterns.items():
            match = re.search(pattern, message.lower())
            if match:
                if 'under' in pattern or 'less' in pattern:
                    preferences.budget_range = formatter(match.group(1))
                else:
                    preferences.budget_range = formatter(None)
                break
        
        # Update degree level
        if 'degree_level' in entities:
            preferences.degree_level = entities['degree_level'][0]
        
        # Update international status
        if any(word in message.lower() for word in ['international', 'foreign', 'abroad']):
            preferences.international_student = True
        elif any(word in message.lower() for word in ['domestic', 'local', 'same country']):
            preferences.international_student = False
        
        # Extract GPA
        gpa_match = re.search(r'gpa.*?(\d+\.?\d*)', message.lower())
        if gpa_match:
            preferences.gpa = float(gpa_match.group(1))
        
        # Extract test scores
        test_patterns = {
            'sat': r'sat.*?(\d+)',
            'act': r'act.*?(\d+)',
            'gre': r'gre.*?(\d+)',
            'gmat': r'gmat.*?(\d+)',
            'toefl': r'toefl.*?(\d+)',
            'ielts': r'ielts.*?(\d+\.?\d*)'
        }
        
        test_scores = {}
        for test, pattern in test_patterns.items():
            match = re.search(pattern, message.lower())
            if match:
                test_scores[test] = int(float(match.group(1)))
        
        if test_scores:
            preferences.test_scores = test_scores

# Initialize components
db_manager = DatabaseManager()
recommendation_engine = RecommendationEngine()
chatbot_controller = ChatbotController(db_manager, recommendation_engine)

# Lifespan event handler (replaces deprecated on_event)
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
        # You can choose to raise the exception to prevent startup
        # raise e
    
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
        return response
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

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
            "preferences": session.preferences.dict(),
            "conversation_history": session.conversation_history,
            "current_stage": session.current_stage
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/admin/load-universities")
async def load_universities(file_path: str):
    """Admin endpoint to load universities from JSON file"""
    try:
        await db_manager.load_universities_from_json(file_path)
        await recommendation_engine.load_universities(db_manager.db)
        return {"message": "Universities loaded successfully"}
    except Exception as e:
        logger.error(f"Error loading universities: {e}")
        raise HTTPException(status_code=500, detail="Failed to load universities")

@app.get("/universities")
async def get_universities(limit: int = Query(50, ge=1, le=100)):
    """Get list of universities"""
    try:
        cursor = db_manager.db.universities.find({}).limit(limit)
        universities = await cursor.to_list(length=limit)
        return {"universities": universities}
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