from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
import uuid
from datetime import datetime, timezone
from emergentintegrations.llm.chat import LlmChat, UserMessage
import json

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Define Models
class Flashcard(BaseModel):
    model_config = ConfigDict(extra="ignore")
    question: str
    answer: str

class FlashcardSet(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    summary: str
    flashcards: List[Flashcard]
    category: str
    count: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class GenerateRequest(BaseModel):
    summary: str
    count: int = Field(default=12, ge=1, le=50)
    category: str = Field(default="General")

class SaveFlashcardsRequest(BaseModel):
    title: str
    summary: str
    flashcards: List[Flashcard]
    category: str
    count: int

class EvaluateAnswerRequest(BaseModel):
    question: str
    correct_answer: str
    user_answer: str

# Routes
@api_router.get("/")
async def root():
    return {"message": "Flashcards API"}

@api_router.post("/generate-flashcards")
async def generate_flashcards(request: GenerateRequest):
    try:
        # Get API key from environment
        api_key = os.environ.get('EMERGENT_LLM_KEY')
        if not api_key:
            raise HTTPException(status_code=500, detail="API key no configurada")
        
        # Create chat instance
        chat = LlmChat(
            api_key=api_key,
            session_id=str(uuid.uuid4()),
            system_message=f"Eres un asistente educativo. Genera exactamente {request.count} flashcards (preguntas y respuestas) basadas en el siguiente resumen. Devuelve SOLO un JSON válido con el formato: {{\"flashcards\": [{{\"question\": \"pregunta\", \"answer\": \"respuesta\"}}]}}. No incluyas explicaciones adicionales, solo el JSON."
        ).with_model("openai", "gpt-5.1")
        
        # Create user message
        user_message = UserMessage(
            text=f"Genera {request.count} flashcards basadas en este resumen:\n\n{request.summary}\n\nRecuerda: devuelve SOLO el JSON sin texto adicional."
        )
        
        # Send message and get response
        response = await chat.send_message(user_message)
        
        # Parse the response
        try:
            # Clean the response text
            response_text = response.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith('```'):
                lines = response_text.split('\n')
                response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            
            flashcards_data = json.loads(response_text)
            
            if "flashcards" not in flashcards_data:
                raise ValueError("Formato de respuesta inválido")
            
            flashcards = [Flashcard(**fc) for fc in flashcards_data["flashcards"]]
            
            return {
                "flashcards": flashcards,
                "count": len(flashcards)
            }
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON: {e}. Response: {response}")
            raise HTTPException(status_code=500, detail=f"Error al parsear la respuesta de IA: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing flashcards: {e}")
            raise HTTPException(status_code=500, detail=f"Error al procesar flashcards: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error generating flashcards: {e}")
        raise HTTPException(status_code=500, detail=f"Error generando flashcards: {str(e)}")


@api_router.post("/evaluate-answer")
async def evaluate_answer(request: EvaluateAnswerRequest):
    try:
        # Get API key from environment
        api_key = os.environ.get('EMERGENT_LLM_KEY')
        if not api_key:
            raise HTTPException(status_code=500, detail="API key no configurada")
        
        # Create chat instance for evaluation
        chat = LlmChat(
            api_key=api_key,
            session_id=str(uuid.uuid4()),
            system_message="Eres un evaluador educativo. Tu trabajo es determinar si la respuesta del estudiante es correcta o incorrecta comparándola con la respuesta correcta. Debes ser flexible y aceptar respuestas que sean semánticamente equivalentes aunque no sean idénticas. Responde SOLO con un JSON en este formato: {\"is_correct\": true/false, \"feedback\": \"breve explicación\"}. No incluyas texto adicional."
        ).with_model("openai", "gpt-5.1")
        
        # Create evaluation message
        user_message = UserMessage(
            text=f"Pregunta: {request.question}\n\nRespuesta correcta: {request.correct_answer}\n\nRespuesta del estudiante: {request.user_answer}\n\n¿La respuesta del estudiante es correcta? Recuerda: acepta respuestas semánticamente equivalentes."
        )
        
        # Send message and get response
        response = await chat.send_message(user_message)
        
        # Parse the response
        try:
            # Clean the response text
            response_text = response.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith('```'):
                lines = response_text.split('\n')
                response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            
            evaluation_data = json.loads(response_text)
            
            return {
                "is_correct": evaluation_data.get("is_correct", False),
                "feedback": evaluation_data.get("feedback", "")
            }
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing evaluation JSON: {e}. Response: {response}")
            raise HTTPException(status_code=500, detail=f"Error al parsear respuesta de evaluación: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error evaluating answer: {e}")
        raise HTTPException(status_code=500, detail=f"Error evaluando respuesta: {str(e)}")

@api_router.post("/save-flashcards", response_model=FlashcardSet)
async def save_flashcards(request: SaveFlashcardsRequest):
    try:
        flashcard_set = FlashcardSet(
            title=request.title,
            summary=request.summary,
            flashcards=request.flashcards,
            category=request.category,
            count=request.count
        )
        
        # Convert to dict and serialize datetime to ISO string for MongoDB
        doc = flashcard_set.model_dump()
        doc['timestamp'] = doc['timestamp'].isoformat()
        
        await db.flashcard_sets.insert_one(doc)
        return flashcard_set
    except Exception as e:
        logger.error(f"Error saving flashcards: {e}")
        raise HTTPException(status_code=500, detail=f"Error guardando flashcards: {str(e)}")

@api_router.get("/flashcards", response_model=List[FlashcardSet])
async def get_flashcards():
    try:
        flashcard_sets = await db.flashcard_sets.find({}, {"_id": 0}).to_list(1000)
        
        # Convert ISO string timestamps back to datetime objects
        for fs in flashcard_sets:
            if isinstance(fs['timestamp'], str):
                fs['timestamp'] = datetime.fromisoformat(fs['timestamp'])
        
        # Sort by timestamp descending (most recent first)
        flashcard_sets.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return flashcard_sets
    except Exception as e:
        logger.error(f"Error fetching flashcards: {e}")
        raise HTTPException(status_code=500, detail=f"Error obteniendo flashcards: {str(e)}")

@api_router.get("/flashcards/category/{category}", response_model=List[FlashcardSet])
async def get_flashcards_by_category(category: str):
    try:
        flashcard_sets = await db.flashcard_sets.find(
            {"category": category}, 
            {"_id": 0}
        ).to_list(1000)
        
        # Convert ISO string timestamps back to datetime objects
        for fs in flashcard_sets:
            if isinstance(fs['timestamp'], str):
                fs['timestamp'] = datetime.fromisoformat(fs['timestamp'])
        
        # Sort by timestamp descending
        flashcard_sets.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return flashcard_sets
    except Exception as e:
        logger.error(f"Error fetching flashcards by category: {e}")
        raise HTTPException(status_code=500, detail=f"Error obteniendo flashcards por categoría: {str(e)}")

@api_router.get("/categories")
async def get_categories():
    try:
        # Get unique categories
        categories = await db.flashcard_sets.distinct("category")
        return {"categories": categories}
    except Exception as e:
        logger.error(f"Error fetching categories: {e}")
        raise HTTPException(status_code=500, detail=f"Error obteniendo categorías: {str(e)}")

@api_router.delete("/flashcards/{flashcard_id}")
async def delete_flashcard_set(flashcard_id: str):
    try:
        result = await db.flashcard_sets.delete_one({"id": flashcard_id})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Flashcard set no encontrado")
        return {"message": "Flashcard set eliminado exitosamente"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting flashcard set: {e}")
        raise HTTPException(status_code=500, detail=f"Error eliminando flashcard set: {str(e)}")

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()