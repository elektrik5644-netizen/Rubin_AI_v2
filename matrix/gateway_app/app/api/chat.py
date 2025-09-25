"""
API для чата с Rubin AI Matrix
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List
import logging
import time

from app.services.matrix_service import MatrixService
from app.services.ollama_service import OllamaService
from app.services.qdrant_service import QdrantService
from app.core.monitoring import record_matrix_task, record_node_response_time

logger = logging.getLogger(__name__)
router = APIRouter()

class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None
    context: Optional[dict] = None
    user_id: Optional[int] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    timestamp: str
    sources: Optional[List[dict]] = None
    processing_time: float

@router.post("/", response_model=ChatResponse)
async def chat_with_matrix(
    chat_message: ChatMessage,
    background_tasks: BackgroundTasks
):
    """
    Основной эндпоинт для общения с Rubin AI Matrix
    """
    start_time = time.time()
    
    try:
        # Инициализация сервисов
        matrix_service = MatrixService()
        ollama_service = OllamaService()
        qdrant_service = QdrantService()
        
        # Поиск релевантного контекста в базе знаний
        logger.info(f"Searching context for message: {chat_message.message[:50]}...")
        context_start = time.time()
        
        relevant_docs = await qdrant_service.search_similar(
            query=chat_message.message,
            limit=5
        )
        
        record_node_response_time(
            "qdrant_db", 
            "search", 
            time.time() - context_start
        )
        
        # Формирование контекста
        context_text = ""
        sources = []
        
        if relevant_docs:
            for doc in relevant_docs:
                context_text += f"\n{doc.get('content', '')}"
                sources.append({
                    "title": doc.get('title', 'Unknown'),
                    "relevance": doc.get('score', 0),
                    "source": doc.get('source', 'Unknown')
                })
        
        # Создание задачи для матрицы
        task_data = {
            "type": "chat_generation",
            "message": chat_message.message,
            "context": context_text,
            "user_id": chat_message.user_id,
            "session_id": chat_message.session_id
        }
        
        # Отправка задачи в Ollama через матрицу
        logger.info("Sending task to Ollama service...")
        ollama_start = time.time()
        
        response = await ollama_service.generate_response(
            prompt=chat_message.message,
            context=context_text,
            system_prompt="Ты - Rubin AI, специализированный помощник по промышленной автоматизации, программированию PLC, PMAC и анализу кода."
        )
        
        record_node_response_time(
            "ollama_service", 
            "generate", 
            time.time() - ollama_start
        )
        
        # Запись метрики задачи
        record_matrix_task("chat_generation", "completed")
        
        # Сохранение в базу данных (в фоне)
        background_tasks.add_task(
            save_chat_interaction,
            chat_message.message,
            response,
            chat_message.session_id,
            chat_message.user_id
        )
        
        processing_time = time.time() - start_time
        
        return ChatResponse(
            response=response,
            session_id=chat_message.session_id or "default",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            sources=sources if sources else None,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        record_matrix_task("chat_generation", "failed")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {e}")

@router.get("/sessions")
async def get_chat_sessions(user_id: int):
    """Получение списка сессий чата"""
    try:
        from app.services.database_service import DatabaseService
        db_service = DatabaseService()
        sessions = await db_service.get_chat_sessions(user_id)
        return {"sessions": sessions}
    except Exception as e:
        logger.error(f"Error getting chat sessions: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting sessions: {e}")

@router.get("/sessions/{session_id}/messages")
async def get_chat_messages(session_id: str, limit: int = 50):
    """Получение сообщений сессии"""
    try:
        from app.services.database_service import DatabaseService
        db_service = DatabaseService()
        messages = await db_service.get_chat_messages(session_id, limit)
        return {"messages": messages}
    except Exception as e:
        logger.error(f"Error getting chat messages: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting messages: {e}")

async def save_chat_interaction(
    user_message: str,
    ai_response: str,
    session_id: str,
    user_id: int
):
    """Сохранение взаимодействия в базу данных"""
    try:
        from app.services.database_service import DatabaseService
        db_service = DatabaseService()
        await db_service.save_chat_interaction(
            user_message, ai_response, session_id, user_id
        )
        logger.info("Chat interaction saved successfully")
    except Exception as e:
        logger.error(f"Error saving chat interaction: {e}")
