"""
Упрощенный сервер Rubin AI Matrix без Docker
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List, Any
import uvicorn
import logging
import time
import json
import os

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rubin_matrix_simple")

# Создание FastAPI приложения
app = FastAPI(
    title="Rubin AI Matrix Simple",
    description="Упрощенная версия Rubin AI Matrix",
    version="2.0.0"
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключение статических файлов
if os.path.exists("gateway_app/static"):
    app.mount("/static", StaticFiles(directory="gateway_app/static"), name="static")

# Модели данных
class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None
    user_id: Optional[int] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    timestamp: str
    processing_time: float

class CodeAnalysisRequest(BaseModel):
    code: str
    language: str = "python"
    analysis_type: str = "full"

class CodeAnalysisResponse(BaseModel):
    language: str
    analysis_type: str
    results: Dict[str, Any]
    processing_time: float
    timestamp: str

# Глобальные переменные для хранения данных
chat_history = []
code_analyses = []

@app.get("/")
async def root():
    """Главная страница"""
    return {
        "message": "Rubin AI Matrix Simple v2.0",
        "status": "running",
        "version": "2.0.0",
        "endpoints": [
            "/health",
            "/api/chat",
            "/api/code/analyze",
            "/api/matrix/status"
        ]
    }

@app.get("/health")
async def health_check():
    """Проверка здоровья системы"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "services": {
            "chat": "available",
            "code_analysis": "available",
            "matrix": "available"
        }
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_rubin(chat_message: ChatMessage):
    """AI чат с Rubin"""
    start_time = time.time()
    
    try:
        # Простая логика ответов
        user_message = chat_message.message.lower()
        
        if "привет" in user_message or "hello" in user_message:
            response = "Привет! Я Rubin AI. Готов помочь с программированием и промышленной автоматизацией!"
        elif "как дела" in user_message:
            response = "Отлично! Система работает стабильно. Чем могу помочь?"
        elif "python" in user_message:
            response = "Python - отличный язык программирования! Могу помочь с анализом кода, созданием скриптов или решением задач."
        elif "plc" in user_message or "плц" in user_message:
            response = "PLC программирование - моя специализация! Помогу с Ladder Logic, Structured Text, диагностикой PMAC."
        elif "pmac" in user_message:
            response = "PMAC контроллеры - это моя область! Могу помочь с настройкой, программированием и диагностикой."
        elif "анализ" in user_message or "анализ кода" in user_message:
            response = "Анализ кода - одна из моих основных функций! Загрузите код через /api/code/analyze для детального анализа."
        elif "помощь" in user_message or "help" in user_message:
            response = """Доступные функции:
            • Анализ кода (Python, PLC, PMAC)
            • Генерация кода
            • Диагностика промышленного оборудования
            • Программирование PLC
            • Работа с PMAC контроллерами
            • Математические вычисления"""
        else:
            response = f"Понял ваш запрос: '{chat_message.message}'. Я специализируюсь на промышленной автоматизации, программировании PLC, PMAC и анализе кода. Чем конкретно могу помочь?"
        
        # Сохранение в историю
        chat_history.append({
            "user_message": chat_message.message,
            "ai_response": response,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "session_id": chat_message.session_id or "default"
        })
        
        processing_time = time.time() - start_time
        
        return ChatResponse(
            response=response,
            session_id=chat_message.session_id or "default",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {e}")

@app.post("/api/code/analyze", response_model=CodeAnalysisResponse)
async def analyze_code(request: CodeAnalysisRequest):
    """Анализ кода"""
    start_time = time.time()
    
    try:
        logger.info(f"Analyzing {request.language} code...")
        
        # Простой анализ кода
        issues = []
        quality_score = 85.0
        recommendations = []
        security_report = {"level": "low", "issues": []}
        
        code = request.code
        
        # Анализ в зависимости от языка
        if request.language.lower() == "python":
            if "import *" in code:
                issues.append({
                    "type": "warning",
                    "message": "Использование 'import *' не рекомендуется",
                    "severity": "medium"
                })
                recommendations.append("Используйте конкретные импорты")
            
            if "eval(" in code:
                issues.append({
                    "type": "security",
                    "message": "Использование eval() может быть небезопасно",
                    "severity": "high"
                })
                recommendations.append("Избегайте использования eval()")
            
            if len(code.split('\n')) > 50:
                issues.append({
                    "type": "quality",
                    "message": "Код довольно длинный",
                    "severity": "low"
                })
                recommendations.append("Рассмотрите разбиение на функции")
        
        elif request.language.lower() in ["ladder", "st", "fbd"]:
            if "TON" not in code and "TOF" not in code:
                recommendations.append("Рассмотрите использование таймеров")
            
            quality_score = 80.0
        
        # Расчет итогового балла
        quality_score = max(60, quality_score - len(issues) * 5)
        
        results = {
            "issues": issues,
            "quality_score": quality_score,
            "recommendations": recommendations,
            "security_report": security_report,
            "summary": {
                "total_issues": len(issues),
                "security_issues": len([i for i in issues if i.get("type") == "security"]),
                "code_length": len(code.split('\n')),
                "language": request.language
            }
        }
        
        # Сохранение анализа
        code_analyses.append({
            "code": request.code,
            "language": request.language,
            "results": results,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
        })
        
        processing_time = time.time() - start_time
        
        return CodeAnalysisResponse(
            language=request.language,
            analysis_type=request.analysis_type,
            results=results,
            processing_time=processing_time,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        
    except Exception as e:
        logger.error(f"Code analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Code analysis failed: {e}")

@app.get("/api/matrix/status")
async def get_matrix_status():
    """Статус матричной системы"""
    return {
        "system": "Rubin AI Matrix Simple",
        "version": "2.0.0",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "running",
        "nodes": {
            "gateway": {"status": "healthy", "uptime": "running"},
            "chat_service": {"status": "healthy", "messages_processed": len(chat_history)},
            "code_analyzer": {"status": "healthy", "analyses_performed": len(code_analyses)},
            "matrix_core": {"status": "healthy", "tasks_completed": len(chat_history) + len(code_analyses)}
        },
        "statistics": {
            "total_chat_messages": len(chat_history),
            "total_code_analyses": len(code_analyses),
            "uptime": "running"
        }
    }

@app.get("/api/chat/history")
async def get_chat_history(limit: int = 20):
    """Получение истории чата"""
    return {
        "history": chat_history[-limit:] if limit > 0 else chat_history,
        "total": len(chat_history)
    }

@app.get("/api/code/analyses")
async def get_code_analyses(limit: int = 10):
    """Получение истории анализов кода"""
    return {
        "analyses": code_analyses[-limit:] if limit > 0 else code_analyses,
        "total": len(code_analyses)
    }

@app.get("/api/matrix/capabilities")
async def get_capabilities():
    """Получение возможностей системы"""
    return {
        "capabilities": [
            {
                "name": "chat",
                "description": "AI чат для общения с Rubin",
                "endpoint": "/api/chat"
            },
            {
                "name": "code_analysis",
                "description": "Анализ кода на различных языках",
                "endpoint": "/api/code/analyze",
                "supported_languages": ["python", "javascript", "ladder", "st", "fbd", "pmac"]
            },
            {
                "name": "matrix_status",
                "description": "Статус матричной системы",
                "endpoint": "/api/matrix/status"
            },
            {
                "name": "health_check",
                "description": "Проверка здоровья системы",
                "endpoint": "/health"
            }
        ]
    }

if __name__ == "__main__":
    logger.info("🚀 Запуск Rubin AI Matrix Simple...")
    logger.info("🌐 Сервер будет доступен по адресу: http://localhost:8083")
    logger.info("📊 API документация: http://localhost:8083/docs")
    
    uvicorn.run(
        "simple_matrix_server:app",
        host="0.0.0.0",
        port=8083,
        reload=True,
        log_level="info"
    )
