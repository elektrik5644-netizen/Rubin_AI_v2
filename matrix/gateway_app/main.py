"""
Rubin AI Matrix Gateway - Центральный шлюз матричной системы
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from dotenv import load_dotenv
import logging

from app.api import auth, chat, code, diagnostics, matrix
from app.core.config import settings
from app.core.database import engine, Base
from app.core.monitoring import setup_monitoring

# Загрузка переменных окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/gateway.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("rubin_matrix_gateway")

# Создание таблиц базы данных
try:
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")
except Exception as e:
    logger.error(f"Error creating database tables: {e}")

# Создание FastAPI приложения
app = FastAPI(
    title="Rubin AI Matrix Gateway",
    description="Центральный шлюз для матричной системы Rubin AI",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В production указать конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключение статических файлов
app.mount("/static", StaticFiles(directory="static"), name="static")

# Настройка мониторинга
setup_monitoring(app)

# Подключение API роутов
app.include_router(auth.router, prefix="/api/auth", tags=["authentication"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(code.router, prefix="/api/code", tags=["code"])
app.include_router(diagnostics.router, prefix="/api/diagnostics", tags=["diagnostics"])
app.include_router(matrix.router, prefix="/api/matrix", tags=["matrix"])

@app.get("/")
async def root():
    """Главная страница"""
    return {
        "message": "Rubin AI Matrix Gateway v2.0",
        "status": "running",
        "version": "2.0.0",
        "nodes": ["gateway", "compute_core", "postgres_db", "qdrant_db", "ollama_service"]
    }

@app.get("/health")
async def health_check():
    """Проверка здоровья системы"""
    try:
        # Проверка подключения к базам данных
        from app.core.database import get_db
        db = next(get_db())
        
        # Проверка подключения к Qdrant
        from app.services.qdrant_service import QdrantService
        qdrant = QdrantService()
        
        # Проверка подключения к Compute Core
        from app.services.compute_service import ComputeService
        compute = ComputeService()
        
        # Проверка подключения к Ollama
        from app.services.ollama_service import OllamaService
        ollama = OllamaService()
        
        return {
            "status": "healthy",
            "version": "2.0.0",
            "timestamp": "2025-09-13T18:00:00Z",
            "services": {
                "database": "connected",
                "qdrant": "connected",
                "compute_core": "connected",
                "ollama": "connected"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {e}")

@app.get("/api/status")
async def system_status():
    """Статус всех узлов матрицы"""
    try:
        from app.services.matrix_service import MatrixService
        matrix_service = MatrixService()
        status = await matrix_service.get_system_status()
        return status
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting status: {e}")

if __name__ == "__main__":
    logger.info("Starting Rubin AI Matrix Gateway...")
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
