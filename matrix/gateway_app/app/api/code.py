"""
API для анализа кода
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, List
import logging
import time

from app.services.compute_service import ComputeService
from app.services.ollama_service import OllamaService
from app.core.monitoring import record_matrix_task, record_node_response_time

logger = logging.getLogger(__name__)
router = APIRouter()

class CodeAnalysisRequest(BaseModel):
    code: str
    language: str = "python"
    analysis_type: str = "full"  # full, security, performance, quality
    user_id: Optional[int] = None

class CodeAnalysisResponse(BaseModel):
    analysis_id: str
    language: str
    analysis_type: str
    results: Dict
    processing_time: float
    timestamp: str

class CodeGenerationRequest(BaseModel):
    description: str
    language: str = "python"
    framework: Optional[str] = None
    user_id: Optional[int] = None

class CodeGenerationResponse(BaseModel):
    generated_code: str
    language: str
    explanation: str
    processing_time: float
    timestamp: str

@router.post("/analyze", response_model=CodeAnalysisResponse)
async def analyze_code(
    request: CodeAnalysisRequest,
    background_tasks: BackgroundTasks
):
    """Анализ кода с помощью Compute Core и AI"""
    start_time = time.time()
    
    try:
        # Инициализация сервисов
        compute_service = ComputeService()
        ollama_service = OllamaService()
        
        # Создание задачи для матрицы
        task_data = {
            "type": "code_analysis",
            "code": request.code,
            "language": request.language,
            "analysis_type": request.analysis_type,
            "user_id": request.user_id
        }
        
        # Анализ кода в Compute Core
        logger.info(f"Analyzing {request.language} code...")
        compute_start = time.time()
        
        analysis_result = await compute_service.analyze_code(
            code=request.code,
            language=request.language,
            analysis_type=request.analysis_type
        )
        
        record_node_response_time(
            "compute_core", 
            "code_analysis", 
            time.time() - compute_start
        )
        
        # Улучшение анализа с помощью AI
        logger.info("Enhancing analysis with AI...")
        ai_start = time.time()
        
        ai_enhancement = await ollama_service.analyze_code(
            code=request.code,
            language=request.language
        )
        
        record_node_response_time(
            "ollama_service", 
            "code_analysis", 
            time.time() - ai_start
        )
        
        # Объединение результатов
        combined_results = {
            "static_analysis": analysis_result,
            "ai_analysis": ai_enhancement,
            "summary": {
                "total_issues": len(analysis_result.get("issues", [])),
                "security_issues": len([i for i in analysis_result.get("issues", []) if i.get("type") == "security"]),
                "quality_score": analysis_result.get("quality_score", 0),
                "recommendations": analysis_result.get("recommendations", [])
            }
        }
        
        # Запись метрики
        record_matrix_task("code_analysis", "completed")
        
        # Сохранение результатов (в фоне)
        analysis_id = f"analysis_{int(time.time())}"
        background_tasks.add_task(
            save_analysis_results,
            analysis_id,
            request,
            combined_results
        )
        
        processing_time = time.time() - start_time
        
        return CodeAnalysisResponse(
            analysis_id=analysis_id,
            language=request.language,
            analysis_type=request.analysis_type,
            results=combined_results,
            processing_time=processing_time,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        
    except Exception as e:
        logger.error(f"Code analysis error: {e}")
        record_matrix_task("code_analysis", "failed")
        raise HTTPException(status_code=500, detail=f"Code analysis failed: {e}")

@router.post("/generate", response_model=CodeGenerationResponse)
async def generate_code(
    request: CodeGenerationRequest,
    background_tasks: BackgroundTasks
):
    """Генерация кода с помощью AI"""
    start_time = time.time()
    
    try:
        # Инициализация сервисов
        ollama_service = OllamaService()
        
        # Генерация кода в зависимости от языка
        if request.language.lower() in ["ladder", "st", "fbd", "sfc"]:
            # PLC код
            generated_code = await ollama_service.generate_plc_code(
                description=request.description,
                language=request.language
            )
        else:
            # Обычный код
            system_prompt = f"""Ты - эксперт по программированию на языке {request.language}.
            Сгенерируй качественный код на основе описания.
            Используй лучшие практики и следуй стандартам языка.
            Отвечай на русском языке."""
            
            generated_code = await ollama_service.generate_response(
                prompt=f"Сгенерируй код на {request.language} для: {request.description}",
                system_prompt=system_prompt
            )
        
        # Генерация объяснения
        explanation = await ollama_service.generate_response(
            prompt=f"Объясни этот код на {request.language}:\n\n{generated_code}",
            system_prompt="Объясни код подробно, включая основные функции и логику."
        )
        
        # Запись метрики
        record_matrix_task("code_generation", "completed")
        
        # Сохранение результатов (в фоне)
        background_tasks.add_task(
            save_generation_results,
            request,
            generated_code,
            explanation
        )
        
        processing_time = time.time() - start_time
        
        return CodeGenerationResponse(
            generated_code=generated_code,
            language=request.language,
            explanation=explanation,
            processing_time=processing_time,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        
    except Exception as e:
        logger.error(f"Code generation error: {e}")
        record_matrix_task("code_generation", "failed")
        raise HTTPException(status_code=500, detail=f"Code generation failed: {e}")

@router.get("/languages")
async def get_supported_languages():
    """Получение списка поддерживаемых языков программирования"""
    return {
        "languages": [
            {"name": "Python", "value": "python", "type": "general"},
            {"name": "JavaScript", "value": "javascript", "type": "general"},
            {"name": "TypeScript", "value": "typescript", "type": "general"},
            {"name": "Java", "value": "java", "type": "general"},
            {"name": "C++", "value": "cpp", "type": "general"},
            {"name": "C#", "value": "csharp", "type": "general"},
            {"name": "Go", "value": "go", "type": "general"},
            {"name": "Rust", "value": "rust", "type": "general"},
            {"name": "Ladder Logic", "value": "ladder", "type": "plc"},
            {"name": "Structured Text", "value": "st", "type": "plc"},
            {"name": "Function Block Diagram", "value": "fbd", "type": "plc"},
            {"name": "Sequential Function Chart", "value": "sfc", "type": "plc"},
            {"name": "PMAC Script", "value": "pmac", "type": "industrial"},
            {"name": "G-Code", "value": "gcode", "type": "industrial"}
        ]
    }

@router.get("/analysis/{analysis_id}")
async def get_analysis_results(analysis_id: str):
    """Получение результатов анализа кода"""
    try:
        from app.services.database_service import DatabaseService
        db_service = DatabaseService()
        
        results = await db_service.get_analysis_results(analysis_id)
        if not results:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting analysis results: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting results: {e}")

@router.get("/history")
async def get_analysis_history(user_id: int, limit: int = 20):
    """Получение истории анализов пользователя"""
    try:
        from app.services.database_service import DatabaseService
        db_service = DatabaseService()
        
        history = await db_service.get_analysis_history(user_id, limit)
        return {"history": history}
        
    except Exception as e:
        logger.error(f"Error getting analysis history: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting history: {e}")

async def save_analysis_results(
    analysis_id: str,
    request: CodeAnalysisRequest,
    results: Dict
):
    """Сохранение результатов анализа в базу данных"""
    try:
        from app.services.database_service import DatabaseService
        db_service = DatabaseService()
        
        await db_service.save_analysis_results(
            analysis_id=analysis_id,
            code=request.code,
            language=request.language,
            analysis_type=request.analysis_type,
            results=results,
            user_id=request.user_id
        )
        
        logger.info(f"Analysis results saved: {analysis_id}")
        
    except Exception as e:
        logger.error(f"Error saving analysis results: {e}")

async def save_generation_results(
    request: CodeGenerationRequest,
    generated_code: str,
    explanation: str
):
    """Сохранение результатов генерации в базу данных"""
    try:
        from app.services.database_service import DatabaseService
        db_service = DatabaseService()
        
        await db_service.save_generation_results(
            description=request.description,
            language=request.language,
            generated_code=generated_code,
            explanation=explanation,
            user_id=request.user_id
        )
        
        logger.info("Code generation results saved")
        
    except Exception as e:
        logger.error(f"Error saving generation results: {e}")
