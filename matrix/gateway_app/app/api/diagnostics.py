"""
API для диагностики PMAC и промышленного оборудования
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, List
import logging
import time

from app.services.compute_service import ComputeService
from app.core.monitoring import record_matrix_task, record_node_response_time

logger = logging.getLogger(__name__)
router = APIRouter()

class PMACDiagnosticsRequest(BaseModel):
    command: str
    parameters: Optional[Dict] = None
    user_id: Optional[int] = None

class PMACDiagnosticsResponse(BaseModel):
    command: str
    status: str
    result: Dict
    processing_time: float
    timestamp: str

class EquipmentStatusRequest(BaseModel):
    equipment_type: str
    equipment_id: str
    user_id: Optional[int] = None

class EquipmentStatusResponse(BaseModel):
    equipment_id: str
    equipment_type: str
    status: str
    health: str
    metrics: Dict
    processing_time: float
    timestamp: str

@router.post("/pmac", response_model=PMACDiagnosticsResponse)
async def pmac_diagnostics(
    request: PMACDiagnosticsRequest,
    background_tasks: BackgroundTasks
):
    """Диагностика контроллера PMAC"""
    start_time = time.time()
    
    try:
        # Инициализация сервиса
        compute_service = ComputeService()
        
        # Выполнение диагностики PMAC
        logger.info(f"Running PMAC diagnostics: {request.command}")
        diagnostics_start = time.time()
        
        result = await compute_service.pmac_diagnostics(
            command=request.command,
            parameters=request.parameters
        )
        
        record_node_response_time(
            "compute_core", 
            "pmac_diagnostics", 
            time.time() - diagnostics_start
        )
        
        # Запись метрики
        record_matrix_task("pmac_diagnostics", "completed")
        
        # Сохранение результатов (в фоне)
        background_tasks.add_task(
            save_diagnostics_results,
            request,
            result
        )
        
        processing_time = time.time() - start_time
        
        return PMACDiagnosticsResponse(
            command=request.command,
            status="completed",
            result=result,
            processing_time=processing_time,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        
    except Exception as e:
        logger.error(f"PMAC diagnostics error: {e}")
        record_matrix_task("pmac_diagnostics", "failed")
        raise HTTPException(status_code=500, detail=f"PMAC diagnostics failed: {e}")

@router.post("/equipment/status", response_model=EquipmentStatusResponse)
async def get_equipment_status(
    request: EquipmentStatusRequest,
    background_tasks: BackgroundTasks
):
    """Получение статуса промышленного оборудования"""
    start_time = time.time()
    
    try:
        # Инициализация сервиса
        compute_service = ComputeService()
        
        # Получение статуса оборудования
        logger.info(f"Getting status for {request.equipment_type}: {request.equipment_id}")
        
        # Имитация получения статуса оборудования
        if request.equipment_type.lower() == "pmac":
            result = await compute_service.pmac_diagnostics("status")
        else:
            result = {
                "equipment_id": request.equipment_id,
                "equipment_type": request.equipment_type,
                "status": "unknown",
                "health": "unknown"
            }
        
        # Запись метрики
        record_matrix_task("equipment_status", "completed")
        
        # Сохранение результатов (в фоне)
        background_tasks.add_task(
            save_equipment_status,
            request,
            result
        )
        
        processing_time = time.time() - start_time
        
        return EquipmentStatusResponse(
            equipment_id=request.equipment_id,
            equipment_type=request.equipment_type,
            status=result.get("status", "unknown"),
            health=result.get("health", "unknown"),
            metrics=result,
            processing_time=processing_time,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        
    except Exception as e:
        logger.error(f"Equipment status error: {e}")
        record_matrix_task("equipment_status", "failed")
        raise HTTPException(status_code=500, detail=f"Equipment status failed: {e}")

@router.get("/pmac/commands")
async def get_pmac_commands():
    """Получение списка доступных команд PMAC"""
    return {
        "commands": [
            {
                "name": "status",
                "description": "Получение общего статуса контроллера",
                "parameters": []
            },
            {
                "name": "axis_status",
                "description": "Статус всех осей",
                "parameters": []
            },
            {
                "name": "error_log",
                "description": "Получение лога ошибок",
                "parameters": []
            },
            {
                "name": "move_axis",
                "description": "Перемещение оси",
                "parameters": [
                    {"name": "axis", "type": "int", "description": "Номер оси"},
                    {"name": "position", "type": "float", "description": "Целевая позиция"},
                    {"name": "velocity", "type": "float", "description": "Скорость"}
                ]
            },
            {
                "name": "home_axis",
                "description": "Возврат оси в исходное положение",
                "parameters": [
                    {"name": "axis", "type": "int", "description": "Номер оси"}
                ]
            },
            {
                "name": "stop_axis",
                "description": "Остановка оси",
                "parameters": [
                    {"name": "axis", "type": "int", "description": "Номер оси"}
                ]
            }
        ]
    }

@router.get("/equipment/types")
async def get_equipment_types():
    """Получение типов поддерживаемого оборудования"""
    return {
        "equipment_types": [
            {
                "type": "pmac",
                "name": "PMAC Controller",
                "description": "Контроллер PMAC для ЧПУ систем",
                "capabilities": ["motion_control", "io_control", "programming"]
            },
            {
                "type": "plc",
                "name": "PLC Controller",
                "description": "Программируемый логический контроллер",
                "capabilities": ["logic_control", "io_control", "communication"]
            },
            {
                "type": "hmi",
                "name": "Human Machine Interface",
                "description": "Человеко-машинный интерфейс",
                "capabilities": ["visualization", "control", "monitoring"]
            },
            {
                "type": "sensor",
                "name": "Industrial Sensor",
                "description": "Промышленный датчик",
                "capabilities": ["measurement", "monitoring", "alarm"]
            }
        ]
    }

@router.get("/history")
async def get_diagnostics_history(user_id: int, limit: int = 20):
    """Получение истории диагностики"""
    try:
        from app.services.database_service import DatabaseService
        db_service = DatabaseService()
        
        # В реальной реализации здесь будет запрос к базе данных
        history = [
            {
                "id": "diag_1",
                "command": "status",
                "timestamp": "2025-09-13T18:00:00Z",
                "status": "completed"
            }
        ]
        
        return {"history": history}
        
    except Exception as e:
        logger.error(f"Error getting diagnostics history: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting history: {e}")

async def save_diagnostics_results(
    request: PMACDiagnosticsRequest,
    result: Dict
):
    """Сохранение результатов диагностики"""
    try:
        from app.services.database_service import DatabaseService
        db_service = DatabaseService()
        
        # В реальной реализации здесь будет сохранение в базу данных
        logger.info(f"Saved PMAC diagnostics results for command: {request.command}")
        
    except Exception as e:
        logger.error(f"Error saving diagnostics results: {e}")

async def save_equipment_status(
    request: EquipmentStatusRequest,
    result: Dict
):
    """Сохранение статуса оборудования"""
    try:
        from app.services.database_service import DatabaseService
        db_service = DatabaseService()
        
        # В реальной реализации здесь будет сохранение в базу данных
        logger.info(f"Saved equipment status: {request.equipment_id}")
        
    except Exception as e:
        logger.error(f"Error saving equipment status: {e}")
