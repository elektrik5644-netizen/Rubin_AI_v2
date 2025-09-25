"""
API для управления матричной системой
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, List
import logging
import time

from app.services.matrix_service import MatrixService
from app.core.monitoring import record_matrix_task, record_node_response_time

logger = logging.getLogger(__name__)
router = APIRouter()

class MatrixTask(BaseModel):
    task_type: str
    source_node: str
    target_node: str
    payload: Dict
    priority: Optional[int] = 1

class MatrixStatus(BaseModel):
    node_name: str
    status: str
    health: str
    last_activity: str
    metrics: Optional[Dict] = None

@router.get("/status")
async def get_system_status():
    """Получение статуса всей матричной системы"""
    try:
        matrix_service = MatrixService()
        status = await matrix_service.get_system_status()
        return status
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting status: {e}")

@router.get("/nodes")
async def get_nodes_status():
    """Получение статуса всех узлов"""
    try:
        matrix_service = MatrixService()
        nodes = await matrix_service.get_nodes_status()
        return {"nodes": nodes}
    except Exception as e:
        logger.error(f"Error getting nodes status: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting nodes: {e}")

@router.get("/nodes/{node_name}")
async def get_node_status(node_name: str):
    """Получение статуса конкретного узла"""
    try:
        matrix_service = MatrixService()
        node_status = await matrix_service.get_node_status(node_name)
        return node_status
    except Exception as e:
        logger.error(f"Error getting node status: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting node status: {e}")

@router.post("/tasks")
async def create_matrix_task(
    task: MatrixTask,
    background_tasks: BackgroundTasks
):
    """Создание новой задачи в матрице"""
    try:
        matrix_service = MatrixService()
        
        # Создание задачи
        task_id = await matrix_service.create_task(
            task_type=task.task_type,
            source_node=task.source_node,
            target_node=task.target_node,
            payload=task.payload,
            priority=task.priority
        )
        
        # Запись метрики
        record_matrix_task(task.task_type, "created")
        
        # Обработка задачи в фоне
        background_tasks.add_task(
            process_matrix_task,
            task_id,
            task
        )
        
        return {
            "task_id": task_id,
            "status": "created",
            "message": "Task created successfully"
        }
        
    except Exception as e:
        logger.error(f"Error creating matrix task: {e}")
        record_matrix_task(task.task_type, "failed")
        raise HTTPException(status_code=500, detail=f"Error creating task: {e}")

@router.get("/tasks")
async def get_matrix_tasks(
    status: Optional[str] = None,
    node_name: Optional[str] = None,
    limit: int = 50
):
    """Получение списка задач матрицы"""
    try:
        matrix_service = MatrixService()
        tasks = await matrix_service.get_tasks(
            status=status,
            node_name=node_name,
            limit=limit
        )
        return {"tasks": tasks}
    except Exception as e:
        logger.error(f"Error getting matrix tasks: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting tasks: {e}")

@router.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Получение статуса конкретной задачи"""
    try:
        matrix_service = MatrixService()
        task_status = await matrix_service.get_task_status(task_id)
        return task_status
    except Exception as e:
        logger.error(f"Error getting task status: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting task status: {e}")

@router.get("/metrics")
async def get_system_metrics():
    """Получение метрик системы"""
    try:
        matrix_service = MatrixService()
        metrics = await matrix_service.get_system_metrics()
        return metrics
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting metrics: {e}")

@router.post("/nodes/{node_name}/restart")
async def restart_node(node_name: str):
    """Перезапуск узла матрицы"""
    try:
        matrix_service = MatrixService()
        result = await matrix_service.restart_node(node_name)
        return result
    except Exception as e:
        logger.error(f"Error restarting node: {e}")
        raise HTTPException(status_code=500, detail=f"Error restarting node: {e}")

async def process_matrix_task(task_id: str, task: MatrixTask):
    """Обработка задачи матрицы в фоне"""
    try:
        matrix_service = MatrixService()
        
        # Обновление статуса задачи
        await matrix_service.update_task_status(task_id, "processing")
        
        # Выполнение задачи в зависимости от типа
        if task.task_type == "code_analysis":
            result = await matrix_service.process_code_analysis(task.payload)
        elif task.task_type == "pmac_diagnostics":
            result = await matrix_service.process_pmac_diagnostics(task.payload)
        elif task.task_type == "knowledge_search":
            result = await matrix_service.process_knowledge_search(task.payload)
        else:
            result = await matrix_service.process_generic_task(task.payload)
        
        # Обновление результата
        await matrix_service.update_task_result(task_id, result)
        await matrix_service.update_task_status(task_id, "completed")
        
        # Запись метрики
        record_matrix_task(task.task_type, "completed")
        
        logger.info(f"Task {task_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Error processing task {task_id}: {e}")
        await matrix_service.update_task_status(task_id, "failed")
        await matrix_service.update_task_error(task_id, str(e))
        record_matrix_task(task.task_type, "failed")
