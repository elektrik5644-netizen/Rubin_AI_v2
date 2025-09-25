"""
Сервис для управления матричной системой
"""

import httpx
import logging
import time
from typing import Dict, List, Optional, Any
from app.core.config import settings

logger = logging.getLogger(__name__)

class MatrixService:
    """Сервис для управления матричной системой"""
    
    def __init__(self):
        self.nodes = {
            "gateway": {"host": "localhost", "port": settings.PORT, "status": "unknown"},
            "compute_core": {"host": settings.COMPUTE_CORE_HOST, "port": settings.COMPUTE_CORE_PORT, "status": "unknown"},
            "postgres_db": {"host": settings.POSTGRES_HOST, "port": settings.POSTGRES_PORT, "status": "unknown"},
            "qdrant_db": {"host": settings.QDRANT_HOST, "port": settings.QDRANT_PORT, "status": "unknown"},
            "ollama_service": {"host": settings.OLLAMA_HOST, "port": settings.OLLAMA_PORT, "status": "unknown"}
        }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Получение статуса всей системы"""
        try:
            status = {
                "system": "Rubin AI Matrix v2.0",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "nodes": {},
                "overall_health": "healthy"
            }
            
            # Проверка статуса каждого узла
            for node_name, node_config in self.nodes.items():
                node_status = await self._check_node_health(node_name, node_config)
                status["nodes"][node_name] = node_status
                
                if node_status["status"] != "healthy":
                    status["overall_health"] = "degraded"
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {
                "system": "Rubin AI Matrix v2.0",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "overall_health": "error",
                "error": str(e)
            }
    
    async def get_nodes_status(self) -> List[Dict[str, Any]]:
        """Получение статуса всех узлов"""
        nodes_status = []
        
        for node_name, node_config in self.nodes.items():
            status = await self._check_node_health(node_name, node_config)
            nodes_status.append({
                "name": node_name,
                **status
            })
        
        return nodes_status
    
    async def get_node_status(self, node_name: str) -> Dict[str, Any]:
        """Получение статуса конкретного узла"""
        if node_name not in self.nodes:
            raise ValueError(f"Unknown node: {node_name}")
        
        node_config = self.nodes[node_name]
        return await self._check_node_health(node_name, node_config)
    
    async def _check_node_health(self, node_name: str, node_config: Dict) -> Dict[str, Any]:
        """Проверка здоровья узла"""
        try:
            if node_name == "gateway":
                return await self._check_gateway_health()
            elif node_name == "compute_core":
                return await self._check_compute_core_health()
            elif node_name == "postgres_db":
                return await self._check_postgres_health()
            elif node_name == "qdrant_db":
                return await self._check_qdrant_health()
            elif node_name == "ollama_service":
                return await self._check_ollama_health()
            else:
                return {
                    "status": "unknown",
                    "health": "unknown",
                    "last_activity": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "error": "Unknown node type"
                }
                
        except Exception as e:
            logger.error(f"Error checking {node_name} health: {e}")
            return {
                "status": "error",
                "health": "unhealthy",
                "last_activity": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "error": str(e)
            }
    
    async def _check_gateway_health(self) -> Dict[str, Any]:
        """Проверка здоровья Gateway"""
        return {
            "status": "healthy",
            "health": "healthy",
            "last_activity": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "metrics": {
                "uptime": "running",
                "version": "2.0.0"
            }
        }
    
    async def _check_compute_core_health(self) -> Dict[str, Any]:
        """Проверка здоровья Compute Core"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"http://{settings.COMPUTE_CORE_HOST}:{settings.COMPUTE_CORE_PORT}/health",
                    timeout=5.0
                )
                
                if response.status_code == 200:
                    return {
                        "status": "healthy",
                        "health": "healthy",
                        "last_activity": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "metrics": response.json()
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "health": "unhealthy",
                        "last_activity": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "error": f"HTTP {response.status_code}"
                    }
                    
        except Exception as e:
            return {
                "status": "unhealthy",
                "health": "unhealthy",
                "last_activity": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "error": str(e)
            }
    
    async def _check_postgres_health(self) -> Dict[str, Any]:
        """Проверка здоровья PostgreSQL"""
        try:
            from app.core.database import check_database_connection
            is_healthy = check_database_connection()
            
            return {
                "status": "healthy" if is_healthy else "unhealthy",
                "health": "healthy" if is_healthy else "unhealthy",
                "last_activity": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "metrics": {
                    "connection": "active" if is_healthy else "failed"
                }
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "health": "unhealthy",
                "last_activity": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "error": str(e)
            }
    
    async def _check_qdrant_health(self) -> Dict[str, Any]:
        """Проверка здоровья Qdrant"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}/health",
                    timeout=5.0
                )
                
                if response.status_code == 200:
                    return {
                        "status": "healthy",
                        "health": "healthy",
                        "last_activity": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "metrics": response.json()
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "health": "unhealthy",
                        "last_activity": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "error": f"HTTP {response.status_code}"
                    }
                    
        except Exception as e:
            return {
                "status": "unhealthy",
                "health": "unhealthy",
                "last_activity": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "error": str(e)
            }
    
    async def _check_ollama_health(self) -> Dict[str, Any]:
        """Проверка здоровья Ollama"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"http://{settings.OLLAMA_HOST}:{settings.OLLAMA_PORT}/api/tags",
                    timeout=5.0
                )
                
                if response.status_code == 200:
                    return {
                        "status": "healthy",
                        "health": "healthy",
                        "last_activity": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "metrics": {
                            "models": len(response.json().get("models", []))
                        }
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "health": "unhealthy",
                        "last_activity": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "error": f"HTTP {response.status_code}"
                    }
                    
        except Exception as e:
            return {
                "status": "unhealthy",
                "health": "unhealthy",
                "last_activity": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "error": str(e)
            }
    
    async def create_task(self, task_type: str, source_node: str, target_node: str, 
                         payload: Dict, priority: int = 1) -> str:
        """Создание новой задачи в матрице"""
        try:
            from app.services.database_service import DatabaseService
            db_service = DatabaseService()
            
            task_id = await db_service.create_matrix_task(
                task_type=task_type,
                source_node=source_node,
                target_node=target_node,
                payload=payload,
                priority=priority
            )
            
            logger.info(f"Created matrix task {task_id} of type {task_type}")
            return task_id
            
        except Exception as e:
            logger.error(f"Error creating matrix task: {e}")
            raise
    
    async def get_tasks(self, status: Optional[str] = None, node_name: Optional[str] = None, 
                       limit: int = 50) -> List[Dict]:
        """Получение списка задач"""
        try:
            from app.services.database_service import DatabaseService
            db_service = DatabaseService()
            
            tasks = await db_service.get_matrix_tasks(
                status=status,
                node_name=node_name,
                limit=limit
            )
            
            return tasks
            
        except Exception as e:
            logger.error(f"Error getting matrix tasks: {e}")
            raise
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Получение статуса задачи"""
        try:
            from app.services.database_service import DatabaseService
            db_service = DatabaseService()
            
            task = await db_service.get_matrix_task(task_id)
            return task
            
        except Exception as e:
            logger.error(f"Error getting task status: {e}")
            raise
    
    async def update_task_status(self, task_id: str, status: str):
        """Обновление статуса задачи"""
        try:
            from app.services.database_service import DatabaseService
            db_service = DatabaseService()
            
            await db_service.update_matrix_task_status(task_id, status)
            
        except Exception as e:
            logger.error(f"Error updating task status: {e}")
            raise
    
    async def update_task_result(self, task_id: str, result: Dict):
        """Обновление результата задачи"""
        try:
            from app.services.database_service import DatabaseService
            db_service = DatabaseService()
            
            await db_service.update_matrix_task_result(task_id, result)
            
        except Exception as e:
            logger.error(f"Error updating task result: {e}")
            raise
    
    async def update_task_error(self, task_id: str, error: str):
        """Обновление ошибки задачи"""
        try:
            from app.services.database_service import DatabaseService
            db_service = DatabaseService()
            
            await db_service.update_matrix_task_error(task_id, error)
            
        except Exception as e:
            logger.error(f"Error updating task error: {e}")
            raise
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Получение метрик системы"""
        try:
            from app.services.database_service import DatabaseService
            db_service = DatabaseService()
            
            metrics = await db_service.get_performance_metrics()
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            raise
    
    async def restart_node(self, node_name: str) -> Dict[str, Any]:
        """Перезапуск узла"""
        # В реальной реализации здесь будет логика перезапуска узла
        # Пока возвращаем заглушку
        return {
            "node": node_name,
            "status": "restart_initiated",
            "message": f"Restart of {node_name} initiated",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
        }
    
    async def process_code_analysis(self, payload: Dict) -> Dict[str, Any]:
        """Обработка анализа кода"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"http://{settings.COMPUTE_CORE_HOST}:{settings.COMPUTE_CORE_PORT}/analyze",
                    json=payload,
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    raise Exception(f"Compute core error: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"Error processing code analysis: {e}")
            raise
    
    async def process_pmac_diagnostics(self, payload: Dict) -> Dict[str, Any]:
        """Обработка диагностики PMAC"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"http://{settings.COMPUTE_CORE_HOST}:{settings.COMPUTE_CORE_PORT}/diagnostics",
                    json=payload,
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    raise Exception(f"Compute core error: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"Error processing PMAC diagnostics: {e}")
            raise
    
    async def process_knowledge_search(self, payload: Dict) -> Dict[str, Any]:
        """Обработка поиска в базе знаний"""
        try:
            from app.services.qdrant_service import QdrantService
            qdrant_service = QdrantService()
            
            results = await qdrant_service.search_similar(
                query=payload.get("query", ""),
                limit=payload.get("limit", 5)
            )
            
            return {"results": results}
            
        except Exception as e:
            logger.error(f"Error processing knowledge search: {e}")
            raise
    
    async def process_generic_task(self, payload: Dict) -> Dict[str, Any]:
        """Обработка общей задачи"""
        return {
            "status": "completed",
            "result": payload,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
        }
