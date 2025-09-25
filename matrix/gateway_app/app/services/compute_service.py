"""
Сервис для работы с Compute Core
"""

import httpx
import logging
import time
from typing import Dict, List, Optional, Any
from app.core.config import settings

logger = logging.getLogger(__name__)

class ComputeService:
    """Сервис для работы с Compute Core"""
    
    def __init__(self):
        self.base_url = f"http://{settings.COMPUTE_CORE_HOST}:{settings.COMPUTE_CORE_PORT}"
        self.timeout = 30.0
    
    async def health_check(self) -> Dict[str, Any]:
        """Проверка здоровья Compute Core"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/health",
                    timeout=5.0
                )
                
                if response.status_code == 200:
                    return {
                        "status": "healthy",
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "details": response.json()
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "error": f"HTTP {response.status_code}",
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
                    }
                    
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
            }
    
    async def analyze_code(
        self, 
        code: str, 
        language: str = "python", 
        analysis_type: str = "full"
    ) -> Dict[str, Any]:
        """Анализ кода"""
        try:
            request_data = {
                "code": code,
                "language": language,
                "analysis_type": analysis_type
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/analyze",
                    json=request_data,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Compute Core analysis error: {response.status_code}")
                    raise Exception(f"Compute Core error: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"Error analyzing code: {e}")
            raise
    
    async def pmac_diagnostics(
        self, 
        command: str, 
        parameters: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Диагностика PMAC"""
        try:
            request_data = {
                "command": command,
                "parameters": parameters or {}
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/diagnostics",
                    json=request_data,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Compute Core diagnostics error: {response.status_code}")
                    raise Exception(f"Compute Core error: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"Error running PMAC diagnostics: {e}")
            raise
    
    async def math_calculation(
        self, 
        formula: str, 
        parameters: Dict[str, float], 
        calculation_type: str = "general"
    ) -> Dict[str, Any]:
        """Математические вычисления"""
        try:
            request_data = {
                "formula": formula,
                "parameters": parameters,
                "calculation_type": calculation_type
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/calculate",
                    json=request_data,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Compute Core calculation error: {response.status_code}")
                    raise Exception(f"Compute Core error: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"Error performing math calculation: {e}")
            raise
    
    async def get_capabilities(self) -> List[str]:
        """Получение возможностей Compute Core"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/",
                    timeout=5.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data.get("capabilities", [])
                else:
                    return []
                    
        except Exception as e:
            logger.error(f"Error getting capabilities: {e}")
            return []
