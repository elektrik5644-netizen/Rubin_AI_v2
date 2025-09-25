"""
Сервис для работы с Ollama AI
"""

import httpx
import logging
import time
from typing import Dict, List, Optional, Any
from app.core.config import settings

logger = logging.getLogger(__name__)

class OllamaService:
    """Сервис для работы с Ollama AI"""
    
    def __init__(self):
        self.base_url = settings.ollama_url
        self.default_model = "phi3"
        self.timeout = 30.0
    
    async def generate_response(
        self, 
        prompt: str, 
        context: str = "", 
        system_prompt: str = "",
        model: str = None,
        options: Dict = None
    ) -> str:
        """Генерация ответа с помощью Ollama"""
        try:
            model = model or self.default_model
            
            # Формирование полного промпта
            full_prompt = self._build_prompt(prompt, context, system_prompt)
            
            # Параметры генерации
            generation_options = {
                "model": model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 2000,
                    **(options or {})
                }
            }
            
            logger.info(f"Generating response with model {model}")
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=generation_options,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get("response", "No response generated")
                else:
                    logger.error(f"Ollama API error: {response.status_code}")
                    raise Exception(f"Ollama API error: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    async def generate_streaming_response(
        self, 
        prompt: str, 
        context: str = "", 
        system_prompt: str = "",
        model: str = None
    ):
        """Генерация ответа с потоковой передачей"""
        try:
            model = model or self.default_model
            
            # Формирование полного промпта
            full_prompt = self._build_prompt(prompt, context, system_prompt)
            
            # Параметры генерации
            generation_options = {
                "model": model,
                "prompt": full_prompt,
                "stream": True,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 2000
                }
            }
            
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/generate",
                    json=generation_options,
                    timeout=self.timeout
                ) as response:
                    if response.status_code == 200:
                        async for line in response.aiter_lines():
                            if line.strip():
                                try:
                                    data = response.json()
                                    if "response" in data:
                                        yield data["response"]
                                    if data.get("done", False):
                                        break
                                except:
                                    continue
                    else:
                        raise Exception(f"Ollama API error: {response.status_code}")
                        
        except Exception as e:
            logger.error(f"Error generating streaming response: {e}")
            raise
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """Получение списка доступных моделей"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/api/tags",
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get("models", [])
                else:
                    logger.error(f"Error getting models: {response.status_code}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    async def pull_model(self, model_name: str) -> bool:
        """Загрузка модели"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/pull",
                    json={"name": model_name},
                    timeout=300.0  # 5 минут для загрузки модели
                )
                
                if response.status_code == 200:
                    logger.info(f"Model {model_name} pulled successfully")
                    return True
                else:
                    logger.error(f"Error pulling model: {response.status_code}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            return False
    
    async def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Получение информации о модели"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/show",
                    json={"name": model_name},
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Error getting model info: {response.status_code}")
                    return {}
                    
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {}
    
    async def health_check(self) -> Dict[str, Any]:
        """Проверка здоровья Ollama сервиса"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/api/tags",
                    timeout=5.0
                )
                
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    return {
                        "status": "healthy",
                        "models_count": len(models),
                        "models": [model.get("name", "") for model in models],
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
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
    
    def _build_prompt(self, prompt: str, context: str = "", system_prompt: str = "") -> str:
        """Построение полного промпта"""
        full_prompt = ""
        
        if system_prompt:
            full_prompt += f"System: {system_prompt}\n\n"
        
        if context:
            full_prompt += f"Context: {context}\n\n"
        
        full_prompt += f"User: {prompt}\n\nAssistant:"
        
        return full_prompt
    
    async def analyze_code(self, code: str, language: str = "python") -> str:
        """Анализ кода с помощью AI"""
        system_prompt = f"""Ты - эксперт по анализу кода на языке {language}.
        Проанализируй предоставленный код и дай подробный анализ:
        1. Качество кода и соответствие стандартам
        2. Потенциальные ошибки и проблемы
        3. Рекомендации по улучшению
        4. Анализ безопасности
        5. Производительность
        
        Отвечай на русском языке."""
        
        return await self.generate_response(
            prompt=f"Проанализируй этот код на {language}:\n\n```{language}\n{code}\n```",
            system_prompt=system_prompt
        )
    
    async def generate_plc_code(self, description: str, language: str = "ladder") -> str:
        """Генерация PLC кода"""
        system_prompt = f"""Ты - эксперт по программированию PLC на языке {language}.
        Сгенерируй код PLC на основе описания задачи.
        Используй стандартные функции и следуй лучшим практикам.
        Отвечай на русском языке."""
        
        return await self.generate_response(
            prompt=f"Сгенерируй PLC код на {language} для: {description}",
            system_prompt=system_prompt
        )
    
    async def explain_pmac_command(self, command: str) -> str:
        """Объяснение команды PMAC"""
        system_prompt = """Ты - эксперт по контроллерам PMAC.
        Объясни команду PMAC, её назначение, параметры и примеры использования.
        Отвечай на русском языке."""
        
        return await self.generate_response(
            prompt=f"Объясни команду PMAC: {command}",
            system_prompt=system_prompt
        )
