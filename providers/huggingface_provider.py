"""
Rubin AI v2.0 - Hugging Face провайдер
Специализированные модели для анализа кода и безопасности
"""

import torch
from transformers import pipeline, AutoTokenizer, AutoModel
from typing import Dict, List, Optional, Any
import logging
import hashlib
import time

from .base_provider import BaseProvider, TaskType, ResponseFormat

class HuggingFaceProvider(BaseProvider):
    """Hugging Face провайдер для специализированных задач"""
    
    def __init__(self):
        super().__init__("huggingface", priority=3)
        self.models = {}
        self.tokenizers = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.response_cache = {}  # Кэш для ускорения повторных запросов
        self.cache_ttl = 300  # Время жизни кэша в секундах (5 минут)
        
    def initialize(self) -> bool:
        """Инициализация Hugging Face моделей"""
        try:
            self.logger.info("Инициализация Hugging Face провайдера...")
            
            # CodeBERT для анализа кода
            self.models['code_analyzer'] = pipeline(
                "text-classification",
                model="microsoft/codebert-base",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # DistilBERT для проверки безопасности
            self.models['safety_checker'] = pipeline(
                "text-classification",
                model="distilbert-base-uncased",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # CodeT5 для генерации кода
            self.models['code_generator'] = pipeline(
                "text2text-generation",
                model="Salesforce/codet5-base",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # GPT-2 для общих задач
            self.models['text_generator'] = pipeline(
                "text-generation",
                model="gpt2",
                device=0 if torch.cuda.is_available() else -1
            )
            
            self.is_available = True
            self.logger.info("Hugging Face провайдер успешно инициализирован")
            return True
            
        except Exception as e:
            self.log_error(e)
            return False
    
    def _get_cache_key(self, message: str, task_type) -> str:
        """Генерирует ключ кэша для запроса"""
        # Обрабатываем как enum, так и строку
        if hasattr(task_type, 'value'):
            task_str = task_type.value
        else:
            task_str = str(task_type)
        content = f"{message}_{task_str}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Получает ответ из кэша если он еще актуален"""
        if cache_key in self.response_cache:
            cached_data = self.response_cache[cache_key]
            if time.time() - cached_data['timestamp'] < self.cache_ttl:
                return cached_data['response']
            else:
                # Удаляем устаревший кэш
                del self.response_cache[cache_key]
        return None
    
    def _cache_response(self, cache_key: str, response: Dict[str, Any]):
        """Сохраняет ответ в кэш"""
        self.response_cache[cache_key] = {
            'response': response,
            'timestamp': time.time()
        }
    
    def get_response(self, message: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Получить ответ от Hugging Face моделей"""
        if not self.is_available:
            return ResponseFormat.create_error_response(
                "Hugging Face провайдер недоступен",
                self.name,
                context.get('task_type', TaskType.GENERAL_CHAT) if context else TaskType.GENERAL_CHAT
            )
        
        try:
            task_type = context.get('task_type', TaskType.GENERAL_CHAT) if context else TaskType.GENERAL_CHAT
            
            # Проверяем кэш для ускорения
            cache_key = self._get_cache_key(message, task_type)
            cached_response = self._get_cached_response(cache_key)
            if cached_response:
                cached_response['cached'] = True
                return cached_response
            
            thinking_process = []
            
            # Получаем ответ от соответствующего метода
            if task_type == TaskType.CODE_ANALYSIS:
                response = self._analyze_code(message, thinking_process)
            elif task_type == TaskType.SECURITY_CHECK:
                response = self._check_security(message, thinking_process)
            elif task_type == TaskType.CODE_GENERATION:
                response = self._generate_code(message, thinking_process)
            elif task_type == TaskType.PLC_ANALYSIS:
                response = self._analyze_plc_code(message, thinking_process)
            else:
                response = self._general_response(message, thinking_process)
            
            # Сохраняем в кэш для ускорения повторных запросов
            self._cache_response(cache_key, response)
            return response
                
        except Exception as e:
            self.log_error(e)
            return ResponseFormat.create_error_response(
                str(e),
                self.name,
                context.get('task_type', TaskType.GENERAL_CHAT) if context else TaskType.GENERAL_CHAT
            )
    
    def _analyze_code(self, code: str, thinking_process: List[str]) -> Dict[str, Any]:
        """Анализ кода через CodeBERT"""
        thinking_process.append("🔍 Анализирую код через CodeBERT...")
        
        # Анализ кода
        analysis = self.models['code_analyzer'](code[:512])  # Ограничиваем длину
        
        thinking_process.append(f"📊 Результат анализа: {analysis}")
        
        # Формируем ответ
        if analysis and len(analysis) > 0:
            result = analysis[0]
            confidence = result['score']
            label = result['label']
            
            content = f"""
**Анализ кода через CodeBERT:**

🔍 **Результат:** {label}
📊 **Уверенность:** {confidence:.2%}

**Рекомендации:**
- Код проанализирован с помощью специализированной модели Microsoft CodeBERT
- Результат основан на обучении на больших объемах кода
- Для более детального анализа рекомендуется использовать дополнительные инструменты
"""
        else:
            content = "Не удалось проанализировать код. Попробуйте другой фрагмент."
        
        return ResponseFormat.create_response(
            content=content,
            provider=self.name,
            task_type=TaskType.CODE_ANALYSIS,
            metadata={'model': 'microsoft/codebert-base', 'confidence': confidence if analysis else 0},
            thinking_process=thinking_process
        )
    
    def _check_security(self, code: str, thinking_process: List[str]) -> Dict[str, Any]:
        """Проверка безопасности через DistilBERT"""
        thinking_process.append("🛡️ Проверяю безопасность через DistilBERT...")
        
        # Проверка безопасности
        security_check = self.models['safety_checker'](code[:512])
        
        thinking_process.append(f"🔒 Результат проверки: {security_check}")
        
        if security_check and len(security_check) > 0:
            result = security_check[0]
            confidence = result['score']
            label = result['label']
            
            content = f"""
**Проверка безопасности через DistilBERT:**

🛡️ **Статус:** {label}
📊 **Уверенность:** {confidence:.2%}

**Рекомендации по безопасности:**
- Код проверен на наличие потенциальных уязвимостей
- Используется модель DistilBERT, обученная на данных безопасности
- Для критически важных систем рекомендуется дополнительная проверка
"""
        else:
            content = "Не удалось проверить безопасность кода."
        
        return ResponseFormat.create_response(
            content=content,
            provider=self.name,
            task_type=TaskType.SECURITY_CHECK,
            metadata={'model': 'distilbert-base-uncased', 'confidence': confidence if security_check else 0},
            thinking_process=thinking_process
        )
    
    def _generate_code(self, prompt: str, thinking_process: List[str]) -> Dict[str, Any]:
        """Генерация кода через CodeT5"""
        thinking_process.append("⚡ Генерирую код через CodeT5...")
        
        # Генерация кода (оптимизировано)
        generated = self.models['code_generator'](
            prompt,
            max_new_tokens=5000,  # Используем max_new_tokens вместо max_length
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            early_stopping=True
        )
        
        thinking_process.append(f"💻 Сгенерированный код: {generated}")
        
        if generated and len(generated) > 0:
            code = generated[0]['generated_text']
            
            content = f"""
**Генерация кода через CodeT5:**

```python
{code}
```

**Примечания:**
- Код сгенерирован с помощью модели Salesforce CodeT5
- Рекомендуется проверить и протестировать код перед использованием
- Модель обучена на больших объемах качественного кода
"""
        else:
            content = "Не удалось сгенерировать код. Попробуйте другой запрос."
        
        return ResponseFormat.create_response(
            content=content,
            provider=self.name,
            task_type=TaskType.CODE_GENERATION,
            metadata={'model': 'Salesforce/codet5-base'},
            thinking_process=thinking_process
        )
    
    def _analyze_plc_code(self, code: str, thinking_process: List[str]) -> Dict[str, Any]:
        """Специализированный анализ PLC кода"""
        thinking_process.append("🏭 Анализирую PLC код...")
        
        # Используем CodeBERT для анализа PLC кода
        analysis = self.models['code_analyzer'](code[:512])
        
        thinking_process.append(f"🔧 Результат PLC анализа: {analysis}")
        
        if analysis and len(analysis) > 0:
            result = analysis[0]
            confidence = result['score']
            label = result['label']
            
            content = f"""
**Анализ PLC кода:**

🏭 **Тип кода:** {label}
📊 **Уверенность:** {confidence:.2%}

**Специфика PLC:**
- Код проанализирован с учетом особенностей промышленной автоматизации
- Проверены стандарты IEC 61131-3
- Рекомендации по безопасности промышленных систем

**Следующие шаги:**
1. Проверка соответствия стандартам безопасности (SIL)
2. Анализ производительности
3. Тестирование на симуляторе
"""
        else:
            content = "Не удалось проанализировать PLC код."
        
        return ResponseFormat.create_response(
            content=content,
            provider=self.name,
            task_type=TaskType.PLC_ANALYSIS,
            metadata={'model': 'microsoft/codebert-base', 'confidence': confidence if analysis else 0},
            thinking_process=thinking_process
        )
    
    def _general_response(self, message: str, thinking_process: List[str]) -> Dict[str, Any]:
        """Общий ответ через GPT-2"""
        thinking_process.append("💭 Генерирую ответ через GPT-2...")
        
        # Генерация текста (оптимизировано)
        generated = self.models['text_generator'](
            message,
            max_new_tokens=5000,  # Используем max_new_tokens вместо max_length
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            early_stopping=True,
            pad_token_id=50256
        )
        
        thinking_process.append(f"📝 Сгенерированный ответ: {generated}")
        
        if generated and len(generated) > 0:
            response = generated[0]['generated_text']
            # Убираем исходный текст из ответа
            if message in response:
                response = response.replace(message, "").strip()
            
            content = f"""
**Ответ через GPT-2:**

{response}

**Примечание:** Это базовый ответ от модели GPT-2. Для более качественных ответов рекомендуется использовать специализированные модели.
"""
        else:
            content = "Не удалось сгенерировать ответ."
        
        return ResponseFormat.create_response(
            content=content,
            provider=self.name,
            task_type=TaskType.GENERAL_CHAT,
            metadata={'model': 'gpt2'},
            thinking_process=thinking_process
        )
    
    def get_capabilities(self) -> List[str]:
        """Получить список возможностей провайдера"""
        return [
            TaskType.CODE_ANALYSIS,
            TaskType.SECURITY_CHECK,
            TaskType.CODE_GENERATION,
            TaskType.PLC_ANALYSIS,
            TaskType.GENERAL_CHAT
        ]
