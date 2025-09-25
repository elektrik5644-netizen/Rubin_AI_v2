"""
Rubin AI v2.0 - Google Cloud AI провайдер
Vision AI, Natural Language API, Speech-to-Text
"""

import os
import json
from typing import Dict, List, Optional, Any
import logging

try:
    from google.cloud import vision, language, speech
    from google.oauth2 import service_account
    GOOGLE_CLOUD_AVAILABLE = True
except ImportError:
    GOOGLE_CLOUD_AVAILABLE = False

from .base_provider import BaseProvider, TaskType, ResponseFormat

class GoogleCloudProvider(BaseProvider):
    """Google Cloud AI провайдер для анализа изображений и текста"""
    
    def __init__(self):
        super().__init__("google_cloud", priority=4)
        self.vision_client = None
        self.nlp_client = None
        self.speech_client = None
        self.credentials = None
        
    def initialize(self) -> bool:
        """Инициализация Google Cloud клиентов"""
        if not GOOGLE_CLOUD_AVAILABLE:
            self.logger.warning("Google Cloud библиотеки не установлены")
            return False
            
        try:
            self.logger.info("Инициализация Google Cloud провайдера...")
            
            # Загружаем учетные данные
            credentials_path = os.getenv('GOOGLE_CLOUD_CREDENTIALS')
            if credentials_path and os.path.exists(credentials_path):
                self.credentials = service_account.Credentials.from_service_account_file(
                    credentials_path
                )
            else:
                # Пытаемся использовать переменную окружения
                credentials_json = os.getenv('GOOGLE_CLOUD_CREDENTIALS_JSON')
                if credentials_json:
                    credentials_info = json.loads(credentials_json)
                    self.credentials = service_account.Credentials.from_service_account_info(
                        credentials_info
                    )
                else:
                    # Используем Application Default Credentials
                    self.credentials = None
            
            # Инициализируем клиентов
            if self.credentials:
                self.vision_client = vision.ImageAnnotatorClient(credentials=self.credentials)
                self.nlp_client = language.LanguageServiceClient(credentials=self.credentials)
                self.speech_client = speech.SpeechClient(credentials=self.credentials)
            else:
                self.vision_client = vision.ImageAnnotatorClient()
                self.nlp_client = language.LanguageServiceClient()
                self.speech_client = speech.SpeechClient()
            
            self.is_available = True
            self.logger.info("Google Cloud провайдер успешно инициализирован")
            return True
            
        except Exception as e:
            self.log_error(e)
            return False
    
    def get_response(self, message: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Получить ответ от Google Cloud AI"""
        if not self.is_available:
            return ResponseFormat.create_error_response(
                "Google Cloud провайдер недоступен",
                self.name,
                context.get('task_type', TaskType.GENERAL_CHAT) if context else TaskType.GENERAL_CHAT
            )
        
        try:
            task_type = context.get('task_type', TaskType.GENERAL_CHAT) if context else TaskType.GENERAL_CHAT
            thinking_process = []
            
            if task_type == TaskType.IMAGE_ANALYSIS:
                image_path = context.get('image_path') if context else None
                return self._analyze_image(image_path, thinking_process)
            elif task_type == TaskType.SPEECH_TO_TEXT:
                audio_path = context.get('audio_path') if context else None
                return self._speech_to_text(audio_path, thinking_process)
            elif task_type == TaskType.TECHNICAL_DOCUMENTATION:
                return self._analyze_documentation(message, thinking_process)
            elif task_type == TaskType.SCHEMATIC_ANALYSIS:
                image_path = context.get('image_path') if context else None
                return self._analyze_schematic(image_path, thinking_process)
            else:
                return self._analyze_text(message, thinking_process)
                
        except Exception as e:
            self.log_error(e)
            return ResponseFormat.create_error_response(
                str(e),
                self.name,
                context.get('task_type', TaskType.GENERAL_CHAT) if context else TaskType.GENERAL_CHAT
            )
    
    def _analyze_image(self, image_path: str, thinking_process: List[str]) -> Dict[str, Any]:
        """Анализ изображения через Google Vision AI"""
        if not image_path or not os.path.exists(image_path):
            return ResponseFormat.create_error_response(
                "Файл изображения не найден",
                self.name,
                TaskType.IMAGE_ANALYSIS
            )
        
        thinking_process.append("🖼️ Анализирую изображение через Google Vision AI...")
        
        try:
            with open(image_path, 'rb') as image_file:
                content = image_file.read()
            
            image = vision.Image(content=content)
            
            # Выполняем различные типы анализа
            responses = {}
            
            # Обнаружение текста
            text_response = self.vision_client.text_detection(image=image)
            texts = text_response.text_annotations
            responses['text'] = [text.description for text in texts] if texts else []
            
            # Обнаружение объектов
            objects_response = self.vision_client.object_localization(image=image)
            objects = objects_response.localized_object_annotations
            responses['objects'] = [obj.name for obj in objects] if objects else []
            
            # Обнаружение меток
            labels_response = self.vision_client.label_detection(image=image)
            labels = labels_response.label_annotations
            responses['labels'] = [label.description for label in labels] if labels else []
            
            thinking_process.append(f"📊 Найдено: {len(responses['text'])} текстов, {len(responses['objects'])} объектов, {len(responses['labels'])} меток")
            
            # Формируем ответ
            content = f"""
**Анализ изображения через Google Vision AI:**

📝 **Обнаруженный текст:**
{chr(10).join(f"- {text}" for text in responses['text'][:5])}

🏷️ **Объекты:**
{chr(10).join(f"- {obj}" for obj in responses['objects'][:5])}

🔖 **Метки:**
{chr(10).join(f"- {label}" for label in responses['labels'][:5])}

**Возможности:**
- Распознавание текста (OCR)
- Обнаружение объектов
- Классификация изображений
- Анализ лиц и эмоций
"""
            
            return ResponseFormat.create_response(
                content=content,
                provider=self.name,
                task_type=TaskType.IMAGE_ANALYSIS,
                metadata={
                    'text_count': len(responses['text']),
                    'objects_count': len(responses['objects']),
                    'labels_count': len(responses['labels']),
                    'image_path': image_path
                },
                thinking_process=thinking_process
            )
            
        except Exception as e:
            self.log_error(e)
            return ResponseFormat.create_error_response(
                f"Ошибка анализа изображения: {str(e)}",
                self.name,
                TaskType.IMAGE_ANALYSIS
            )
    
    def _analyze_schematic(self, image_path: str, thinking_process: List[str]) -> Dict[str, Any]:
        """Специализированный анализ схем и чертежей"""
        if not image_path or not os.path.exists(image_path):
            return ResponseFormat.create_error_response(
                "Файл схемы не найден",
                self.name,
                TaskType.SCHEMATIC_ANALYSIS
            )
        
        thinking_process.append("🔧 Анализирую схему через Google Vision AI...")
        
        try:
            with open(image_path, 'rb') as image_file:
                content = image_file.read()
            
            image = vision.Image(content=content)
            
            # Специализированный анализ для схем
            text_response = self.vision_client.text_detection(image=image)
            texts = text_response.text_annotations
            
            # Фильтруем технические термины
            technical_terms = []
            if texts:
                for text in texts:
                    text_lower = text.description.lower()
                    if any(term in text_lower for term in ['resistor', 'capacitor', 'transistor', 'ic', 'pin', 'vcc', 'gnd', 'plc', 'io']):
                        technical_terms.append(text.description)
            
            thinking_process.append(f"🔍 Найдено {len(technical_terms)} технических терминов")
            
            content = f"""
**Анализ схемы через Google Vision AI:**

🔧 **Технические элементы:**
{chr(10).join(f"- {term}" for term in technical_terms[:10])}

📝 **Весь обнаруженный текст:**
{chr(10).join(f"- {text.description}" for text in texts[:10]) if texts else "Текст не обнаружен"}

**Специализация для промышленности:**
- Распознавание компонентов схем
- Анализ соединений
- Идентификация разъемов и контактов
- Поддержка форматов: .sch, .brd, .kicad_pcb
"""
            
            return ResponseFormat.create_response(
                content=content,
                provider=self.name,
                task_type=TaskType.SCHEMATIC_ANALYSIS,
                metadata={
                    'technical_terms': technical_terms,
                    'total_texts': len(texts) if texts else 0,
                    'image_path': image_path
                },
                thinking_process=thinking_process
            )
            
        except Exception as e:
            self.log_error(e)
            return ResponseFormat.create_error_response(
                f"Ошибка анализа схемы: {str(e)}",
                self.name,
                TaskType.SCHEMATIC_ANALYSIS
            )
    
    def _speech_to_text(self, audio_path: str, thinking_process: List[str]) -> Dict[str, Any]:
        """Преобразование речи в текст"""
        if not audio_path or not os.path.exists(audio_path):
            return ResponseFormat.create_error_response(
                "Аудио файл не найден",
                self.name,
                TaskType.SPEECH_TO_TEXT
            )
        
        thinking_process.append("🎤 Преобразую речь в текст...")
        
        try:
            with open(audio_path, 'rb') as audio_file:
                content = audio_file.read()
            
            audio = speech.RecognitionAudio(content=content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code='ru-RU'
            )
            
            response = self.speech_client.recognize(config=config, audio=audio)
            
            results = []
            for result in response.results:
                results.append(result.alternatives[0].transcript)
            
            thinking_process.append(f"📝 Распознано: {len(results)} фрагментов")
            
            content = f"""
**Преобразование речи в текст:**

🎤 **Распознанный текст:**
{chr(10).join(f"- {text}" for text in results)}

**Возможности:**
- Поддержка русского и английского языков
- Высокая точность распознавания
- Обработка технических терминов
- Поддержка различных аудио форматов
"""
            
            return ResponseFormat.create_response(
                content=content,
                provider=self.name,
                task_type=TaskType.SPEECH_TO_TEXT,
                metadata={
                    'recognized_texts': results,
                    'audio_path': audio_path,
                    'language': 'ru-RU'
                },
                thinking_process=thinking_process
            )
            
        except Exception as e:
            self.log_error(e)
            return ResponseFormat.create_error_response(
                f"Ошибка распознавания речи: {str(e)}",
                self.name,
                TaskType.SPEECH_TO_TEXT
            )
    
    def _analyze_documentation(self, text: str, thinking_process: List[str]) -> Dict[str, Any]:
        """Анализ технической документации"""
        thinking_process.append("📚 Анализирую техническую документацию...")
        
        try:
            document = language.Document(
                content=text,
                type_=language.Document.Type.PLAIN_TEXT
            )
            
            # Анализ сущностей
            entities_response = self.nlp_client.analyze_entities(document=document)
            entities = entities_response.entities
            
            # Анализ тональности
            sentiment_response = self.nlp_client.analyze_sentiment(document=document)
            sentiment = sentiment_response.document_sentiment
            
            thinking_process.append(f"🔍 Найдено {len(entities)} сущностей, тональность: {sentiment.score}")
            
            # Фильтруем технические сущности
            technical_entities = [entity for entity in entities if entity.type_ in [language.Entity.Type.OTHER, language.Entity.Type.ORGANIZATION]]
            
            content = f"""
**Анализ технической документации:**

🔍 **Технические сущности:**
{chr(10).join(f"- {entity.name} ({entity.type_.name})" for entity in technical_entities[:10])}

📊 **Тональность:**
- Оценка: {sentiment.score:.2f}
- Величина: {sentiment.magnitude:.2f}

**Возможности:**
- Извлечение технических терминов
- Анализ тональности документации
- Классификация контента
- Поддержка многоязычности
"""
            
            return ResponseFormat.create_response(
                content=content,
                provider=self.name,
                task_type=TaskType.TECHNICAL_DOCUMENTATION,
                metadata={
                    'entities_count': len(entities),
                    'technical_entities': [entity.name for entity in technical_entities],
                    'sentiment_score': sentiment.score,
                    'sentiment_magnitude': sentiment.magnitude
                },
                thinking_process=thinking_process
            )
            
        except Exception as e:
            self.log_error(e)
            return ResponseFormat.create_error_response(
                f"Ошибка анализа документации: {str(e)}",
                self.name,
                TaskType.TECHNICAL_DOCUMENTATION
            )
    
    def _analyze_text(self, text: str, thinking_process: List[str]) -> Dict[str, Any]:
        """Общий анализ текста"""
        thinking_process.append("📝 Анализирую текст...")
        
        try:
            document = language.Document(
                content=text,
                type_=language.Document.Type.PLAIN_TEXT
            )
            
            # Анализ сущностей
            entities_response = self.nlp_client.analyze_entities(document=document)
            entities = entities_response.entities
            
            thinking_process.append(f"🔍 Найдено {len(entities)} сущностей")
            
            content = f"""
**Анализ текста через Google Natural Language API:**

🔍 **Сущности:**
{chr(10).join(f"- {entity.name} ({entity.type_.name})" for entity in entities[:10])}

**Возможности:**
- Извлечение именованных сущностей
- Анализ тональности
- Классификация контента
- Поддержка множества языков
"""
            
            return ResponseFormat.create_response(
                content=content,
                provider=self.name,
                task_type=TaskType.GENERAL_CHAT,
                metadata={
                    'entities_count': len(entities),
                    'entities': [entity.name for entity in entities[:10]]
                },
                thinking_process=thinking_process
            )
            
        except Exception as e:
            self.log_error(e)
            return ResponseFormat.create_error_response(
                f"Ошибка анализа текста: {str(e)}",
                self.name,
                TaskType.GENERAL_CHAT
            )
    
    def get_capabilities(self) -> List[str]:
        """Получить список возможностей провайдера"""
        return [
            TaskType.IMAGE_ANALYSIS,
            TaskType.SPEECH_TO_TEXT,
            TaskType.TECHNICAL_DOCUMENTATION,
            TaskType.SCHEMATIC_ANALYSIS,
            TaskType.GENERAL_CHAT
        ]
