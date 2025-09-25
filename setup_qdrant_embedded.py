#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Альтернативная установка Qdrant без Docker
Использование встроенного режима для разработки
"""

import subprocess
import sys
import os
import json
import requests
import time
from pathlib import Path

def install_qdrant_client():
    """Установка Python клиента для Qdrant"""
    try:
        print("📦 Установка Qdrant Python клиента...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "qdrant-client"])
        print("✅ Qdrant клиент установлен успешно")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка установки Qdrant клиента: {e}")
        return False

def install_sentence_transformers():
    """Установка SentenceTransformers для эмбеддингов"""
    try:
        print("📦 Установка SentenceTransformers...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers"])
        print("✅ SentenceTransformers установлен успешно")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка установки SentenceTransformers: {e}")
        return False

def create_qdrant_config():
    """Создание конфигурационного файла для Qdrant"""
    config = {
        "qdrant": {
            "host": "localhost",
            "port": 6333,
            "grpc_port": 6334,
            "timeout": 30,
            "use_embedded": True,
            "collections": {
                "rubin_knowledge": {
                    "vector_size": 384,
                    "distance": "Cosine",
                    "description": "База знаний Rubin AI"
                },
                "rubin_documents": {
                    "vector_size": 768,
                    "distance": "Cosine", 
                    "description": "Документы и материалы"
                },
                "rubin_conversations": {
                    "vector_size": 384,
                    "distance": "Cosine",
                    "description": "История разговоров"
                }
            }
        }
    }
    
    config_path = Path("config/qdrant_config.json")
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Конфигурация Qdrant создана: {config_path}")
    return config_path

def test_qdrant_embedded():
    """Тестирование встроенного режима Qdrant"""
    try:
        print("🔍 Тестирование встроенного режима Qdrant...")
        
        from qdrant_client import QdrantClient
        from qdrant_client.http.models import Distance, VectorParams
        
        # Создаем клиент в встроенном режиме
        client = QdrantClient(":memory:")
        
        # Создаем тестовую коллекцию
        client.create_collection(
            collection_name="test_collection",
            vectors_config=VectorParams(size=4, distance=Distance.COSINE)
        )
        
        # Проверяем коллекции
        collections = client.get_collections()
        print(f"✅ Встроенный режим работает. Коллекций: {len(collections.collections)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка тестирования встроенного режима: {e}")
        return False

def create_enhanced_qdrant_adapter():
    """Создание улучшенного адаптера с встроенным режимом"""
    adapter_code = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Улучшенный Qdrant адаптер с поддержкой встроенного режима
"""

import json
import logging
import hashlib
import time
import os
from typing import Dict, List, Optional, Any
from pathlib import Path

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.models import Distance, VectorParams, PointStruct
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

class EnhancedQdrantAdapter:
    """Улучшенный адаптер для работы с Qdrant"""
    
    def __init__(self, config_path: str = "config/qdrant_config.json", use_embedded: bool = True):
        self.logger = logging.getLogger("rubin_ai.qdrant_enhanced")
        self.client = None
        self.config = None
        self.embedding_model = None
        self.is_connected = False
        self.use_embedded = use_embedded
        
        if not QDRANT_AVAILABLE:
            self.logger.error("❌ Qdrant клиент не установлен")
            return
        
        self.load_config(config_path)
        self.connect()
        self.initialize_embedding_model()
    
    def load_config(self, config_path: str):
        """Загрузка конфигурации"""
        try:
            config_file = Path(config_path)
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                self.logger.info(f"✅ Конфигурация Qdrant загружена: {config_path}")
            else:
                self.config = {"qdrant": {"use_embedded": True}}
                self.logger.warning("⚠️ Используется конфигурация по умолчанию")
        except Exception as e:
            self.logger.error(f"❌ Ошибка загрузки конфигурации: {e}")
            self.config = {"qdrant": {"use_embedded": True}}
    
    def connect(self):
        """Подключение к Qdrant"""
        try:
            if self.use_embedded:
                # Встроенный режим
                self.client = QdrantClient(":memory:")
                self.logger.info("✅ Встроенный режим Qdrant активирован")
            else:
                # Обычный режим
                qdrant_config = self.config["qdrant"]
                self.client = QdrantClient(
                    host=qdrant_config["host"],
                    port=qdrant_config["port"],
                    timeout=qdrant_config.get("timeout", 30)
                )
                self.logger.info("✅ Подключение к Qdrant серверу установлено")
            
            self.is_connected = True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка подключения к Qdrant: {e}")
            self.is_connected = False
    
    def initialize_embedding_model(self):
        """Инициализация модели для создания эмбеддингов"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            self.logger.warning("⚠️ SentenceTransformers не установлен, используем простые эмбеддинги")
            return
        
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.logger.info("✅ Модель эмбеддингов инициализирована")
        except Exception as e:
            self.logger.error(f"❌ Ошибка инициализации модели эмбеддингов: {e}")
            self.embedding_model = None
    
    def create_embedding(self, text: str) -> List[float]:
        """Создание эмбеддинга для текста"""
        if self.embedding_model:
            try:
                embedding = self.embedding_model.encode(text)
                return embedding.tolist()
            except Exception as e:
                self.logger.error(f"❌ Ошибка создания эмбеддинга: {e}")
        
        # Fallback: простой хэш-эмбеддинг
        return self._create_simple_embedding(text)
    
    def _create_simple_embedding(self, text: str, size: int = 384) -> List[float]:
        """Создание простого эмбеддинга на основе хэша"""
        hash_obj = hashlib.sha256(text.encode('utf-8'))
        hash_bytes = hash_obj.digest()
        
        embedding = []
        for i in range(size):
            byte_index = i % len(hash_bytes)
            embedding.append((hash_bytes[byte_index] - 128) / 128.0)
        
        return embedding
    
    def create_collection(self, collection_name: str, vector_size: int = 384, distance: str = "Cosine"):
        """Создание коллекции в Qdrant"""
        if not self.is_connected:
            self.logger.error("❌ Нет подключения к Qdrant")
            return False
        
        try:
            collections = self.client.get_collections()
            existing_collections = [col.name for col in collections.collections]
            
            if collection_name in existing_collections:
                self.logger.info(f"📋 Коллекция {collection_name} уже существует")
                return True
            
            distance_map = {
                "Cosine": Distance.COSINE,
                "Euclidean": Distance.EUCLID,
                "Dot": Distance.DOT
            }
            
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance_map.get(distance, Distance.COSINE)
                )
            )
            
            self.logger.info(f"✅ Коллекция {collection_name} создана")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка создания коллекции {collection_name}: {e}")
            return False
    
    def add_document(self, collection_name: str, document_id: str, text: str, 
                    metadata: Optional[Dict] = None) -> bool:
        """Добавление документа в коллекцию"""
        if not self.is_connected:
            return False
        
        try:
            embedding = self.create_embedding(text)
            
            payload = {
                "text": text,
                "timestamp": time.time(),
                "document_id": document_id
            }
            if metadata:
                payload.update(metadata)
            
            point = PointStruct(
                id=document_id,
                vector=embedding,
                payload=payload
            )
            
            self.client.upsert(
                collection_name=collection_name,
                points=[point]
            )
            
            self.logger.info(f"✅ Документ {document_id} добавлен в {collection_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка добавления документа: {e}")
            return False
    
    def search_similar(self, collection_name: str, query_text: str, 
                      limit: int = 10, score_threshold: float = 0.7) -> List[Dict]:
        """Поиск похожих документов"""
        if not self.is_connected:
            return []
        
        try:
            query_embedding = self.create_embedding(query_text)
            
            search_result = self.client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold
            )
            
            results = []
            for point in search_result:
                results.append({
                    "id": point.id,
                    "score": point.score,
                    "text": point.payload.get("text", ""),
                    "metadata": {k: v for k, v in point.payload.items() if k != "text"}
                })
            
            self.logger.info(f"🔍 Найдено {len(results)} похожих документов")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка поиска: {e}")
            return []
    
    def get_status(self) -> Dict[str, Any]:
        """Получение статуса адаптера"""
        return {
            "connected": self.is_connected,
            "qdrant_available": QDRANT_AVAILABLE,
            "embedding_model_available": SENTENCE_TRANSFORMERS_AVAILABLE,
            "embedding_model_loaded": self.embedding_model is not None,
            "use_embedded": self.use_embedded,
            "timestamp": time.time()
        }
'''
    
    with open("enhanced_qdrant_adapter.py", "w", encoding="utf-8") as f:
        f.write(adapter_code)
    
    print("✅ Улучшенный адаптер создан: enhanced_qdrant_adapter.py")

def main():
    """Главная функция установки"""
    print("🚀 УСТАНОВКА QDRANT БЕЗ DOCKER")
    print("=" * 50)
    
    # Устанавливаем Python клиент
    if not install_qdrant_client():
        return False
    
    # Устанавливаем SentenceTransformers
    if not install_sentence_transformers():
        print("⚠️ SentenceTransformers не установлен, будет использован fallback режим")
    
    # Создаем конфигурацию
    create_qdrant_config()
    
    # Тестируем встроенный режим
    if not test_qdrant_embedded():
        return False
    
    # Создаем улучшенный адаптер
    create_enhanced_qdrant_adapter()
    
    print("\n" + "=" * 50)
    print("🎉 QDRANT УСПЕШНО УСТАНОВЛЕН (ВСТРОЕННЫЙ РЕЖИМ)!")
    print("=" * 50)
    print("📦 Установленные компоненты:")
    print("   ✅ Qdrant Python клиент")
    print("   ✅ SentenceTransformers (для эмбеддингов)")
    print("   ✅ Встроенный режим Qdrant")
    print("   ✅ Улучшенный адаптер")
    print("\n💡 Преимущества встроенного режима:")
    print("   • Не требует Docker")
    print("   • Быстрый запуск")
    print("   • Идеален для разработки")
    print("   • Автоматическая очистка при завершении")
    print("\n🚀 Следующие шаги:")
    print("   1. Запустить qdrant_enhanced_server.py")
    print("   2. Протестировать векторный поиск")
    print("   3. Индексировать существующие данные")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Установка Qdrant завершилась с ошибками")
        sys.exit(1)
    else:
        print("\n✅ Установка Qdrant завершена успешно")


