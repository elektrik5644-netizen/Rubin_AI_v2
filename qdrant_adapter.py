#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qdrant адаптер для Rubin AI v2.0
Интеграция векторной базы данных для улучшения поиска и производительности
"""

import json
import logging
import hashlib
import time
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import numpy as np

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

class QdrantAdapter:
    """Адаптер для работы с Qdrant векторной базой данных"""
    
    def __init__(self, config_path: str = "config/qdrant_config.json"):
        self.logger = logging.getLogger("rubin_ai.qdrant")
        self.client = None
        self.config = None
        self.embedding_model = None
        self.is_connected = False
        
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
                # Создаем конфигурацию по умолчанию
                self.config = {
                    "qdrant": {
                        "host": "localhost",
                        "port": 6333,
                        "grpc_port": 6334,
                        "timeout": 30
                    }
                }
                self.logger.warning("⚠️ Используется конфигурация по умолчанию")
        except Exception as e:
            self.logger.error(f"❌ Ошибка загрузки конфигурации: {e}")
            self.config = {"qdrant": {"host": "localhost", "port": 6333}}
    
    def connect(self):
        """Подключение к Qdrant"""
        try:
            qdrant_config = self.config["qdrant"]
            self.client = QdrantClient(
                host=qdrant_config["host"],
                port=qdrant_config["port"],
                timeout=qdrant_config.get("timeout", 30)
            )
            
            # Проверяем подключение
            collections = self.client.get_collections()
            self.is_connected = True
            self.logger.info("✅ Подключение к Qdrant установлено")
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка подключения к Qdrant: {e}")
            self.is_connected = False
    
    def initialize_embedding_model(self):
        """Инициализация модели для создания эмбеддингов"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            self.logger.warning("⚠️ SentenceTransformers не установлен, используем простые эмбеддинги")
            return
        
        try:
            # Используем легкую модель для быстрой работы
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
        
        # Преобразуем хэш в вектор фиксированного размера
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
            # Проверяем, существует ли коллекция
            collections = self.client.get_collections()
            existing_collections = [col.name for col in collections.collections]
            
            if collection_name in existing_collections:
                self.logger.info(f"📋 Коллекция {collection_name} уже существует")
                return True
            
            # Создаем коллекцию
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
            # Создаем эмбеддинг
            embedding = self.create_embedding(text)
            
            # Подготавливаем метаданные
            payload = {
                "text": text,
                "timestamp": time.time(),
                "document_id": document_id
            }
            if metadata:
                payload.update(metadata)
            
            # Добавляем точку в коллекцию
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
            # Создаем эмбеддинг для запроса
            query_embedding = self.create_embedding(query_text)
            
            # Выполняем поиск
            search_result = self.client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold
            )
            
            # Форматируем результаты
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
    
    def hybrid_search(self, collection_name: str, query_text: str, 
                     filters: Optional[Dict] = None, limit: int = 10) -> List[Dict]:
        """Гибридный поиск с фильтрацией"""
        if not self.is_connected:
            return []
        
        try:
            query_embedding = self.create_embedding(query_text)
            
            # Подготавливаем фильтры
            query_filter = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    conditions.append({
                        "key": key,
                        "match": {"value": value}
                    })
                
                if conditions:
                    query_filter = {"must": conditions}
            
            # Выполняем поиск с фильтрами
            search_result = self.client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                query_filter=query_filter,
                limit=limit
            )
            
            results = []
            for point in search_result:
                results.append({
                    "id": point.id,
                    "score": point.score,
                    "text": point.payload.get("text", ""),
                    "metadata": {k: v for k, v in point.payload.items() if k != "text"}
                })
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка гибридного поиска: {e}")
            return []
    
    def get_collection_info(self, collection_name: str) -> Optional[Dict]:
        """Получение информации о коллекции"""
        if not self.is_connected:
            return None
        
        try:
            collection_info = self.client.get_collection(collection_name)
            return {
                "name": collection_name,
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "points_count": collection_info.points_count,
                "status": collection_info.status
            }
        except Exception as e:
            self.logger.error(f"❌ Ошибка получения информации о коллекции: {e}")
            return None
    
    def optimize_collection(self, collection_name: str):
        """Оптимизация коллекции"""
        if not self.is_connected:
            return False
        
        try:
            # Обновляем конфигурацию коллекции для оптимизации
            self.client.update_collection(
                collection_name=collection_name,
                optimizer_config=models.OptimizersConfigDiff(
                    default_segment_number=2,
                    max_segment_size=20000,
                    memmap_threshold=50000,
                    indexing_threshold=20000
                )
            )
            
            self.logger.info(f"✅ Коллекция {collection_name} оптимизирована")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка оптимизации коллекции: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Получение статуса адаптера"""
        return {
            "connected": self.is_connected,
            "qdrant_available": QDRANT_AVAILABLE,
            "embedding_model_available": SENTENCE_TRANSFORMERS_AVAILABLE,
            "embedding_model_loaded": self.embedding_model is not None,
            "config_loaded": self.config is not None,
            "timestamp": time.time()
        }

def main():
    """Тестирование Qdrant адаптера"""
    logging.basicConfig(level=logging.INFO)
    
    print("🔍 ТЕСТИРОВАНИЕ QDRANT АДАПТЕРА")
    print("=" * 40)
    
    # Создаем адаптер
    adapter = QdrantAdapter()
    
    # Проверяем статус
    status = adapter.get_status()
    print(f"📊 Статус адаптера:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    if not status["connected"]:
        print("❌ Нет подключения к Qdrant")
        return
    
    # Создаем тестовую коллекцию
    collection_name = "test_rubin_knowledge"
    if adapter.create_collection(collection_name):
        print(f"✅ Коллекция {collection_name} создана")
        
        # Добавляем тестовые документы
        test_documents = [
            ("doc1", "Python - это мощный язык программирования"),
            ("doc2", "Электротехника изучает электрические цепи и компоненты"),
            ("doc3", "Автоматизация использует PLC контроллеры для управления процессами")
        ]
        
        for doc_id, text in test_documents:
            if adapter.add_document(collection_name, doc_id, text):
                print(f"✅ Документ {doc_id} добавлен")
        
        # Тестируем поиск
        query = "программирование Python"
        results = adapter.search_similar(collection_name, query, limit=3)
        
        print(f"\n🔍 Результаты поиска для '{query}':")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['text']} (score: {result['score']:.3f})")
        
        # Получаем информацию о коллекции
        info = adapter.get_collection_info(collection_name)
        if info:
            print(f"\n📋 Информация о коллекции:")
            for key, value in info.items():
                print(f"  {key}: {value}")

if __name__ == "__main__":
    main()


