#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль векторного поиска для Rubin AI v2.0
Поддерживает семантический поиск с использованием embeddings
"""

import os
import sys
import sqlite3
import numpy as np
import logging
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import json
import hashlib
from datetime import datetime

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️ sentence-transformers не установлен. Установите: pip install sentence-transformers")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️ faiss-cpu не установлен. Установите: pip install faiss-cpu")

from rubin_text_preprocessor import RubinTextPreprocessor # Добавляем для централизованной очистки текста

logger = logging.getLogger(__name__)

class VectorSearchEngine:
    """Движок векторного поиска для семантического поиска документов"""
    
    def __init__(self, db_path="rubin_ai_documents.db", model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.db_path = db_path
        self.model_name = model_name
        self.model = None
        self.index = None
        self.dimension = 384  # Размерность для multilingual-MiniLM-L12-v2
        self.setup_logging()
        self.text_preprocessor = RubinTextPreprocessor() # Инициализация препроцессора текста
        self.setup_database()
        self.load_model()
        
    def setup_logging(self):
        """Настройка логирования"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('vector_search.log', encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_database(self):
        """Создание таблиц для векторного поиска"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Таблица для хранения векторов документов
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS document_vectors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    vector_hash TEXT NOT NULL,
                    vector_data BLOB NOT NULL,
                    content_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document_id) REFERENCES documents (id)
                )
            ''')
            
            # Таблица для метаданных векторов
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS vector_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    content_preview TEXT,
                    keywords TEXT,
                    category TEXT,
                    similarity_threshold REAL DEFAULT 0.7,
                    FOREIGN KEY (document_id) REFERENCES documents (id)
                )
            ''')
            
            # Индекс для быстрого поиска
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_vector_hash ON document_vectors(vector_hash)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_document_id ON document_vectors(document_id)
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.info("✅ Таблицы векторного поиска инициализированы")
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка инициализации векторных таблиц: {e}")
            
    def load_model(self):
        """Загрузка модели для генерации embeddings"""

        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            self.logger.warning("⚠️ sentence-transformers недоступен, векторный поиск отключен")
            return
            
        try:
            self.logger.info(f"🔄 Загрузка модели: {self.model_name}")
            
            # Исправление ошибки PyTorch meta tensor
            import torch
            import os
            
            # Устанавливаем переменные окружения для исправления ошибки
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            
            # Загружаем модель с явным указанием устройства
            self.model = SentenceTransformer(self.model_name, device='cpu')
            
            # Дополнительная проверка и исправление
            if hasattr(self.model, '_modules'):
                for name, module in self.model._modules.items():
                    if hasattr(module, 'to'):
                        try:
                            module.to('cpu')
                        except Exception as module_error:
                            self.logger.warning(f"⚠️ Ошибка перемещения модуля {name}: {module_error}")
                            # Пытаемся исправить через to_empty
                            try:
                                if hasattr(module, 'to_empty'):
                                    module.to_empty(device='cpu')
                            except:
                                pass
            
            self.dimension = self.model.get_sentence_embedding_dimension()
            self.logger.info(f"✅ Модель загружена, размерность: {self.dimension}")
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка загрузки модели: {e}")
            
            # Попытка альтернативной загрузки
            try:
                self.logger.info("🔄 Попытка альтернативной загрузки модели...")
                
                # Очистка кэша
                import torch
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                # Загрузка с минимальными настройками
                self.model = SentenceTransformer(
                    self.model_name, 
                    device='cpu',
                    trust_remote_code=True
                )
                
                # Принудительная инициализация на CPU
                self.model.eval()
                self.dimension = self.model.get_sentence_embedding_dimension()
                self.logger.info(f"✅ Модель загружена альтернативным способом, размерность: {self.dimension}")
                
            except Exception as e2:
                self.logger.error(f"❌ Альтернативная загрузка также не удалась: {e2}")
                self.model = None
            
    def generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """Генерация embedding для текста"""
        if not self.model:
            return None
            
        try:
            # Очистка и подготовка текста
            clean_text = self.clean_text(text)
            if not clean_text:
                return None
                
            # Генерация embedding
            embedding = self.model.encode(clean_text, convert_to_numpy=True)
            return embedding.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка генерации embedding: {e}")
            return None
            
    def clean_text(self, text: str) -> str:
        """Очистка текста для лучшего качества embeddings с использованием RubinTextPreprocessor"""
        return self.text_preprocessor.preprocess_text(
            text,
            to_lower=True,
            remove_spaces=True,
            remove_special=True,
            limit_len=512 # Ограничение длины для embeddings
        )
        
    def vector_to_blob(self, vector: np.ndarray) -> bytes:
        """Конвертация вектора в BLOB для SQLite"""
        return vector.tobytes()
        
    def blob_to_vector(self, blob: bytes) -> np.ndarray:
        """Конвертация BLOB в вектор"""
        return np.frombuffer(blob, dtype=np.float32)
        
    def get_content_hash(self, content: str) -> str:
        """Получение хеша содержимого"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
        
    def index_document(self, document_id: int, content: str, metadata: Dict = None) -> bool:
        """Индексация документа в векторном пространстве"""
        if not self.model:
            self.logger.warning("⚠️ Модель не загружена, индексация пропущена")
            return False
            
        try:
            # Генерация embedding
            embedding = self.generate_embedding(content)
            if embedding is None:
                return False
                
            # Проверка существования
            content_hash = self.get_content_hash(content)
            vector_hash = hashlib.md5(embedding.tobytes()).hexdigest()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Проверка, не индексирован ли уже документ
            cursor.execute('''
                SELECT id FROM document_vectors 
                WHERE document_id = ? AND content_hash = ?
            ''', (document_id, content_hash))
            
            if cursor.fetchone():
                self.logger.info(f"📄 Документ {document_id} уже проиндексирован")
                conn.close()
                return True
                
            # Сохранение вектора
            vector_blob = self.vector_to_blob(embedding)
            cursor.execute('''
                INSERT INTO document_vectors 
                (document_id, vector_hash, vector_data, content_hash)
                VALUES (?, ?, ?, ?)
            ''', (document_id, vector_hash, vector_blob, content_hash))
            
            # Сохранение метаданных
            if metadata:
                cursor.execute('''
                    INSERT OR REPLACE INTO vector_metadata
                    (document_id, content_preview, keywords, category)
                    VALUES (?, ?, ?, ?)
                ''', (
                    document_id,
                    content[:200] + "..." if len(content) > 200 else content,
                    metadata.get('keywords', ''),
                    metadata.get('category', '')
                ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"✅ Документ {document_id} проиндексирован в векторном пространстве")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка индексации документа {document_id}: {e}")
            return False
            
    def build_faiss_index(self) -> bool:
        """Построение FAISS индекса для быстрого поиска"""
        if not FAISS_AVAILABLE:
            self.logger.warning("⚠️ FAISS недоступен, используется медленный поиск")
            return False
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Получение всех векторов
            cursor.execute('''
                SELECT dv.document_id, dv.vector_data, vm.content_preview, vm.category
                FROM document_vectors dv
                LEFT JOIN vector_metadata vm ON dv.document_id = vm.document_id
            ''')
            
            results = cursor.fetchall()
            conn.close()
            
            if not results:
                self.logger.warning("⚠️ Нет векторов для индексации")
                return False
                
            # Подготовка данных
            vectors = []
            self.document_ids = []
            self.metadata = []
            
            for doc_id, vector_blob, preview, category in results:
                vector = self.blob_to_vector(vector_blob)
                vectors.append(vector)
                self.document_ids.append(doc_id)
                self.metadata.append({
                    'preview': preview,
                    'category': category
                })
                
            # Создание FAISS индекса
            vectors_array = np.vstack(vectors).astype(np.float32)
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product для косинусного сходства
            
            # Нормализация векторов для косинусного сходства
            faiss.normalize_L2(vectors_array)
            self.index.add(vectors_array)
            
            self.logger.info(f"✅ FAISS индекс построен: {len(vectors)} векторов")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка построения FAISS индекса: {e}")
            return False
            
    def search_similar(self, query: str, top_k: int = 5, threshold: float = 0.7) -> List[Dict]:
        """Поиск похожих документов по запросу"""
        if not self.model:
            self.logger.warning("⚠️ Модель не загружена")
            return []
            
        try:
            # Генерация embedding для запроса
            query_embedding = self.generate_embedding(query)
            if query_embedding is None:
                return []
                
            # Нормализация для косинусного сходства
            query_embedding = query_embedding.reshape(1, -1)
            faiss.normalize_L2(query_embedding)
            
            results = []
            
            if self.index and FAISS_AVAILABLE:
                # Быстрый поиск через FAISS
                similarities, indices = self.index.search(query_embedding, min(top_k, len(self.document_ids)))
                
                for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                    if similarity >= threshold:
                        results.append({
                            'document_id': self.document_ids[idx],
                            'similarity': float(similarity),
                            'metadata': self.metadata[idx]
                        })
            else:
                # Медленный поиск через SQLite
                results = self._slow_search(query_embedding, top_k, threshold)
                
            # Сортировка по сходству
            results.sort(key=lambda x: x['similarity'], reverse=True)
            
            self.logger.info(f"🔍 Найдено {len(results)} похожих документов")
            return results[:top_k]
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка векторного поиска: {e}")
            return []
            
    def _slow_search(self, query_embedding: np.ndarray, top_k: int, threshold: float) -> List[Dict]:
        """Медленный поиск через SQLite (fallback)"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT dv.document_id, dv.vector_data, vm.content_preview, vm.category
                FROM document_vectors dv
                LEFT JOIN vector_metadata vm ON dv.document_id = vm.document_id
            ''')
            
            results = []
            for doc_id, vector_blob, preview, category in cursor.fetchall():
                vector = self.blob_to_vector(vector_blob)
                similarity = np.dot(query_embedding[0], vector)
                
                if similarity >= threshold:
                    results.append({
                        'document_id': doc_id,
                        'similarity': float(similarity),
                        'metadata': {
                            'preview': preview,
                            'category': category
                        }
                    })
                    
            conn.close()
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка медленного поиска: {e}")
            return []
            
    def get_document_content(self, document_id: int) -> Optional[str]:
        """Получение содержимого документа"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT content FROM documents WHERE id = ?
            ''', (document_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            return result[0] if result else None
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка получения документа {document_id}: {e}")
            return None
            
    def get_stats(self) -> Dict:
        """Получение статистики векторного поиска"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Общее количество векторов
            cursor.execute("SELECT COUNT(*) FROM document_vectors")
            total_vectors = cursor.fetchone()[0]
            
            # Количество документов с векторами
            cursor.execute("SELECT COUNT(DISTINCT document_id) FROM document_vectors")
            documents_with_vectors = cursor.fetchone()[0]
            
            # Статистика по категориям
            cursor.execute('''
                SELECT vm.category, COUNT(*) 
                FROM vector_metadata vm
                GROUP BY vm.category
            ''')
            categories = dict(cursor.fetchall())
            
            conn.close()
            
            return {
                'total_vectors': total_vectors,
                'documents_with_vectors': documents_with_vectors,
                'categories': categories,
                'model_loaded': self.model is not None,
                'faiss_available': FAISS_AVAILABLE,
                'faiss_index_built': self.index is not None
            }
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка получения статистики: {e}")
            return {}

if __name__ == "__main__":
    # Тестирование векторного поиска
    engine = VectorSearchEngine()
    
    # Получение статистики
    stats = engine.get_stats()
    print("📊 Статистика векторного поиска:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
