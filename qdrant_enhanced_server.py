#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rubin AI v2.0 с интеграцией Qdrant
Высокопроизводительный сервер с векторным поиском
"""

import os
import sys
import json
import logging
import gc
import sqlite3
import time
from datetime import datetime
from typing import Optional, Dict, List, Any
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Добавляем пути к модулям
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Импорт адаптеров
try:
    from enhanced_qdrant_adapter import EnhancedQdrantAdapter as QdrantAdapter
    QDRANT_ADAPTER_AVAILABLE = True
except ImportError:
    QDRANT_ADAPTER_AVAILABLE = False

try:
    from memory_optimizer import MemoryOptimizer
    MEMORY_OPTIMIZER_AVAILABLE = True
except ImportError:
    MEMORY_OPTIMIZER_AVAILABLE = False

# Инициализация Flask приложения
app = Flask(__name__)
CORS(app, 
     origins=['http://localhost:8084', 'http://localhost:8085', 'http://127.0.0.1:8084', 'http://127.0.0.1:8085'],
     allow_headers=['Content-Type', 'Authorization'],
     methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
     supports_credentials=True)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rubin_qdrant.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("rubin_ai_qdrant")

class QdrantEnhancedRubinAI:
    """Rubin AI с интеграцией Qdrant для векторного поиска"""
    
    def __init__(self):
        self.knowledge_base = {}
        self.conversation_history = []
        self.response_cache = {}
        self.cache_ttl = 300
        self.max_cache_size = 1000
        
        # Инициализация Qdrant адаптера в встроенном режиме
        if QDRANT_ADAPTER_AVAILABLE:
            self.qdrant_adapter = QdrantAdapter(use_embedded=True)
            logger.info("✅ Qdrant адаптер инициализирован (встроенный режим)")
        else:
            self.qdrant_adapter = None
            logger.warning("⚠️ Qdrant адаптер недоступен")
        
        # Инициализация оптимизатора памяти
        if MEMORY_OPTIMIZER_AVAILABLE:
            self.memory_optimizer = MemoryOptimizer(threshold_mb=400, cleanup_interval=180)
            self.memory_optimizer.start_monitoring()
            logger.info("✅ Оптимизатор памяти активирован")
        else:
            self.memory_optimizer = None
            logger.warning("⚠️ Оптимизатор памяти недоступен")
        
        self.load_knowledge_base()
        self.initialize_qdrant_collections()
        logger.info("🚀 Qdrant-улучшенный Rubin AI инициализирован")
    
    def load_knowledge_base(self):
        """Загрузка базы знаний с интеграцией в Qdrant"""
        try:
            # Загружаем из SQLite
            db_files = [
                'rubin_knowledge_base.db',
                'rubin_ai_knowledge.db',
                'rubin_knowledge.db'
            ]
            
            for db_file in db_files:
                if os.path.exists(db_file):
                    logger.info(f"📚 Загружаем базу знаний: {db_file}")
                    conn = sqlite3.connect(db_file)
                    cursor = conn.cursor()
                    
                    # Получаем данные из таблиц
                    tables_to_process = [
                        'knowledge_base', 'knowledge', 'knowledge_entries'
                    ]
                    
                    for table in tables_to_process:
                        try:
                            cursor.execute(f"SELECT * FROM {table}")
                            rows = cursor.fetchall()
                            
                            if rows:
                                logger.info(f"  📋 Обрабатываем таблицу {table}: {len(rows)} записей")
                                self._process_knowledge_table(table, rows, cursor.description)
                                
                        except sqlite3.OperationalError:
                            # Таблица не существует
                            continue
                    
                    conn.close()
                    break
            else:
                logger.info("📚 База знаний не найдена, создаем новую")
                self.create_default_knowledge_base()
                
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки базы знаний: {e}")
            self.create_default_knowledge_base()
    
    def _process_knowledge_table(self, table_name: str, rows: List, columns: List):
        """Обработка таблицы знаний и добавление в Qdrant"""
        if not self.qdrant_adapter:
            return
        
        column_names = [desc[0] for desc in columns]
        
        for row in rows:
            try:
                # Создаем словарь из строки
                row_dict = dict(zip(column_names, row))
                
                # Извлекаем текст для индексации
                text_fields = ['content', 'text', 'description', 'answer', 'question']
                text_content = ""
                
                for field in text_fields:
                    if field in row_dict and row_dict[field]:
                        text_content += str(row_dict[field]) + " "
                
                if text_content.strip():
                    # Создаем уникальный ID
                    doc_id = f"{table_name}_{row[0]}" if row else f"{table_name}_{len(self.knowledge_base)}"
                    
                    # Добавляем в Qdrant
                    metadata = {
                        "table": table_name,
                        "timestamp": time.time(),
                        **{k: v for k, v in row_dict.items() if k not in text_fields}
                    }
                    
                    self.qdrant_adapter.add_document(
                        "rubin_knowledge", 
                        doc_id, 
                        text_content.strip(),
                        metadata
                    )
                    
            except Exception as e:
                logger.error(f"❌ Ошибка обработки строки из {table_name}: {e}")
    
    def initialize_qdrant_collections(self):
        """Инициализация коллекций Qdrant"""
        if not self.qdrant_adapter:
            return
        
        collections = [
            ("rubin_knowledge", 384, "База знаний Rubin AI"),
            ("rubin_documents", 768, "Документы и материалы"),
            ("rubin_conversations", 384, "История разговоров")
        ]
        
        for collection_name, vector_size, description in collections:
            if self.qdrant_adapter.create_collection(collection_name, vector_size):
                logger.info(f"✅ Коллекция {collection_name} инициализирована")
    
    def create_default_knowledge_base(self):
        """Создание базовой базы знаний"""
        self.knowledge_base = {
            'programming': {
                'keywords': ['python', 'программирование', 'код', 'алгоритм'],
                'responses': [
                    "Python - мощный язык программирования! Могу помочь с синтаксисом, библиотеками и архитектурой.",
                    "Основные концепции: переменные, функции, классы, модули, обработка исключений."
                ]
            },
            'electronics': {
                'keywords': ['электротехника', 'схема', 'транзистор', 'диод'],
                'responses': [
                    "Электротехника - основа современной техники! Закон Ома: U = I × R",
                    "Основные компоненты: резисторы, конденсаторы, катушки, полупроводники."
                ]
            },
            'automation': {
                'keywords': ['автоматизация', 'plc', 'контроллер', 'scada'],
                'responses': [
                    "Промышленная автоматизация включает PLC, SCADA, HMI системы.",
                    "Протоколы: Modbus, Profinet, Ethernet/IP, OPC UA."
                ]
            }
        }
        
        # Добавляем в Qdrant
        if self.qdrant_adapter:
            for category, data in self.knowledge_base.items():
                for i, response in enumerate(data['responses']):
                    doc_id = f"default_{category}_{i}"
                    self.qdrant_adapter.add_document(
                        "rubin_knowledge",
                        doc_id,
                        response,
                        {"category": category, "type": "default_response"}
                    )
    
    def get_response(self, message: str, context: Optional[dict] = None) -> dict:
        """Получить ответ с использованием векторного поиска"""
        try:
            # Проверяем кэш
            cache_key = hash(message.lower().strip())
            if cache_key in self.response_cache:
                cache_entry = self.response_cache[cache_key]
                if time.time() - cache_entry['timestamp'] < self.cache_ttl:
                    logger.info("📋 Ответ из кэша")
                    return cache_entry['response']
                else:
                    del self.response_cache[cache_key]
            
            # Генерируем новый ответ
            response = self._generate_enhanced_response(message, context)
            
            # Сохраняем в кэш
            if len(self.response_cache) >= self.max_cache_size:
                oldest_key = min(self.response_cache.keys(), 
                               key=lambda k: self.response_cache[k]['timestamp'])
                del self.response_cache[oldest_key]
            
            self.response_cache[cache_key] = {
                'response': response,
                'timestamp': time.time()
            }
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Ошибка генерации ответа: {e}")
            return {
                'response': 'Извините, произошла ошибка при обработке запроса.',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _generate_enhanced_response(self, message: str, context: Optional[dict] = None) -> dict:
        """Генерация улучшенного ответа с векторным поиском"""
        # Сначала пробуем векторный поиск
        if self.qdrant_adapter:
            similar_docs = self.qdrant_adapter.search_similar(
                "rubin_knowledge", 
                message, 
                limit=3, 
                score_threshold=0.6
            )
            
            if similar_docs:
                # Используем лучший результат
                best_match = similar_docs[0]
                logger.info(f"🔍 Найден похожий документ (score: {best_match['score']:.3f})")
                
                return {
                    'response': best_match['text'],
                    'source': 'vector_search',
                    'confidence': best_match['score'],
                    'similar_docs': len(similar_docs),
                    'metadata': best_match.get('metadata', {}),
                    'timestamp': datetime.now().isoformat(),
                    'memory_usage': self._get_memory_stats()
                }
        
        # Fallback: традиционный поиск по ключевым словам
        message_lower = message.lower()
        
        for category, data in self.knowledge_base.items():
            for keyword in data['keywords']:
                if keyword in message_lower:
                    import random
                    response_text = random.choice(data['responses'])
                    
                    return {
                        'response': response_text,
                        'source': 'keyword_search',
                        'category': category,
                        'confidence': 0.8,
                        'timestamp': datetime.now().isoformat(),
                        'memory_usage': self._get_memory_stats()
                    }
        
        # Общий ответ
        return {
            'response': f'Получил ваше сообщение: "{message}". Это интересный вопрос! Могу помочь с программированием, электротехникой или автоматизацией.',
            'source': 'general',
            'confidence': 0.5,
            'timestamp': datetime.now().isoformat(),
            'memory_usage': self._get_memory_stats()
        }
    
    def _get_memory_stats(self) -> dict:
        """Получить статистику памяти"""
        if self.memory_optimizer:
            return self.memory_optimizer.get_memory_usage()
        else:
            return {'status': 'optimizer_unavailable'}
    
    def cleanup(self):
        """Очистка ресурсов"""
        if self.memory_optimizer:
            self.memory_optimizer.stop_monitoring()
        
        self.response_cache.clear()
        collected = gc.collect()
        logger.info(f"🧹 Очистка завершена, собрано объектов: {collected}")

# Инициализация AI
rubin_ai = QdrantEnhancedRubinAI()

@app.route('/')
def index():
    """Главная страница"""
    return jsonify({
        'message': 'Rubin AI v2.0 - Qdrant Enhanced',
        'version': '2.0-qdrant',
        'status': 'running',
        'qdrant_available': QDRANT_ADAPTER_AVAILABLE,
        'memory_optimizer': MEMORY_OPTIMIZER_AVAILABLE,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    """API для чата с векторным поиском"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        context = data.get('context', {})
        
        if not message:
            return jsonify({'error': 'Сообщение не может быть пустым'}), 400
        
        response = rubin_ai.get_response(message, context)
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"❌ Ошибка в API чата: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/search', methods=['POST'])
def vector_search():
    """API для векторного поиска"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        collection = data.get('collection', 'rubin_knowledge')
        limit = data.get('limit', 10)
        threshold = data.get('threshold', 0.6)
        
        if not query:
            return jsonify({'error': 'Запрос не может быть пустым'}), 400
        
        if not rubin_ai.qdrant_adapter:
            return jsonify({'error': 'Qdrant адаптер недоступен'}), 400
        
        results = rubin_ai.qdrant_adapter.search_similar(
            collection, query, limit, threshold
        )
        
        return jsonify({
            'query': query,
            'collection': collection,
            'results': results,
            'count': len(results),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Ошибка векторного поиска: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health():
    """Проверка здоровья системы"""
    memory_stats = rubin_ai._get_memory_stats()
    qdrant_status = rubin_ai.qdrant_adapter.get_status() if rubin_ai.qdrant_adapter else {}
    
    return jsonify({
        'status': 'healthy',
        'system': 'Rubin AI v2.0 Qdrant Enhanced',
        'qdrant_status': qdrant_status,
        'memory_usage': memory_stats,
        'cache_size': len(rubin_ai.response_cache),
        'conversations': len(rubin_ai.conversation_history),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/collections')
def collections():
    """Информация о коллекциях Qdrant"""
    if not rubin_ai.qdrant_adapter:
        return jsonify({'error': 'Qdrant адаптер недоступен'}), 400
    
    collections_info = {}
    collections = ['rubin_knowledge', 'rubin_documents', 'rubin_conversations']
    
    for collection_name in collections:
        info = rubin_ai.qdrant_adapter.get_collection_info(collection_name)
        if info:
            collections_info[collection_name] = info
    
    return jsonify({
        'collections': collections_info,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/RubinIDE.html')
def rubin_ide():
    """Веб-интерфейс Rubin IDE"""
    return send_from_directory('.', 'RubinIDE.html')

@app.route('/RubinDeveloper.html')
def rubin_developer():
    """Developer интерфейс"""
    return send_from_directory('.', 'RubinDeveloper.html')

def main():
    """Главная функция"""
    logger.info("🚀 Запуск Qdrant-улучшенного Rubin AI v2.0 сервера...")
    
    try:
        app.run(
            host='0.0.0.0',
            port=8084,
            debug=False,
            threaded=True,
            use_reloader=False
        )
    except KeyboardInterrupt:
        logger.info("🛑 Получен сигнал остановки...")
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
    finally:
        logger.info("🧹 Очистка ресурсов...")
        rubin_ai.cleanup()
        logger.info("✅ Сервер остановлен")

if __name__ == "__main__":
    main()
