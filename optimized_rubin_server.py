#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Оптимизированный Rubin AI v2.0 сервер
С улучшенным управлением памятью и производительностью
"""

import os
import sys
import json
import logging
import gc
from datetime import datetime
from typing import Optional
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sqlite3
import threading
import time

# Добавляем пути к модулям
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'providers'))

# Импорт оптимизатора памяти
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
        logging.FileHandler('rubin_optimized.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("rubin_ai_optimized")

class OptimizedRubinAI:
    """Оптимизированная версия Rubin AI"""
    
    def __init__(self):
        self.knowledge_base = {}
        self.conversation_history = []
        self.response_cache = {}
        self.cache_ttl = 300  # 5 минут
        self.max_cache_size = 1000
        
        # Инициализация оптимизатора памяти
        if MEMORY_OPTIMIZER_AVAILABLE:
            self.memory_optimizer = MemoryOptimizer(threshold_mb=300, cleanup_interval=180)
            self.memory_optimizer.start_monitoring()
            logger.info("✅ Оптимизатор памяти активирован")
        else:
            self.memory_optimizer = None
            logger.warning("⚠️ Оптимизатор памяти недоступен")
        
        self.load_knowledge_base()
        logger.info("🚀 Оптимизированный Rubin AI инициализирован")
    
    def load_knowledge_base(self):
        """Загрузка базы знаний с оптимизацией"""
        try:
            # Проверяем существование файлов базы данных
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
                    
                    # Получаем таблицы
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = cursor.fetchall()
                    
                    for table in tables:
                        table_name = table[0]
                        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                        count = cursor.fetchone()[0]
                        logger.info(f"  📋 Таблица {table_name}: {count} записей")
                    
                    conn.close()
                    break
            else:
                logger.info("📚 База знаний не найдена, создаем новую")
                self.create_default_knowledge_base()
                
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки базы знаний: {e}")
            self.create_default_knowledge_base()
    
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
    
    def get_response(self, message: str, context: Optional[dict] = None) -> dict:
        """Получить ответ с оптимизацией"""
        try:
            # Проверяем кэш
            cache_key = hash(message.lower().strip())
            if cache_key in self.response_cache:
                cache_entry = self.response_cache[cache_key]
                if time.time() - cache_entry['timestamp'] < self.cache_ttl:
                    logger.info("📋 Ответ из кэша")
                    return cache_entry['response']
                else:
                    # Удаляем устаревший кэш
                    del self.response_cache[cache_key]
            
            # Генерируем новый ответ
            response = self._generate_response(message, context)
            
            # Сохраняем в кэш (с ограничением размера)
            if len(self.response_cache) >= self.max_cache_size:
                # Удаляем самые старые записи
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
    
    def _generate_response(self, message: str, context: Optional[dict] = None) -> dict:
        """Генерация ответа"""
        message_lower = message.lower()
        
        # Поиск по ключевым словам
        for category, data in self.knowledge_base.items():
            for keyword in data['keywords']:
                if keyword in message_lower:
                    import random
                    response_text = random.choice(data['responses'])
                    
                    return {
                        'response': response_text,
                        'category': category,
                        'confidence': 0.8,
                        'timestamp': datetime.now().isoformat(),
                        'memory_usage': self._get_memory_stats()
                    }
        
        # Общий ответ
        return {
            'response': f'Получил ваше сообщение: "{message}". Это интересный вопрос! Могу помочь с программированием, электротехникой или автоматизацией.',
            'category': 'general',
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
        
        # Очистка кэша
        self.response_cache.clear()
        
        # Принудительная сборка мусора
        collected = gc.collect()
        logger.info(f"🧹 Очистка завершена, собрано объектов: {collected}")

# Инициализация AI
rubin_ai = OptimizedRubinAI()

@app.route('/')
def index():
    """Главная страница"""
    return jsonify({
        'message': 'Rubin AI v2.0 - Оптимизированная версия',
        'version': '2.0-optimized',
        'status': 'running',
        'memory_optimizer': MEMORY_OPTIMIZER_AVAILABLE,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    """API для чата"""
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

@app.route('/api/health')
def health():
    """Проверка здоровья системы"""
    memory_stats = rubin_ai._get_memory_stats()
    
    return jsonify({
        'status': 'healthy',
        'system': 'Rubin AI v2.0 Optimized',
        'memory_optimizer': MEMORY_OPTIMIZER_AVAILABLE,
        'memory_usage': memory_stats,
        'cache_size': len(rubin_ai.response_cache),
        'conversations': len(rubin_ai.conversation_history),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/memory/cleanup', methods=['POST'])
def memory_cleanup():
    """Принудительная очистка памяти"""
    try:
        if rubin_ai.memory_optimizer:
            result = rubin_ai.memory_optimizer.cleanup_memory()
            return jsonify({
                'status': 'success',
                'freed_mb': result['freed_mb'],
                'collected_objects': result['collected_objects'],
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({'error': 'Оптимизатор памяти недоступен'}), 400
            
    except Exception as e:
        logger.error(f"❌ Ошибка очистки памяти: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def stats():
    """Статистика системы"""
    if rubin_ai.memory_optimizer:
        system_stats = rubin_ai.memory_optimizer.get_system_stats()
    else:
        system_stats = {'status': 'optimizer_unavailable'}
    
    return jsonify({
        'system_stats': system_stats,
        'cache_size': len(rubin_ai.response_cache),
        'knowledge_categories': len(rubin_ai.knowledge_base),
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
    logger.info("🚀 Запуск оптимизированного Rubin AI v2.0 сервера...")
    
    try:
        # Настройка для продакшена
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
