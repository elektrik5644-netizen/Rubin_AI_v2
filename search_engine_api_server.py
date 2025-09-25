#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Search Engine API Server
Гибридный поиск и индексация
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import logging
from datetime import datetime
import re
import os

# Попытка импорта библиотек для поиска
try:
    # from hybrid_search import hybrid_search  # Временно отключено из-за проблем с памятью
    from sequential_search_engine import SequentialSearchEngine
    SEARCH_ENGINES_AVAILABLE = True
except ImportError:
    SEARCH_ENGINES_AVAILABLE = False

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Простая база знаний для демонстрации
KNOWLEDGE_BASE = {
    "электротехника": {
        "закон ома": "U = I × R, где U - напряжение, I - ток, R - сопротивление",
        "закон кирхгофа": "Сумма токов в узле равна нулю, сумма напряжений в контуре равна нулю",
        "мощность": "P = U × I = I² × R = U² / R",
        "резистор": "Пассивный элемент, ограничивающий ток в цепи",
        "конденсатор": "Накопитель электрической энергии в электрическом поле"
    },
    "программирование": {
        "python": "Высокоуровневый язык программирования с простым синтаксисом",
        "алгоритм": "Последовательность инструкций для решения задачи",
        "функция": "Блок кода, который можно вызывать многократно",
        "цикл": "Конструкция для повторения блока кода",
        "условие": "Конструкция для выполнения кода при определенных условиях"
    },
    "математика": {
        "квадратное уравнение": "ax² + bx + c = 0, решается через дискриминант",
        "интеграл": "Предел суммы площадей под кривой",
        "производная": "Скорость изменения функции",
        "система уравнений": "Несколько уравнений с несколькими неизвестными",
        "тригонометрия": "Раздел математики о треугольниках и углах"
    },
    "plc": {
        "программирование plc": "Создание программ для программируемых логических контроллеров",
        "ladder logic": "Графический язык программирования PLC",
        "функциональные блоки": "Модульные элементы программы PLC",
        "таймеры": "Элементы для измерения времени в PLC",
        "счетчики": "Элементы для подсчета событий в PLC"
    }
}

def simple_text_search(query, knowledge_base):
    """Простой текстовый поиск"""
    query_lower = query.lower()
    results = []
    
    for category, items in knowledge_base.items():
        for key, value in items.items():
            score = 0
            query_words = query_lower.split()
            key_words = key.lower().split()
            value_words = value.lower().split()
            
            # Подсчет совпадений
            for q_word in query_words:
                if q_word in key_words:
                    score += 2  # Больший вес для совпадений в ключе
                if q_word in value_words:
                    score += 1  # Меньший вес для совпадений в значении
            
            if score > 0:
                results.append({
                    "category": category,
                    "key": key,
                    "value": value,
                    "score": score,
                    "relevance": "high" if score >= 3 else "medium" if score >= 2 else "low"
                })
    
    # Сортируем по релевантности
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:10]  # Возвращаем топ-10 результатов

def semantic_search(query, knowledge_base):
    """Семантический поиск (упрощенная версия)"""
    query_lower = query.lower()
    semantic_mappings = {
        "напряжение": ["вольт", "вольты", "u", "voltage"],
        "ток": ["ампер", "амперы", "i", "current"],
        "сопротивление": ["ом", "омы", "r", "resistance"],
        "мощность": ["ватт", "ватты", "p", "power"],
        "программа": ["код", "скрипт", "алгоритм", "code"],
        "уравнение": ["формула", "выражение", "equation"],
        "контроллер": ["plc", "автоматизация", "логика", "controller"]
    }
    
    # Расширяем запрос синонимами
    expanded_query = query_lower
    for main_term, synonyms in semantic_mappings.items():
        if main_term in query_lower:
            expanded_query += " " + " ".join(synonyms)
        for synonym in synonyms:
            if synonym in query_lower:
                expanded_query += " " + main_term
    
    # Выполняем поиск с расширенным запросом
    return simple_text_search(expanded_query, knowledge_base)

def hybrid_search_implementation(query, knowledge_base):
    """Гибридный поиск (комбинация текстового и семантического)"""
    # Текстовый поиск
    text_results = simple_text_search(query, knowledge_base)
    
    # Семантический поиск
    semantic_results = semantic_search(query, knowledge_base)
    
    # Объединяем результаты
    all_results = {}
    for result in text_results + semantic_results:
        key = f"{result['category']}:{result['key']}"
        if key not in all_results:
            all_results[key] = result
        else:
            # Увеличиваем score для дублирующихся результатов
            all_results[key]["score"] += result["score"]
    
    # Сортируем по обновленному score
    final_results = list(all_results.values())
    final_results.sort(key=lambda x: x["score"], reverse=True)
    
    return final_results[:15]  # Возвращаем топ-15 результатов

def index_documents(documents):
    """Индексация документов"""
    indexed = {}
    
    for doc_id, content in documents.items():
        if isinstance(content, str):
            words = re.findall(r'\b\w+\b', content.lower())
            indexed[doc_id] = {
                "content": content,
                "words": words,
                "word_count": len(words),
                "unique_words": len(set(words))
            }
    
    return indexed

@app.route('/api/search/hybrid', methods=['POST'])
def hybrid_search_endpoint():
    """Гибридный поиск"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        search_type = data.get('type', 'hybrid')
        
        if not query:
            return jsonify({
                "success": False,
                "error": "Запрос не может быть пустым"
            }), 400
        
        results = []
        
        if search_type == 'text':
            results = simple_text_search(query, KNOWLEDGE_BASE)
        elif search_type == 'semantic':
            results = semantic_search(query, KNOWLEDGE_BASE)
        elif search_type == 'hybrid':
            results = hybrid_search_implementation(query, KNOWLEDGE_BASE)
        else:
            return jsonify({
                "success": False,
                "error": f"Неизвестный тип поиска: {search_type}"
            }), 400
        
        return jsonify({
            "success": True,
            "query": query,
            "search_type": search_type,
            "results_count": len(results),
            "results": results,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Ошибка поиска: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка поиска: {str(e)}"
        }), 500

@app.route('/api/search/index', methods=['POST'])
def index_documents_endpoint():
    """Индексация документов"""
    try:
        data = request.get_json()
        documents = data.get('documents', {})
        
        if not documents:
            return jsonify({
                "success": False,
                "error": "Документы не могут быть пустыми"
            }), 400
        
        indexed = index_documents(documents)
        
        return jsonify({
            "success": True,
            "documents_count": len(documents),
            "indexed_count": len(indexed),
            "indexed_documents": indexed,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Ошибка индексации: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка индексации: {str(e)}"
        }), 500

@app.route('/api/search/knowledge', methods=['POST'])
def search_knowledge():
    """Поиск в базе знаний"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        category = data.get('category', 'all')
        
        if not query:
            return jsonify({
                "success": False,
                "error": "Запрос не может быть пустым"
            }), 400
        
        # Фильтруем по категории если указана
        if category != 'all' and category in KNOWLEDGE_BASE:
            filtered_kb = {category: KNOWLEDGE_BASE[category]}
        else:
            filtered_kb = KNOWLEDGE_BASE
        
        results = hybrid_search_implementation(query, filtered_kb)
        
        return jsonify({
            "success": True,
            "query": query,
            "category": category,
            "results_count": len(results),
            "results": results,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Ошибка поиска в базе знаний: {e}")
        return jsonify({
            "success": False,
            "error": f"Ошибка поиска: {str(e)}"
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Проверка здоровья Search Engine API"""
    return jsonify({
        'service': 'Search Engine API',
        'status': 'healthy',
        'port': 8102,
        'version': '1.0',
        'capabilities': ['semantic_search', 'document_search', 'knowledge_base']
    })

@app.route('/api/search/health', methods=['GET'])
def health_check():
    """Проверка здоровья сервера"""
    return jsonify({
        "service": "Search Engine API",
        "status": "online",
        "version": "1.0",
        "search_engines_available": SEARCH_ENGINES_AVAILABLE,
        "knowledge_base_size": sum(len(items) for items in KNOWLEDGE_BASE.values()),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/search/status', methods=['GET'])
def get_status():
    """Статус сервера"""
    return jsonify({
        "service": "Search Engine API",
        "status": "running",
        "port": 8102,
        "endpoints": [
            "/api/search/hybrid",
            "/api/search/index",
            "/api/search/knowledge",
            "/api/search/health",
            "/api/search/status"
        ],
        "capabilities": [
            "Гибридный поиск",
            "Текстовый поиск",
            "Семантический поиск",
            "Индексация документов",
            "Поиск в базе знаний"
        ],
        "knowledge_categories": list(KNOWLEDGE_BASE.keys()),
        "dependencies": {
            "hybrid_search": SEARCH_ENGINES_AVAILABLE,
            "sequential_search": SEARCH_ENGINES_AVAILABLE
        }
    })

if __name__ == '__main__':
    print("🔍 Search Engine API Server запущен")
    print("URL: http://localhost:8102")
    print("Endpoints:")
    print("  - POST /api/search/hybrid - Гибридный поиск")
    print("  - POST /api/search/index - Индексация документов")
    print("  - POST /api/search/knowledge - Поиск в базе знаний")
    print("  - GET /api/search/health - Проверка здоровья")
    print("  - GET /api/search/status - Статус сервера")
    app.run(host='0.0.0.0', port=8102, debug=True)


