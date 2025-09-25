"""
Оптимизированный API модуля электротехники Rubin AI
Включает кэширование, пул соединений и оптимизацию производительности
"""

import os
import sys
import json
import logging
import time
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
from functools import lru_cache
import hashlib

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Инициализация Flask приложения
app = Flask(__name__)
CORS(app, origins=['http://localhost:8084', 'http://127.0.0.1:8084'])

# Кэш для ответов
response_cache = {}
cache_lock = threading.Lock()
CACHE_TTL = 300  # 5 минут

# Статистика производительности
performance_stats = {
    'total_requests': 0,
    'cache_hits': 0,
    'cache_misses': 0,
    'avg_response_time': 0.0,
    'error_count': 0
}

def get_cache_key(query: str) -> str:
    """Генерация ключа кэша"""
    return hashlib.md5(query.encode()).hexdigest()

def is_cache_valid(timestamp: float) -> bool:
    """Проверка валидности кэша"""
    return time.time() - timestamp < CACHE_TTL

def update_performance_stats(response_time: float, cache_hit: bool, error: bool = False):
    """Обновление статистики производительности"""
    global performance_stats
    
    performance_stats['total_requests'] += 1
    if cache_hit:
        performance_stats['cache_hits'] += 1
    else:
        performance_stats['cache_misses'] += 1
    
    if error:
        performance_stats['error_count'] += 1
    
    # Обновляем среднее время ответа
    total = performance_stats['total_requests']
    current_avg = performance_stats['avg_response_time']
    performance_stats['avg_response_time'] = (current_avg * (total - 1) + response_time) / total

@lru_cache(maxsize=1000)
def get_electrical_response_cached(query: str) -> str:
    """Кэшированный ответ по электротехнике"""
    # Базовые ответы по электротехнике
    electrical_responses = {
        'закон ома': '''
**Закон Ома для участка цепи:**
• U = I × R (напряжение = ток × сопротивление)
• I = U / R (ток = напряжение / сопротивление)  
• R = U / I (сопротивление = напряжение / ток)

**Закон Ома для полной цепи:**
• I = E / (R + r)
• E - ЭДС источника
• R - внешнее сопротивление
• r - внутреннее сопротивление источника

**Применение:**
• Расчет токов в цепях
• Выбор резисторов
• Анализ цепей постоянного тока
        ''',
        'закон кирхгофа': '''
**Первый закон Кирхгофа (закон токов):**
• Сумма токов, входящих в узел = сумме токов, выходящих из узла
• ΣIвх = ΣIвых

**Второй закон Кирхгофа (закон напряжений):**
• Сумма ЭДС в замкнутом контуре = сумме падений напряжений
• ΣE = Σ(I×R)

**Применение:**
• Анализ сложных цепей
• Расчет токов в разветвленных цепях
• Проверка правильности расчетов
        ''',
        'мощность': '''
**Формулы мощности:**
• P = U × I (мощность = напряжение × ток)
• P = I² × R (мощность = ток² × сопротивление)
• P = U² / R (мощность = напряжение² / сопротивление)

**Единицы измерения:**
• Ватт (Вт) - основная единица
• Киловатт (кВт) = 1000 Вт
• Мегаватт (МВт) = 1000000 Вт

**Применение:**
• Расчет потребляемой мощности
• Выбор проводов и защитных устройств
• Энергетические расчеты
        ''',
        'трансформатор': '''
**Принцип работы трансформатора:**
• Основан на явлении электромагнитной индукции
• Переменный ток в первичной обмотке создает переменное магнитное поле
• Магнитное поле индуцирует ЭДС во вторичной обмотке

**Основные параметры:**
• Коэффициент трансформации: K = U1/U2 = N1/N2
• КПД трансформатора: η = P2/P1 × 100%
• Потери: магнитные и электрические

**Применение:**
• Изменение напряжения
• Гальваническая развязка
• Согласование сопротивлений
        ''',
        'защита': '''
**Защита электрических цепей от короткого замыкания:**

**1. Предохранители (Fuses):**
• Плавкие вставки - перегорают при превышении тока
• Быстродействующие предохранители - для защиты полупроводников
• Замедленные предохранители - для защиты двигателей

**2. Автоматические выключатели (Circuit Breakers):**
• Тепловые расцепители - срабатывают при перегрузке
• Электромагнитные расцепители - мгновенное отключение при КЗ
• Электронные расцепители - программируемая защита

**3. Супрессоры и ограничители перенапряжений:**
• Варисторы - защита от импульсных перенапряжений
• Газоразрядные приборы - защита от грозовых разрядов
• TVS-диоды - быстрая защита электронных схем

**Принципы защиты:**
• Селективность - отключение только поврежденного участка
• Быстродействие - минимальное время срабатывания
• Надежность - гарантированное срабатывание при КЗ
        '''
    }
    
    # Поиск наиболее подходящего ответа
    query_lower = query.lower()
    for key, response in electrical_responses.items():
        if key in query_lower:
            return response
    
    # Общий ответ по электротехнике
    return '''
**Основы электротехники:**

**Основные величины:**
• Напряжение (U) - в вольтах (В)
• Ток (I) - в амперах (А)
• Сопротивление (R) - в омах (Ом)
• Мощность (P) - в ваттах (Вт)

**Основные законы:**
• Закон Ома: U = I × R
• Законы Кирхгофа
• Закон Джоуля-Ленца: Q = I²Rt

**Применение:**
• Расчет электрических цепей
• Выбор электрооборудования
• Анализ энергопотребления
    '''

@app.route('/health', methods=['GET'])
def health_check():
    """Проверка здоровья сервиса"""
    return jsonify({
        'status': 'healthy',
        'service': 'electrical_api',
        'timestamp': datetime.now().isoformat(),
        'performance': performance_stats
    })

@app.route('/api/electrical/status', methods=['GET'])
def get_status():
    """Получение статуса модуля"""
    cache_hit_rate = 0.0
    if performance_stats['total_requests'] > 0:
        cache_hit_rate = performance_stats['cache_hits'] / performance_stats['total_requests']
    
    return jsonify({
        'status': 'online',
        'module': 'electrical',
        'timestamp': datetime.now().isoformat(),
        'performance': {
            'total_requests': performance_stats['total_requests'],
            'cache_hit_rate': cache_hit_rate,
            'avg_response_time': performance_stats['avg_response_time'],
            'error_rate': performance_stats['error_count'] / max(1, performance_stats['total_requests'])
        }
    })

@app.route('/api/electrical/explain', methods=['POST'])
def explain_electrical():
    """Объяснение по электротехнике с кэшированием"""
    start_time = time.time()
    
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Не указан запрос'}), 400
        
        query = data['query']
        cache_key = get_cache_key(query)
        cache_hit = False
        
        # Проверяем кэш
        with cache_lock:
            if cache_key in response_cache:
                cached_data = response_cache[cache_key]
                if is_cache_valid(cached_data['timestamp']):
                    cache_hit = True
                    response_time = time.time() - start_time
                    update_performance_stats(response_time, cache_hit)
                    
                    return jsonify({
                        'response': cached_data['response'],
                        'cached': True,
                        'response_time': response_time,
                        'timestamp': datetime.now().isoformat()
                    })
        
        # Генерируем новый ответ
        response = get_electrical_response_cached(query)
        
        # Сохраняем в кэш
        with cache_lock:
            response_cache[cache_key] = {
                'response': response,
                'timestamp': time.time()
            }
        
        response_time = time.time() - start_time
        update_performance_stats(response_time, cache_hit)
        
        return jsonify({
            'response': response,
            'cached': False,
            'response_time': response_time,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        response_time = time.time() - start_time
        update_performance_stats(response_time, False, error=True)
        logger.error(f"Ошибка в explain_electrical: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/electrical/clear-cache', methods=['POST'])
def clear_cache():
    """Очистка кэша"""
    try:
        with cache_lock:
            response_cache.clear()
        
        # Очищаем кэш функции
        get_electrical_response_cached.cache_clear()
        
        return jsonify({
            'success': True,
            'message': 'Кэш очищен',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Ошибка очистки кэша: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/electrical/performance', methods=['GET'])
def get_performance():
    """Получение статистики производительности"""
    cache_hit_rate = 0.0
    error_rate = 0.0
    
    if performance_stats['total_requests'] > 0:
        cache_hit_rate = performance_stats['cache_hits'] / performance_stats['total_requests']
        error_rate = performance_stats['error_count'] / performance_stats['total_requests']
    
    return jsonify({
        'performance_stats': performance_stats,
        'cache_hit_rate': cache_hit_rate,
        'error_rate': error_rate,
        'cache_size': len(response_cache),
        'timestamp': datetime.now().isoformat()
    })

def cleanup_old_cache():
    """Очистка устаревшего кэша"""
    current_time = time.time()
    with cache_lock:
        keys_to_remove = []
        for key, data in response_cache.items():
            if not is_cache_valid(data['timestamp']):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del response_cache[key]
    
    if keys_to_remove:
        logger.info(f"Очищено {len(keys_to_remove)} устаревших записей кэша")

def start_cache_cleanup():
    """Запуск периодической очистки кэша"""
    def cleanup_loop():
        while True:
            try:
                time.sleep(300)  # 5 минут
                cleanup_old_cache()
            except Exception as e:
                logger.error(f"Ошибка очистки кэша: {e}")
    
    cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
    cleanup_thread.start()
    logger.info("Запущена периодическая очистка кэша")

if __name__ == '__main__':
    logger.info("Запуск оптимизированного сервера электротехники на порту 8087...")
    
    # Запускаем очистку кэша
    start_cache_cleanup()
    
    # Запускаем сервер
    app.run(host='0.0.0.0', port=8087, debug=True)

















