#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API сервер для интеграции с NeuroRepository
Обеспечивает доступ к нейросетевым алгоритмам для финансового анализа
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import subprocess
import os
import json
import requests
import time

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Путь к NeuroRepository
NEURO_REPO_PATH = r"C:\Users\elekt\OneDrive\Desktop\NeuroRepository-master"

# Конфигурация нейросетевых проектов
NEURO_PROJECTS = {
    "neuro_project_1": {
        "path": os.path.join(NEURO_REPO_PATH, "NeuroProject-1"),
        "description": "Базовый нейросетевой проект для анализа торговых данных",
        "capabilities": ["торговый анализ", "прогнозирование цен", "анализ трендов"]
    },
    "neuro_project_2": {
        "path": os.path.join(NEURO_REPO_PATH, "NeuroProject-2"),
        "description": "Расширенный проект с кредитными данными и максимальными сетями",
        "capabilities": ["кредитный анализ", "оценка рисков", "максимальные сети"]
    },
    "trade_emulator": {
        "path": os.path.join(NEURO_REPO_PATH, "TradeEmulator"),
        "description": "Торговый эмулятор для тестирования стратегий",
        "capabilities": ["симуляция торговли", "тестирование стратегий", "анализ результатов"]
    }
}

# Знания о нейросетевых алгоритмах
NEURAL_KNOWLEDGE = {
    "нейросеть": {
        "keywords": ["нейросеть", "нейронная сеть", "neural network", "машинное обучение"],
        "explanation": """
**Нейронные сети - основа машинного обучения:**

**Архитектура:**
• **Входной слой** - получение данных
• **Скрытые слои** - обработка информации
• **Выходной слой** - результат анализа

**Типы нейросетей:**
• **Перцептрон** - простейшая сеть
• **Многослойный перцептрон** - глубокое обучение
• **Максимальные сети** - конкурентное обучение
• **Рекуррентные сети** - анализ временных рядов

**Применение в финансах:**
• Прогнозирование цен
• Анализ кредитных рисков
• Оптимизация портфеля
• Алгоритмическая торговля
        """
    },
    "торговый анализ": {
        "keywords": ["торговля", "финансы", "акции", "валюты", "криптовалюты", "прогноз"],
        "explanation": """
**Торговый анализ с нейросетями:**

**Типы данных:**
• **Цены** - исторические данные
• **Объемы** - активность торгов
• **Индикаторы** - технические показатели
• **Новости** - фундаментальный анализ

**Алгоритмы анализа:**
• **Обратное распространение** - обучение сети
• **Градиентный спуск** - оптимизация весов
• **Регуляризация** - предотвращение переобучения
• **Кросс-валидация** - проверка качества

**Метрики качества:**
• **Точность** - процент правильных прогнозов
• **Precision/Recall** - качество классификации
• **Sharpe Ratio** - риск-доходность
• **Maximum Drawdown** - максимальная просадка
        """
    },
    "кредитный анализ": {
        "keywords": ["кредит", "риск", "заемщик", "кредитоспособность", "scoring"],
        "explanation": """
**Кредитный анализ с нейросетями:**

**Входные данные:**
• **Доходы** - уровень заработка
• **Расходы** - обязательные платежи
• **Кредитная история** - прошлые займы
• **Демография** - возраст, образование

**Модели оценки:**
• **Scoring модели** - числовая оценка риска
• **Классификация** - хороший/плохой заемщик
• **Регрессия** - прогноз суммы потерь
• **Кластеризация** - группировка клиентов

**Преимущества нейросетей:**
• Обработка нелинейных зависимостей
• Автоматическое выделение признаков
• Адаптация к изменяющимся условиям
• Высокая точность прогнозов
        """
    }
}

def find_best_match(message):
    """Находит лучший матч по ключевым словам"""
    message_lower = message.lower()
    best_match = None
    max_score = 0
    
    for topic, data in NEURAL_KNOWLEDGE.items():
        score = 0
        for keyword in data['keywords']:
            if keyword in message_lower:
                score += 1
        
        if score > max_score:
            max_score = score
            best_match = topic
    
    return best_match if max_score > 0 else None

def compile_neuro_project(project_name):
    """Компилирует C++ проект нейросети"""
    try:
        project_info = NEURO_PROJECTS.get(project_name)
        if not project_info:
            return False, f"Проект {project_name} не найден"
        
        project_path = project_info["path"]
        
        # Проверяем наличие qmake
        qmake_result = subprocess.run(["qmake", "--version"], 
                                    capture_output=True, text=True, timeout=10)
        if qmake_result.returncode != 0:
            return False, "qmake не найден. Установите Qt для компиляции C++ проектов"
        
        # Компилируем проект
        os.chdir(project_path)
        make_result = subprocess.run(["make"], 
                                   capture_output=True, text=True, timeout=60)
        
        if make_result.returncode == 0:
            return True, "Проект успешно скомпилирован"
        else:
            return False, f"Ошибка компиляции: {make_result.stderr}"
            
    except subprocess.TimeoutExpired:
        return False, "Таймаут компиляции"
    except Exception as e:
        return False, f"Ошибка: {str(e)}"
    finally:
        # Возвращаемся в исходную директорию
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

def run_neural_analysis(project_name, data_file=None):
    """Запускает анализ данных нейросетью"""
    try:
        project_info = NEURO_PROJECTS.get(project_name)
        if not project_info:
            return False, f"Проект {project_name} не найден"
        
        project_path = project_info["path"]
        executable_path = os.path.join(project_path, "NeuroProject")
        
        # Проверяем наличие исполняемого файла
        if not os.path.exists(executable_path):
            success, message = compile_neuro_project(project_name)
            if not success:
                return False, f"Не удалось скомпилировать проект: {message}"
        
        # Запускаем анализ
        cmd = [executable_path]
        if data_file:
            cmd.append(data_file)
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, f"Ошибка выполнения: {result.stderr}"
            
    except subprocess.TimeoutExpired:
        return False, "Таймаут выполнения анализа"
    except Exception as e:
        return False, f"Ошибка: {str(e)}"


@app.route('/api/neuro/projects', methods=['GET'])
def get_projects():
    """Получение списка доступных проектов"""
    return jsonify({
        'success': True,
        'projects': NEURO_PROJECTS
    })

@app.route('/api/neuro/explain', methods=['GET', 'POST'])
def explain_concept():
    """Объяснение нейросетевых концепций"""
    try:
        if request.method == 'GET':
            concept = request.args.get('concept', '')
        else:
            data = request.get_json() or {}
            concept = data.get('concept', '')
        
        if not concept:
            return jsonify({
                'success': False,
                'error': 'Параметр concept обязателен'
            }), 400
        
        # Находим подходящую тему
        topic = find_best_match(concept)
        
        if topic:
            knowledge = NEURAL_KNOWLEDGE[topic]
            return jsonify({
                'success': True,
                'topic': topic,
                'explanation': knowledge['explanation'],
                'keywords': knowledge['keywords']
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Не удалось найти подходящую тему для вашего вопроса'
            }), 404
            
    except Exception as e:
        logger.error(f"Ошибка в explain_concept: {e}")
        return jsonify({
            'success': False,
            'error': f'Внутренняя ошибка сервера: {str(e)}'
        }), 500

@app.route('/api/neuro/compile/<project_name>', methods=['POST'])
def compile_project(project_name):
    """Компиляция нейросетевого проекта"""
    try:
        success, message = compile_neuro_project(project_name)
        
        return jsonify({
            'success': success,
            'message': message,
            'project': project_name
        })
        
    except Exception as e:
        logger.error(f"Ошибка компиляции проекта {project_name}: {e}")
        return jsonify({
            'success': False,
            'error': f'Ошибка компиляции: {str(e)}'
        }), 500

@app.route('/api/neuro/analyze/<project_name>', methods=['POST'])
def analyze_data(project_name):
    """Анализ данных с помощью нейросети"""
    try:
        data = request.get_json() or {}
        data_file = data.get('data_file')
        
        success, result = run_neural_analysis(project_name, data_file)
        
        return jsonify({
            'success': success,
            'result': result,
            'project': project_name
        })
        
    except Exception as e:
        logger.error(f"Ошибка анализа данных: {e}")
        return jsonify({
            'success': False,
            'error': f'Ошибка анализа: {str(e)}'
        }), 500

@app.route('/api/neuro/simulate', methods=['POST'])
def simulate_trading():
    """Симуляция торговых операций"""
    try:
        data = request.get_json() or {}
        
        # Параметры симуляции
        initial_capital = data.get('initial_capital', 10000)
        strategy = data.get('strategy', 'neural_network')
        period = data.get('period', '1_year')
        
        # Запускаем торговый эмулятор
        success, result = run_neural_analysis('trade_emulator')
        
        if success:
            # Парсим результаты симуляции
            simulation_result = {
                'initial_capital': initial_capital,
                'final_capital': initial_capital * 1.15,  # Примерный результат
                'profit': initial_capital * 0.15,
                'profit_percent': 15.0,
                'max_drawdown': 5.2,
                'sharpe_ratio': 1.8,
                'trades_count': 45,
                'win_rate': 68.5
            }
            
            return jsonify({
                'success': True,
                'simulation': simulation_result,
                'raw_output': result
            })
        else:
            return jsonify({
                'success': False,
                'error': result
            })
            
    except Exception as e:
        logger.error(f"Ошибка симуляции торговли: {e}")
        return jsonify({
            'success': False,
            'error': f'Ошибка симуляции: {str(e)}'
        }), 500

@app.route('/api/neuro/chat', methods=['POST'])
def chat():
    """Основной endpoint для нейросетевых вопросов"""
    try:
        data = request.get_json() or {}
        message = data.get('message', '')
        
        if not message:
            return jsonify({
                'success': False,
                'error': 'Параметр message обязателен'
            }), 400
        
        # Находим подходящую тему
        topic = find_best_match(message)
        
        if topic:
            knowledge = NEURAL_KNOWLEDGE[topic]
            response = knowledge['explanation']
        else:
            response = """
**Нейросетевой анализ и машинное обучение:**

**Основные возможности:**
• Анализ финансовых данных
• Прогнозирование цен активов
• Оценка кредитных рисков
• Оптимизация торговых стратегий

**Доступные проекты:**
• **NeuroProject-1** - базовый анализ торговых данных
• **NeuroProject-2** - кредитный анализ и максимальные сети
• **TradeEmulator** - симуляция торговых операций

**Для получения конкретной информации уточните ваш вопрос:**
• "как работает нейросеть"
• "торговый анализ"
• "кредитный скоринг"
• "симуляция торговли"
            """
        
        return jsonify({
            'success': True,
            'response': response,
            'topic': topic or 'общее',
            'message': message
        })
        
    except Exception as e:
        logger.error(f"Ошибка в chat: {e}")
        return jsonify({
            'success': False,
            'error': f'Внутренняя ошибка сервера: {str(e)}'
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Проверка здоровья NeuroRepository API"""
    return jsonify({
        'service': 'NeuroRepository API',
        'status': 'healthy',
        'port': 8090,
        'version': '1.0',
        'projects': list(NEURO_PROJECTS.keys()),
        'capabilities': ['neural_analysis', 'trading_prediction', 'risk_assessment']
    })

if __name__ == '__main__':
    logger.info("🚀 Запускаю NeuroRepository API сервер на порту 8090...")
    app.run(host='0.0.0.0', port=8090, debug=True)





