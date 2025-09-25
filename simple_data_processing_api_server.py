#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 SIMPLE DATA PROCESSING API SERVER
Упрощенная версия без внешних библиотек обработки данных
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import logging
import os
import json
import csv
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Упрощенная база знаний по обработке данных
DATA_PROCESSING_KNOWLEDGE = {
    "анализ": {
        "explanation": "📈 **Анализ данных** - процесс исследования данных для выявления закономерностей.\n\n**Типы анализа:**\n- **Описательный** - что произошло?\n- **Диагностический** - почему произошло?\n- **Предиктивный** - что произойдет?\n- **Предписывающий** - что делать?\n\n**Методы анализа:**\n- **Статистический анализ** - средние, медианы, стандартные отклонения\n- **Корреляционный анализ** - связи между переменными\n- **Регрессионный анализ** - прогнозирование\n- **Кластерный анализ** - группировка данных\n\n**Инструменты:**\n- Excel, Google Sheets\n- Python (pandas, numpy)\n- R (статистика)\n- Tableau, Power BI\n- SQL (базы данных)\n\n**Этапы:**\n1. Сбор данных\n2. Очистка данных\n3. Исследовательский анализ\n4. Моделирование\n5. Интерпретация результатов",
        "keywords": ["анализ", "исследование", "закономерности", "статистика", "корреляция"]
    },
    "очистка": {
        "explanation": "🧹 **Очистка данных** - процесс подготовки данных к анализу.\n\n**Проблемы в данных:**\n- **Пропущенные значения** - отсутствующие данные\n- **Дубликаты** - повторяющиеся записи\n- **Выбросы** - аномальные значения\n- **Некорректные форматы** - неправильные типы данных\n- **Опечатки** - ошибки в тексте\n\n**Методы очистки:**\n- **Удаление** - исключение проблемных записей\n- **Замена** - заполнение пропусков\n- **Нормализация** - приведение к единому формату\n- **Валидация** - проверка корректности\n- **Стандартизация** - унификация значений\n\n**Техники:**\n- **Медиана/среднее** - для числовых пропусков\n- **Мода** - для категориальных пропусков\n- **Интерполяция** - для временных рядов\n- **Регулярные выражения** - для текста\n- **Статистические тесты** - для выбросов\n\n**Инструменты:**\n- Python (pandas, numpy)\n- OpenRefine\n- Excel функции\n- SQL запросы",
        "keywords": ["очистка", "пропуски", "дубликаты", "выбросы", "нормализация"]
    },
    "визуализация": {
        "explanation": "📊 **Визуализация данных** - представление данных в графическом виде.\n\n**Типы графиков:**\n- **Линейные** - для временных рядов\n- **Столбчатые** - для сравнения категорий\n- **Круговые** - для долей\n- **Точечные** - для корреляций\n- **Гистограммы** - для распределений\n- **Box plots** - для статистики\n- **Heat maps** - для матриц\n\n**Принципы визуализации:**\n- **Простота** - избегайте избыточности\n- **Ясность** - понятные подписи\n- **Точность** - правильные масштабы\n- **Красота** - привлекательный дизайн\n- **Функциональность** - интерактивность\n\n**Цвета:**\n- Используйте цветовую слепоту\n- Ограничьте палитру\n- Избегайте красного/зеленого\n- Используйте градации серого\n\n**Инструменты:**\n- **Python:** matplotlib, seaborn, plotly\n- **R:** ggplot2, lattice\n- **JavaScript:** D3.js, Chart.js\n- **BI:** Tableau, Power BI, QlikView\n- **Онлайн:** Google Charts, Highcharts",
        "keywords": ["визуализация", "графики", "диаграммы", "чарты", "отображение"]
    },
    "машинное обучение": {
        "explanation": "🤖 **Машинное обучение** - алгоритмы, которые учатся на данных.\n\n**Типы ML:**\n- **Обучение с учителем** - есть правильные ответы\n- **Обучение без учителя** - поиск закономерностей\n- **Обучение с подкреплением** - обучение через взаимодействие\n\n**Алгоритмы:**\n- **Классификация:** логистическая регрессия, SVM, случайный лес\n- **Регрессия:** линейная регрессия, полиномиальная\n- **Кластеризация:** k-means, иерархическая\n- **Ансамбли:** случайный лес, градиентный бустинг\n\n**Этапы ML:**\n1. **Подготовка данных** - сбор, очистка, разделение\n2. **Выбор модели** - подходящий алгоритм\n3. **Обучение** - тренировка на данных\n4. **Валидация** - проверка качества\n5. **Тестирование** - финальная оценка\n6. **Развертывание** - внедрение в производство\n\n**Метрики качества:**\n- **Точность** - доля правильных предсказаний\n- **Precision/Recall** - для классификации\n- **RMSE/MAE** - для регрессии\n- **F1-score** - гармоническое среднее\n\n**Инструменты:**\n- **Python:** scikit-learn, pandas, numpy\n- **R:** caret, randomForest\n- **Платформы:** TensorFlow, PyTorch, Azure ML",
        "keywords": ["машинное обучение", "алгоритмы", "модели", "предсказание", "классификация"]
    },
    "базы данных": {
        "explanation": "🗄️ **Базы данных** - организованное хранение данных.\n\n**Типы БД:**\n- **Реляционные** - таблицы с связями (MySQL, PostgreSQL)\n- **NoSQL** - документы, ключ-значение (MongoDB, Redis)\n- **Временные ряды** - для временных данных (InfluxDB)\n- **Графовые** - для связей (Neo4j)\n\n**SQL операции:**\n- **SELECT** - выборка данных\n- **INSERT** - добавление записей\n- **UPDATE** - изменение данных\n- **DELETE** - удаление записей\n- **JOIN** - соединение таблиц\n- **GROUP BY** - группировка\n- **ORDER BY** - сортировка\n\n**Нормализация:**\n- **1NF** - атомарные значения\n- **2NF** - зависимость от ключа\n- **3NF** - транзитивная зависимость\n\n**Индексы:**\n- Ускоряют поиск\n- Замедляют вставку\n- Типы: B-tree, Hash, Bitmap\n\n**Оптимизация:**\n- Правильные индексы\n- Эффективные запросы\n- Партиционирование\n- Кэширование\n\n**Инструменты:**\n- **MySQL, PostgreSQL** - реляционные\n- **MongoDB** - документные\n- **Redis** - ключ-значение\n- **Elasticsearch** - поиск",
        "keywords": ["базы данных", "sql", "таблицы", "запросы", "индексы"]
    },
    "big data": {
        "explanation": "🌐 **Big Data** - обработка больших объемов данных.\n\n**Характеристики (3V):**\n- **Volume** - объем данных\n- **Velocity** - скорость поступления\n- **Variety** - разнообразие форматов\n\n**Технологии:**\n- **Hadoop** - распределенная файловая система\n- **Spark** - быстрая обработка в памяти\n- **Kafka** - потоковая обработка\n- **Elasticsearch** - поиск и аналитика\n- **Cassandra** - NoSQL для больших данных\n\n**Архитектуры:**\n- **Lambda** - пакетная + потоковая обработка\n- **Kappa** - только потоковая\n- **Microservices** - сервисная архитектура\n\n**Облачные платформы:**\n- **AWS:** EMR, Redshift, Kinesis\n- **Google Cloud:** BigQuery, Dataflow\n- **Azure:** HDInsight, Data Factory\n\n**Инструменты:**\n- **Python:** pandas, dask, pyspark\n- **R:** sparklyr, bigrquery\n- **SQL:** Hive, Impala, Presto\n- **Визуализация:** Tableau, Power BI\n\n**Вызовы:**\n- Масштабируемость\n- Производительность\n- Безопасность\n- Стоимость хранения",
        "keywords": ["big data", "большие данные", "hadoop", "spark", "потоковая обработка"]
    }
}

@app.route('/api/data_processing/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "message": "Simple Data Processing API работает",
        "module": "Data Processing API",
        "version": "2.0",
        "capabilities": ["Анализ данных", "Очистка данных", "Визуализация", "Машинное обучение", "Базы данных", "Big Data"]
    }), 200

@app.route('/api/data_processing/status', methods=['GET'])
def status():
    return jsonify({
        "status": "healthy",
        "message": "Simple Data Processing API работает",
        "timestamp": os.getenv('FLASK_RUN_FROM_START_TIME')
    }), 200

@app.route('/api/data_processing/process', methods=['POST'])
def process():
    """Обработка данных"""
    data = request.get_json()
    request_data = data.get('data', '').lower()
    
    # Поиск по ключевым словам
    for key, info in DATA_PROCESSING_KNOWLEDGE.items():
        for keyword in info['keywords']:
            if keyword in request_data:
                return jsonify({
                    "processed_data": info['explanation'],
                    "category": "data_processing",
                    "matched_keyword": keyword,
                    "tools": [
                        "Python (pandas, numpy)",
                        "R (статистика)",
                        "SQL (базы данных)",
                        "Excel/Google Sheets",
                        "Tableau/Power BI"
                    ]
                }), 200
    
    return jsonify({
        "processed_data": "Для обработки данных уточните тему: 'анализ', 'очистка', 'визуализация', 'машинное обучение', 'базы данных' или 'big data'.",
        "available_topics": list(DATA_PROCESSING_KNOWLEDGE.keys()),
        "category": "data_processing"
    }), 200

@app.route('/api/data_processing/analyze', methods=['POST'])
def analyze():
    """Анализ данных"""
    data = request.get_json()
    dataset = data.get('dataset', '')
    
    # Простой анализ CSV данных
    if isinstance(dataset, str) and ',' in dataset:
        try:
            # Парсим CSV строку
            csv_reader = csv.reader(io.StringIO(dataset))
            rows = list(csv_reader)
            
            if len(rows) > 1:
                headers = rows[0]
                data_rows = rows[1:]
                
                analysis = {
                    "rows_count": len(data_rows),
                    "columns_count": len(headers),
                    "headers": headers,
                    "sample_data": data_rows[:3] if len(data_rows) > 3 else data_rows,
                    "analysis": "Данные успешно загружены. Рекомендуется провести очистку и статистический анализ."
                }
                
                return jsonify({
                    "analysis": analysis,
                    "recommendations": [
                        "Проверьте на пропущенные значения",
                        "Удалите дубликаты",
                        "Проверьте типы данных",
                        "Создайте визуализации",
                        "Проведите статистический анализ"
                    ],
                    "category": "data_processing"
                }), 200
        except Exception as e:
            return jsonify({
                "error": f"Ошибка анализа данных: {str(e)}",
                "category": "data_processing"
            }), 400
    
    return jsonify({
        "analysis": "Для анализа данных предоставьте CSV данные или уточните тип анализа.",
        "example_format": "name,age,city\nJohn,25,NYC\nJane,30,LA",
        "category": "data_processing"
    }), 200

if __name__ == '__main__':
    logger.info("📊 Simple Data Processing API Server запущен")
    app.run(port=8101, debug=False, use_reloader=False)
