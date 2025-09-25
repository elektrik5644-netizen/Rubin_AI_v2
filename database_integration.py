#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Database Integration System для Rubin AI
Система интеграции с базами данных для замены жестко закодированных знаний
"""

import sqlite3
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import os

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RubinAIDatabaseManager:
    """Менеджер базы данных для Rubin AI"""
    
    def __init__(self, db_path: str = "rubin_ai_knowledge.db"):
        self.db_path = db_path
        self.connection = None
        self._initialize_database()
        
        logger.info(f"🗄️ Database Manager инициализирован: {db_path}")
    
    def _initialize_database(self):
        """Инициализация базы данных"""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row  # Для доступа к колонкам по имени
            
            # Создаем таблицы
            self._create_tables()
            
            # Заполняем начальными данными
            self._populate_initial_data()
            
            logger.info("✅ База данных инициализирована успешно")
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации базы данных: {e}")
            raise
    
    def _create_tables(self):
        """Создание таблиц базы данных"""
        cursor = self.connection.cursor()
        
        # Таблица категорий
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS categories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                keywords TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Таблица знаний
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_base (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category_id INTEGER,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                keywords TEXT,
                formula TEXT,
                example TEXT,
                difficulty_level INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (category_id) REFERENCES categories (id)
            )
        """)
        
        # Таблица ответов
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS response_templates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category_id INTEGER,
                template TEXT NOT NULL,
                variables TEXT,
                usage_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (category_id) REFERENCES categories (id)
            )
        """)
        
        # Таблица пользовательских запросов
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                category TEXT,
                confidence REAL,
                response TEXT,
                user_feedback INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Таблица обучения
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                category TEXT NOT NULL,
                confidence REAL,
                is_correct BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Таблица метрик
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                category TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.connection.commit()
        logger.info("📊 Таблицы базы данных созданы")
    
    def _populate_initial_data(self):
        """Заполнение начальными данными"""
        cursor = self.connection.cursor()
        
        # Проверяем, есть ли уже данные
        cursor.execute("SELECT COUNT(*) FROM categories")
        if cursor.fetchone()[0] > 0:
            logger.info("📊 База данных уже содержит данные")
            return
        
        # Категории
        categories = [
            ("mathematics", "Математические задачи и вычисления", "решить,уравнение,формула,вычислить,математика,число,функция"),
            ("programming", "Программирование и разработка ПО", "код,программа,алгоритм,функция,класс,переменная,программирование"),
            ("electrical", "Электротехника и электрические схемы", "схема,ток,напряжение,сопротивление,электричество,контур,элемент"),
            ("controllers", "Контроллеры и автоматизация", "контроллер,plc,логика,управление,автоматизация,датчик,исполнитель"),
            ("radiomechanics", "Радиомеханика и радиотехника", "антенна,радио,сигнал,частота,передатчик,приемник,волна"),
            ("general", "Общие вопросы и разговор", "привет,спасибо,как дела,что умеешь,расскажи")
        ]
        
        for name, description, keywords in categories:
            cursor.execute("""
                INSERT INTO categories (name, description, keywords)
                VALUES (?, ?, ?)
            """, (name, description, keywords))
        
        # База знаний
        knowledge_items = [
            (1, "Закон Ома", "U = I × R", "закон ома,напряжение,ток,сопротивление", "U = I × R", "Найти напряжение при токе 2 А и сопротивлении 5 Ом: U = 2 × 5 = 10 В"),
            (1, "Кинетическая энергия", "E = 0.5 × m × v²", "кинетическая энергия,масса,скорость", "E = 0.5 × m × v²", "Рассчитать кинетическую энергию тела массой 10 кг со скоростью 5 м/с"),
            (1, "Квадратное уравнение", "ax² + bx + c = 0", "квадратное уравнение,корни", "x = (-b ± √(b²-4ac)) / 2a", "Решить уравнение x² + 5x + 6 = 0"),
            (2, "Быстрая сортировка", "Алгоритм сортировки с временной сложностью O(n log n)", "быстрая сортировка,алгоритм,сортировка", "", "Реализация алгоритма быстрой сортировки на Python"),
            (2, "Singleton Pattern", "Паттерн проектирования для создания единственного экземпляра класса", "singleton,паттерн,проектирование", "", "Пример реализации паттерна Singleton"),
            (3, "Последовательное соединение", "R = R1 + R2 + R3", "последовательное соединение,резистор", "R = R1 + R2 + R3", "Рассчитать общее сопротивление при последовательном соединении"),
            (3, "Параллельное соединение", "1/R = 1/R1 + 1/R2 + 1/R3", "параллельное соединение,резистор", "1/R = 1/R1 + 1/R2 + 1/R3", "Рассчитать общее сопротивление при параллельном соединении"),
            (4, "Ladder Logic", "Язык программирования для PLC", "ladder logic,plc,программирование", "", "Создание программы управления двигателем на Ladder Logic"),
            (4, "PID регулятор", "Пропорционально-интегрально-дифференциальный регулятор", "pid,регулятор,управление", "", "Настройка PID регулятора для стабилизации температуры"),
            (5, "Дипольная антенна", "Простая антенна из двух проводников", "дипольная антенна,антенна", "", "Расчет параметров дипольной антенны для частоты 2.4 ГГц"),
            (5, "Модуляция AM/FM", "Амплитудная и частотная модуляция", "модуляция,am,fm,радио", "", "Объяснение принципов модуляции AM и FM")
        ]
        
        for category_id, title, content, keywords, formula, example in knowledge_items:
            cursor.execute("""
                INSERT INTO knowledge_base (category_id, title, content, keywords, formula, example)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (category_id, title, content, keywords, formula, example))
        
        # Шаблоны ответов
        response_templates = [
            (1, "Решение математической задачи: {solution}", "solution"),
            (1, "Применяя формулу {formula}, получаем: {answer}", "formula,answer"),
            (2, "Программистское решение: {solution}", "solution"),
            (2, "Код для решения задачи: {code}", "code"),
            (3, "Электротехническое решение: {solution}", "solution"),
            (3, "Анализ схемы: {analysis}", "analysis"),
            (4, "Решение для контроллера: {solution}", "solution"),
            (4, "Программа PLC: {program}", "program"),
            (5, "Радиотехническое решение: {solution}", "solution"),
            (5, "Расчет антенны: {calculation}", "calculation"),
            (6, "Общий ответ: {response}", "response"),
            (6, "Информация: {information}", "information")
        ]
        
        for category_id, template, variables in response_templates:
            cursor.execute("""
                INSERT INTO response_templates (category_id, template, variables)
                VALUES (?, ?, ?)
            """, (category_id, template, variables))
        
        self.connection.commit()
        logger.info("📚 Начальные данные загружены в базу данных")
    
    def get_category_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Получение категории по имени"""
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM categories WHERE name = ?", (name,))
        row = cursor.fetchone()
        
        if row:
            return dict(row)
        return None
    
    def get_knowledge_by_category(self, category_name: str) -> List[Dict[str, Any]]:
        """Получение знаний по категории"""
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT kb.*, c.name as category_name
            FROM knowledge_base kb
            JOIN categories c ON kb.category_id = c.id
            WHERE c.name = ?
        """, (category_name,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_response_template(self, category_name: str) -> Optional[Dict[str, Any]]:
        """Получение шаблона ответа для категории"""
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT rt.*, c.name as category_name
            FROM response_templates rt
            JOIN categories c ON rt.category_id = c.id
            WHERE c.name = ?
            ORDER BY rt.usage_count ASC
            LIMIT 1
        """, (category_name,))
        
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None
    
    def add_user_query(self, question: str, category: str, confidence: float, response: str, feedback: Optional[int] = None):
        """Добавление пользовательского запроса"""
        cursor = self.connection.cursor()
        cursor.execute("""
            INSERT INTO user_queries (question, category, confidence, response, user_feedback)
            VALUES (?, ?, ?, ?, ?)
        """, (question, category, confidence, response, feedback))
        
        self.connection.commit()
        logger.info(f"📝 Пользовательский запрос добавлен: {category}")
    
    def add_training_data(self, question: str, category: str, confidence: float, is_correct: bool):
        """Добавление данных для обучения"""
        cursor = self.connection.cursor()
        cursor.execute("""
            INSERT INTO training_data (question, category, confidence, is_correct)
            VALUES (?, ?, ?, ?)
        """, (question, category, confidence, is_correct))
        
        self.connection.commit()
    
    def update_template_usage(self, template_id: int):
        """Обновление счетчика использования шаблона"""
        cursor = self.connection.cursor()
        cursor.execute("""
            UPDATE response_templates 
            SET usage_count = usage_count + 1
            WHERE id = ?
        """, (template_id,))
        
        self.connection.commit()
    
    def add_metric(self, metric_name: str, metric_value: float, category: Optional[str] = None):
        """Добавление метрики"""
        cursor = self.connection.cursor()
        cursor.execute("""
            INSERT INTO metrics (metric_name, metric_value, category)
            VALUES (?, ?, ?)
        """, (metric_name, metric_value, category))
        
        self.connection.commit()
    
    def get_metrics(self, metric_name: Optional[str] = None, category: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Получение метрик"""
        cursor = self.connection.cursor()
        
        query = "SELECT * FROM metrics WHERE 1=1"
        params = []
        
        if metric_name:
            query += " AND metric_name = ?"
            params.append(metric_name)
        
        if category:
            query += " AND category = ?"
            params.append(category)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики базы данных"""
        cursor = self.connection.cursor()
        
        stats = {}
        
        # Количество категорий
        cursor.execute("SELECT COUNT(*) FROM categories")
        stats['total_categories'] = cursor.fetchone()[0]
        
        # Количество знаний
        cursor.execute("SELECT COUNT(*) FROM knowledge_base")
        stats['total_knowledge'] = cursor.fetchone()[0]
        
        # Количество шаблонов ответов
        cursor.execute("SELECT COUNT(*) FROM response_templates")
        stats['total_templates'] = cursor.fetchone()[0]
        
        # Количество пользовательских запросов
        cursor.execute("SELECT COUNT(*) FROM user_queries")
        stats['total_queries'] = cursor.fetchone()[0]
        
        # Количество данных для обучения
        cursor.execute("SELECT COUNT(*) FROM training_data")
        stats['total_training_data'] = cursor.fetchone()[0]
        
        # Статистика по категориям
        cursor.execute("""
            SELECT c.name, COUNT(kb.id) as knowledge_count
            FROM categories c
            LEFT JOIN knowledge_base kb ON c.id = kb.category_id
            GROUP BY c.id, c.name
        """)
        stats['category_stats'] = [dict(row) for row in cursor.fetchall()]
        
        return stats
    
    def search_knowledge(self, query: str, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Поиск в базе знаний"""
        cursor = self.connection.cursor()
        
        if category:
            cursor.execute("""
                SELECT kb.*, c.name as category_name
                FROM knowledge_base kb
                JOIN categories c ON kb.category_id = c.id
                WHERE c.name = ? AND (
                    kb.title LIKE ? OR 
                    kb.content LIKE ? OR 
                    kb.keywords LIKE ? OR
                    kb.example LIKE ?
                )
            """, (category, f"%{query}%", f"%{query}%", f"%{query}%", f"%{query}%"))
        else:
            cursor.execute("""
                SELECT kb.*, c.name as category_name
                FROM knowledge_base kb
                JOIN categories c ON kb.category_id = c.id
                WHERE kb.title LIKE ? OR 
                      kb.content LIKE ? OR 
                      kb.keywords LIKE ? OR
                      kb.example LIKE ?
            """, (f"%{query}%", f"%{query}%", f"%{query}%", f"%{query}%"))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def close(self):
        """Закрытие соединения с базой данных"""
        if self.connection:
            self.connection.close()
            logger.info("🔒 Соединение с базой данных закрыто")

class DatabaseIntegratedRubinAI:
    """Rubin AI с интеграцией базы данных"""
    
    def __init__(self):
        self.db_manager = RubinAIDatabaseManager()
        logger.info("🧠 Database Integrated Rubin AI инициализирован")
    
    def categorize_question(self, question: str) -> Tuple[str, float]:
        """Категоризация вопроса с использованием базы данных"""
        question_lower = question.lower()
        
        # Получаем все категории из базы данных
        cursor = self.db_manager.connection.cursor()
        cursor.execute("SELECT name, keywords FROM categories")
        categories = cursor.fetchall()
        
        scores = {}
        for name, keywords in categories:
            keyword_list = keywords.split(',')
            score = sum(1 for keyword in keyword_list if keyword in question_lower)
            scores[name] = score
        
        if max(scores.values()) > 0:
            best_category = max(scores, key=scores.get)
            category_data = self.db_manager.get_category_by_name(best_category)
            if category_data:
                keyword_count = len(category_data['keywords'].split(','))
                confidence = min(0.9, scores[best_category] / keyword_count)
            else:
                confidence = 0.5
            return best_category, confidence
        else:
            return 'general', 0.5
    
    def generate_response(self, question: str) -> Dict[str, Any]:
        """Генерация ответа с использованием базы данных"""
        try:
            logger.info(f"🧠 Database Integrated Rubin AI обрабатывает: {question[:50]}...")
            
            # Категоризация
            category, confidence = self.categorize_question(question)
            
            # Поиск релевантных знаний
            knowledge_items = self.db_manager.search_knowledge(question, category)
            
            # Получение шаблона ответа
            template_data = self.db_manager.get_response_template(category)
            
            # Формирование ответа
            if knowledge_items:
                # Используем первое найденное знание
                knowledge = knowledge_items[0]
                response = self._format_response(knowledge, template_data)
            else:
                # Используем общий ответ
                response = self._get_general_response(category, template_data)
            
            # Сохраняем запрос в базу данных
            self.db_manager.add_user_query(question, category, confidence, response)
            
            # Обновляем счетчик использования шаблона
            if template_data:
                self.db_manager.update_template_usage(template_data['id'])
            
            result = {
                'response': response,
                'category': category,
                'confidence': confidence,
                'method': 'database_integrated',
                'timestamp': datetime.now().isoformat(),
                'knowledge_used': len(knowledge_items),
                'template_id': template_data['id'] if template_data else None
            }
            
            logger.info(f"✅ Ответ сгенерирован: {category} (уверенность: {confidence:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Ошибка генерации ответа: {e}")
            return {
                'response': 'Произошла ошибка при обработке вопроса',
                'category': 'error',
                'confidence': 0.0,
                'method': 'error_handler',
                'timestamp': datetime.now().isoformat(),
                'knowledge_used': 0,
                'template_id': None,
                'error': str(e)
            }
    
    def _format_response(self, knowledge: Dict[str, Any], template_data: Optional[Dict[str, Any]]) -> str:
        """Форматирование ответа на основе знаний и шаблона"""
        if template_data:
            template = template_data['template']
            variables = template_data['variables'].split(',') if template_data['variables'] else []
            
            # Заполняем переменные
            response = template
            if 'solution' in variables:
                response = response.replace('{solution}', knowledge['content'])
            if 'formula' in variables and knowledge['formula']:
                response = response.replace('{formula}', knowledge['formula'])
            if 'answer' in variables and knowledge['example']:
                response = response.replace('{answer}', knowledge['example'])
            
            return response
        else:
            return f"{knowledge['title']}: {knowledge['content']}"
    
    def _get_general_response(self, category: str, template_data: Optional[Dict[str, Any]]) -> str:
        """Получение общего ответа"""
        if template_data:
            template = template_data['template']
            if '{response}' in template:
                return template.replace('{response}', f"Отвечаю на вопрос по категории {category}")
            elif '{information}' in template:
                return template.replace('{information}', f"Информация по теме {category}")
        
        return f"Это вопрос по теме {category}. Могу предоставить дополнительную информацию."
    
    def get_database_statistics(self) -> Dict[str, Any]:
        """Получение статистики базы данных"""
        return self.db_manager.get_statistics()
    
    def add_knowledge(self, category_name: str, title: str, content: str, keywords: str = "", formula: str = "", example: str = ""):
        """Добавление нового знания в базу данных"""
        category = self.db_manager.get_category_by_name(category_name)
        if not category:
            logger.error(f"❌ Категория {category_name} не найдена")
            return False
        
        cursor = self.db_manager.connection.cursor()
        cursor.execute("""
            INSERT INTO knowledge_base (category_id, title, content, keywords, formula, example)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (category['id'], title, content, keywords, formula, example))
        
        self.db_manager.connection.commit()
        logger.info(f"📚 Новое знание добавлено: {title}")
        return True
    
    def close(self):
        """Закрытие соединения с базой данных"""
        self.db_manager.close()

def main():
    """Основная функция для тестирования интеграции с базой данных"""
    print("🗄️ ТЕСТИРОВАНИЕ ИНТЕГРАЦИИ С БАЗОЙ ДАННЫХ")
    print("=" * 50)
    
    # Создаем Rubin AI с интеграцией базы данных
    rubin_ai = DatabaseIntegratedRubinAI()
    
    # Тестовые вопросы
    test_questions = [
        "Реши уравнение x^2 + 5x + 6 = 0",
        "Напиши код на Python для сортировки массива",
        "Рассчитай ток в цепи с сопротивлением 10 Ом и напряжением 220 В",
        "Создай программу PLC для управления двигателем",
        "Рассчитай параметры антенны для частоты 2.4 ГГц",
        "Привет! Как дела?"
    ]
    
    print("\n🧪 ТЕСТИРОВАНИЕ ОТВЕТОВ:")
    print("-" * 30)
    
    for question in test_questions:
        print(f"\n📝 Вопрос: {question}")
        response = rubin_ai.generate_response(question)
        print(f"🎯 Категория: {response['category']}")
        print(f"📊 Уверенность: {response['confidence']:.3f}")
        print(f"💡 Ответ: {response['response']}")
        print(f"📚 Использовано знаний: {response['knowledge_used']}")
    
    # Статистика базы данных
    print("\n📊 СТАТИСТИКА БАЗЫ ДАННЫХ:")
    print("-" * 30)
    stats = rubin_ai.get_database_statistics()
    
    print(f"📁 Категорий: {stats['total_categories']}")
    print(f"📚 Знаний: {stats['total_knowledge']}")
    print(f"📝 Шаблонов ответов: {stats['total_templates']}")
    print(f"❓ Пользовательских запросов: {stats['total_queries']}")
    print(f"🎓 Данных для обучения: {stats['total_training_data']}")
    
    print("\n📊 Статистика по категориям:")
    for cat_stat in stats['category_stats']:
        print(f"  - {cat_stat['name']}: {cat_stat['knowledge_count']} знаний")
    
    # Тестирование поиска
    print("\n🔍 ТЕСТИРОВАНИЕ ПОИСКА:")
    print("-" * 25)
    
    search_results = rubin_ai.db_manager.search_knowledge("закон ома")
    print(f"🔍 Поиск 'закон ома': найдено {len(search_results)} результатов")
    for result in search_results[:2]:  # Показываем первые 2
        print(f"  - {result['title']}: {result['content'][:50]}...")
    
    # Закрываем соединение
    rubin_ai.close()
    
    print("\n✅ Тестирование интеграции с базой данных завершено!")

if __name__ == "__main__":
    main()
