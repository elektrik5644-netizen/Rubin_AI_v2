#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Центральная База Знаний Rubin AI
Единое хранилище всех знаний с автоматическим предложением новых фактов
"""

import sqlite3
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import hashlib
import re
from pathlib import Path

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CentralKnowledgeBase:
    """Центральная База Знаний Rubin AI"""
    
    def __init__(self, db_path: str = "rubin_knowledge.db"):
        self.db_path = db_path
        self.init_database()
        self.pending_suggestions = []  # Ожидающие подтверждения предложения
        self.user_feedback_history = []  # История обратной связи
        
    def init_database(self):
        """Инициализация базы данных"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Таблица основных знаний
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS knowledge_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    keywords TEXT,
                    formulas TEXT,
                    examples TEXT,
                    confidence REAL DEFAULT 1.0,
                    source TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    usage_count INTEGER DEFAULT 0,
                    user_rating REAL DEFAULT 0.0,
                    is_verified BOOLEAN DEFAULT FALSE
                )
            ''')
            
            # Таблица связей между знаниями
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS knowledge_relations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    from_fact_id INTEGER,
                    to_fact_id INTEGER,
                    relation_type TEXT NOT NULL,
                    strength REAL DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (from_fact_id) REFERENCES knowledge_facts (id),
                    FOREIGN KEY (to_fact_id) REFERENCES knowledge_facts (id)
                )
            ''')
            
            # Таблица предложений новых знаний
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS knowledge_suggestions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    keywords TEXT,
                    formulas TEXT,
                    examples TEXT,
                    confidence REAL DEFAULT 0.5,
                    source TEXT,
                    suggested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'pending',
                    user_feedback TEXT,
                    feedback_at TIMESTAMP
                )
            ''')
            
            # Таблица истории обучения
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS learning_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    category TEXT,
                    confidence REAL,
                    user_rating INTEGER,
                    feedback TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Создаем индексы для быстрого поиска
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_category ON knowledge_facts(category)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_keywords ON knowledge_facts(keywords)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_title ON knowledge_facts(title)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_suggestions_status ON knowledge_suggestions(status)')
            
            conn.commit()
            conn.close()
            
            logger.info("✅ Центральная База Знаний инициализирована")
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации БД: {e}")
    
    def add_knowledge(self, category: str, title: str, content: str, 
                     keywords: str = "", formulas: str = "", examples: str = "",
                     confidence: float = 1.0, source: str = "") -> bool:
        """Добавляет новое знание в базу"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Проверяем, не существует ли уже такое знание
            cursor.execute('''
                SELECT id FROM knowledge_facts 
                WHERE category = ? AND title = ? AND content = ?
            ''', (category, title, content))
            
            if cursor.fetchone():
                logger.warning(f"⚠️ Знание уже существует: {title}")
                conn.close()
                return False
            
            # Добавляем новое знание
            cursor.execute('''
                INSERT INTO knowledge_facts 
                (category, title, content, keywords, formulas, examples, confidence, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (category, title, content, keywords, formulas, examples, confidence, source))
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ Добавлено знание: {title} в категории {category}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка добавления знания: {e}")
            return False
    
    def suggest_knowledge(self, category: str, title: str, content: str,
                         keywords: str = "", formulas: str = "", examples: str = "",
                         confidence: float = 0.5, source: str = "") -> int:
        """Предлагает новое знание для подтверждения пользователем"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Проверяем, не предлагалось ли уже
            cursor.execute('''
                SELECT id FROM knowledge_suggestions 
                WHERE category = ? AND title = ? AND status = 'pending'
            ''', (category, title))
            
            if cursor.fetchone():
                logger.warning(f"⚠️ Предложение уже ожидает подтверждения: {title}")
                conn.close()
                return -1
            
            # Добавляем предложение
            cursor.execute('''
                INSERT INTO knowledge_suggestions 
                (category, title, content, keywords, formulas, examples, confidence, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (category, title, content, keywords, formulas, examples, confidence, source))
            
            suggestion_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            logger.info(f"💡 Предложено новое знание: {title} в категории {category}")
            return suggestion_id
            
        except Exception as e:
            logger.error(f"❌ Ошибка предложения знания: {e}")
            return -1
    
    def get_pending_suggestions(self) -> List[Dict]:
        """Получает все ожидающие подтверждения предложения"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, category, title, content, keywords, formulas, examples, 
                       confidence, source, suggested_at
                FROM knowledge_suggestions 
                WHERE status = 'pending'
                ORDER BY suggested_at DESC
            ''')
            
            suggestions = []
            for row in cursor.fetchall():
                suggestions.append({
                    'id': row[0],
                    'category': row[1],
                    'title': row[2],
                    'content': row[3],
                    'keywords': row[4],
                    'formulas': row[5],
                    'examples': row[6],
                    'confidence': row[7],
                    'source': row[8],
                    'suggested_at': row[9]
                })
            
            conn.close()
            return suggestions
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения предложений: {e}")
            return []
    
    def approve_suggestion(self, suggestion_id: int, user_feedback: str = "") -> bool:
        """Подтверждает предложение и добавляет его в основную базу"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Получаем предложение
            cursor.execute('''
                SELECT category, title, content, keywords, formulas, examples, confidence, source
                FROM knowledge_suggestions WHERE id = ? AND status = 'pending'
            ''', (suggestion_id,))
            
            suggestion = cursor.fetchone()
            if not suggestion:
                logger.warning(f"⚠️ Предложение {suggestion_id} не найдено")
                conn.close()
                return False
            
            # Добавляем в основную базу
            cursor.execute('''
                INSERT INTO knowledge_facts 
                (category, title, content, keywords, formulas, examples, confidence, source, is_verified)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, TRUE)
            ''', suggestion)
            
            # Обновляем статус предложения
            cursor.execute('''
                UPDATE knowledge_suggestions 
                SET status = 'approved', user_feedback = ?, feedback_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (user_feedback, suggestion_id))
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ Предложение {suggestion_id} подтверждено и добавлено в базу")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка подтверждения предложения: {e}")
            return False
    
    def reject_suggestion(self, suggestion_id: int, user_feedback: str = "") -> bool:
        """Отклоняет предложение"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE knowledge_suggestions 
                SET status = 'rejected', user_feedback = ?, feedback_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (user_feedback, suggestion_id))
            
            conn.commit()
            conn.close()
            
            logger.info(f"❌ Предложение {suggestion_id} отклонено")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка отклонения предложения: {e}")
            return False
    
    def search_knowledge(self, query: str, category: str = None, limit: int = 10) -> List[Dict]:
        """Ищет знания по запросу"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Простой поиск по ключевым словам и содержимому
            if category:
                cursor.execute('''
                    SELECT id, category, title, content, keywords, formulas, examples, 
                           confidence, usage_count, user_rating
                    FROM knowledge_facts 
                    WHERE category = ? AND (
                        title LIKE ? OR content LIKE ? OR keywords LIKE ?
                    )
                    ORDER BY confidence DESC, usage_count DESC
                    LIMIT ?
                ''', (category, f'%{query}%', f'%{query}%', f'%{query}%', limit))
            else:
                cursor.execute('''
                    SELECT id, category, title, content, keywords, formulas, examples, 
                           confidence, usage_count, user_rating
                    FROM knowledge_facts 
                    WHERE title LIKE ? OR content LIKE ? OR keywords LIKE ?
                    ORDER BY confidence DESC, usage_count DESC
                    LIMIT ?
                ''', (f'%{query}%', f'%{query}%', f'%{query}%', limit))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'id': row[0],
                    'category': row[1],
                    'title': row[2],
                    'content': row[3],
                    'keywords': row[4],
                    'formulas': row[5],
                    'examples': row[6],
                    'confidence': row[7],
                    'usage_count': row[8],
                    'user_rating': row[9]
                })
            
            conn.close()
            return results
            
        except Exception as e:
            logger.error(f"❌ Ошибка поиска знаний: {e}")
            return []
    
    def get_knowledge_stats(self) -> Dict:
        """Получает статистику базы знаний"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Общая статистика
            cursor.execute('SELECT COUNT(*) FROM knowledge_facts')
            total_facts = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM knowledge_suggestions WHERE status = "pending"')
            pending_suggestions = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM knowledge_suggestions WHERE status = "approved"')
            approved_suggestions = cursor.fetchone()[0]
            
            # Статистика по категориям
            cursor.execute('''
                SELECT category, COUNT(*) as count 
                FROM knowledge_facts 
                GROUP BY category 
                ORDER BY count DESC
            ''')
            categories = dict(cursor.fetchall())
            
            conn.close()
            
            return {
                'total_facts': total_facts,
                'pending_suggestions': pending_suggestions,
                'approved_suggestions': approved_suggestions,
                'categories': categories
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения статистики: {e}")
            return {}
    
    def load_initial_knowledge(self):
        """Загружает начальные знания из существующих модулей"""
        try:
            logger.info("🔄 Загрузка начальных знаний...")
            
            # Электротехнические знания
            electrical_knowledge = [
                {
                    'category': 'электротехника',
                    'title': 'Закон Ома',
                    'content': 'Закон Ома устанавливает связь между напряжением, током и сопротивлением в электрической цепи.',
                    'keywords': 'напряжение, ток, сопротивление, закон ома',
                    'formulas': 'U = I * R, I = U / R, R = U / I',
                    'examples': 'Если напряжение 12В, а сопротивление 4Ом, то ток = 12/4 = 3А'
                },
                {
                    'category': 'электротехника',
                    'title': 'Мощность электрического тока',
                    'content': 'Мощность электрического тока - это работа, совершаемая электрическим полем за единицу времени.',
                    'keywords': 'мощность, ток, напряжение, работа',
                    'formulas': 'P = U * I, P = I² * R, P = U² / R',
                    'examples': 'При напряжении 220В и токе 2А мощность = 220 * 2 = 440Вт'
                },
                {
                    'category': 'электротехника',
                    'title': 'Конденсатор',
                    'content': 'Конденсатор - это устройство для накопления электрического заряда и энергии электрического поля.',
                    'keywords': 'конденсатор, заряд, емкость, накопление',
                    'formulas': 'Q = C * U, C = Q / U',
                    'examples': 'Конденсатор емкостью 1000мкФ при напряжении 12В накапливает заряд 0.012Кл'
                }
            ]
            
            # Математические знания
            math_knowledge = [
                {
                    'category': 'математика',
                    'title': 'Квадратное уравнение',
                    'content': 'Квадратное уравнение - это уравнение вида ax² + bx + c = 0, где a ≠ 0.',
                    'keywords': 'квадратное уравнение, дискриминант, корни',
                    'formulas': 'D = b² - 4ac, x = (-b ± √D) / 2a',
                    'examples': 'x² - 5x + 6 = 0: D = 25-24 = 1, x₁ = 3, x₂ = 2'
                },
                {
                    'category': 'математика',
                    'title': 'Теорема Пифагора',
                    'content': 'В прямоугольном треугольнике квадрат гипотенузы равен сумме квадратов катетов.',
                    'keywords': 'пифагор, треугольник, гипотенуза, катеты',
                    'formulas': 'c² = a² + b²',
                    'examples': 'Если катеты 3 и 4, то гипотенуза = √(3² + 4²) = 5'
                }
            ]
            
            # Программирование
            programming_knowledge = [
                {
                    'category': 'программирование',
                    'title': 'Цикл for в Python',
                    'content': 'Цикл for используется для итерации по последовательности элементов.',
                    'keywords': 'цикл, for, python, итерация',
                    'formulas': 'for item in sequence:',
                    'examples': 'for i in range(5): print(i)  # выведет 0,1,2,3,4'
                }
            ]
            
            # Добавляем все знания
            all_knowledge = electrical_knowledge + math_knowledge + programming_knowledge
            
            for knowledge in all_knowledge:
                self.add_knowledge(**knowledge)
            
            logger.info(f"✅ Загружено {len(all_knowledge)} начальных знаний")
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки начальных знаний: {e}")

class KnowledgeSuggestionEngine:
    """Движок для автоматического предложения новых знаний"""
    
    def __init__(self, knowledge_base: CentralKnowledgeBase):
        self.kb = knowledge_base
        self.suggestion_patterns = [
            # Паттерны для электротехники
            (r'что такое (\w+)', 'электротехника', 'Определение'),
            (r'как работает (\w+)', 'электротехника', 'Принцип работы'),
            (r'формула для (\w+)', 'электротехника', 'Формула'),
            
            # Паттерны для математики
            (r'реши уравнение', 'математика', 'Решение уравнения'),
            (r'вычисли (\w+)', 'математика', 'Вычисление'),
            (r'формула (\w+)', 'математика', 'Математическая формула'),
            
            # Паттерны для программирования
            (r'как написать (\w+)', 'программирование', 'Программирование'),
            (r'синтаксис (\w+)', 'программирование', 'Синтаксис'),
        ]
    
    def analyze_question(self, question: str) -> Optional[Dict]:
        """Анализирует вопрос и предлагает новое знание"""
        try:
            question_lower = question.lower()
            
            # Проверяем паттерны
            for pattern, category, knowledge_type in self.suggestion_patterns:
                match = re.search(pattern, question_lower)
                if match:
                    # Извлекаем ключевое слово
                    keyword = match.group(1) if match.groups() else question_lower
                    
                    # Проверяем, есть ли уже такое знание
                    existing = self.kb.search_knowledge(keyword, category)
                    if not existing:
                        # Предлагаем новое знание
                        suggestion = {
                            'category': category,
                            'title': f"{knowledge_type}: {keyword}",
                            'content': f"Информация о {keyword} в контексте {category}",
                            'keywords': keyword,
                            'confidence': 0.6,
                            'source': 'auto_suggestion'
                        }
                        
                        suggestion_id = self.kb.suggest_knowledge(**suggestion)
                        if suggestion_id > 0:
                            return {
                                'suggestion_id': suggestion_id,
                                'suggestion': suggestion,
                                'reason': f"Обнаружен вопрос о {keyword}, но знания нет в базе"
                            }
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Ошибка анализа вопроса: {e}")
            return None
    
    def generate_suggestion_message(self, suggestion_data: Dict) -> str:
        """Генерирует сообщение с предложением нового знания"""
        suggestion = suggestion_data['suggestion']
        reason = suggestion_data['reason']
        
        message = f"""
💡 **Rubin AI предлагает добавить новое знание:**

**Категория:** {suggestion['category']}
**Название:** {suggestion['title']}
**Содержание:** {suggestion['content']}
**Ключевые слова:** {suggestion['keywords']}

**Причина:** {reason}

**Хотите добавить это знание в базу?**
- ✅ Да, добавить
- ❌ Нет, не нужно
- ✏️ Да, но с изменениями

*Используйте команды: approve {suggestion_data['suggestion_id']}, reject {suggestion_data['suggestion_id']}, или edit {suggestion_data['suggestion_id']}*
"""
        return message

# Глобальный экземпляр базы знаний
knowledge_base = None

def get_knowledge_base():
    """Получает глобальный экземпляр базы знаний"""
    global knowledge_base
    if knowledge_base is None:
        knowledge_base = CentralKnowledgeBase()
        knowledge_base.load_initial_knowledge()
    return knowledge_base

def get_suggestion_engine():
    """Получает движок предложений"""
    return KnowledgeSuggestionEngine(get_knowledge_base())

if __name__ == "__main__":
    print("🚀 Тестирование Центральной Базы Знаний Rubin AI")
    
    # Инициализация
    kb = get_knowledge_base()
    engine = get_suggestion_engine()
    
    # Статистика
    stats = kb.get_knowledge_stats()
    print(f"\n📊 Статистика базы знаний:")
    print(f"• Всего фактов: {stats['total_facts']}")
    print(f"• Ожидающих подтверждения: {stats['pending_suggestions']}")
    print(f"• Подтвержденных предложений: {stats['approved_suggestions']}")
    print(f"• Категории: {stats['categories']}")
    
    # Тестирование поиска
    print(f"\n🔍 Тестирование поиска:")
    results = kb.search_knowledge("закон ома")
    for result in results:
        print(f"• {result['title']}: {result['content'][:50]}...")
    
    # Тестирование предложений
    print(f"\n💡 Тестирование предложений:")
    test_questions = [
        "Что такое транзистор?",
        "Как работает ШИМ?",
        "Реши уравнение x^2 + 5x + 6 = 0"
    ]
    
    for question in test_questions:
        suggestion = engine.analyze_question(question)
        if suggestion:
            print(f"• Вопрос: {question}")
            print(f"  Предложение: {suggestion['suggestion']['title']}")
            print(f"  Причина: {suggestion['reason']}")
    
    # Показываем ожидающие предложения
    pending = kb.get_pending_suggestions()
    if pending:
        print(f"\n⏳ Ожидающие подтверждения предложения:")
        for suggestion in pending:
            print(f"• ID {suggestion['id']}: {suggestion['title']}")
    
    print(f"\n✅ Тестирование завершено!")










