#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Улучшение векторного поиска и добавление синонимов
"""

import sqlite3
import json
import re
from typing import Dict, List, Set

class VectorSearchImprover:
    def __init__(self, db_path="rubin_ai_v2.db"):
        self.db_path = db_path
        self.connection = None
        
        # Словарь синонимов для технических терминов
        self.synonyms = {
            # ПИД-регуляторы
            "ПИД": ["PID", "пид", "pid", "регулятор", "контроллер", "управление"],
            "PID": ["ПИД", "пид", "pid", "регулятор", "контроллер", "управление"],
            "регулятор": ["ПИД", "PID", "пид", "pid", "контроллер", "управление"],
            "контроллер": ["ПИД", "PID", "пид", "pid", "регулятор", "управление"],
            
            # Электротехника
            "электротехника": ["электрика", "электротехнический", "электрический"],
            "электрика": ["электротехника", "электротехнический", "электрический"],
            "закон ома": ["ом", "сопротивление", "напряжение", "ток"],
            "ом": ["закон ома", "сопротивление", "напряжение", "ток"],
            
            # Программирование
            "python": ["питон", "пайтон", "программирование", "код"],
            "питон": ["python", "пайтон", "программирование", "код"],
            "программирование": ["python", "питон", "пайтон", "код", "алгоритм"],
            "код": ["python", "питон", "пайтон", "программирование", "алгоритм"],
            
            # Радиомеханика
            "антенна": ["антенны", "радио", "радиоволны", "передача"],
            "антенны": ["антенна", "радио", "радиоволны", "передача"],
            "радио": ["антенна", "антенны", "радиоволны", "передача"],
            "дипольная": ["диполь", "антенна", "полуволновая"],
            
            # Автоматизация
            "автоматизация": ["автоматический", "автомат", "управление", "система"],
            "автоматический": ["автоматизация", "автомат", "управление", "система"],
            "управление": ["автоматизация", "автоматический", "система", "контроль"],
            "система": ["автоматизация", "автоматический", "управление", "контроль"],
            
            # ПЛК и контроллеры
            "ПЛК": ["PLC", "plc", "контроллер", "программируемый"],
            "PLC": ["ПЛК", "plc", "контроллер", "программируемый"],
            "ladder": ["лестница", "логика", "программирование ПЛК"],
            "лестница": ["ladder", "логика", "программирование ПЛК"],
            
            # Температура и датчики
            "температура": ["термометр", "датчик", "нагрев", "охлаждение"],
            "датчик": ["сенсор", "измерение", "сигнал"],
            "сенсор": ["датчик", "измерение", "сигнал"]
        }
        
        # Ключевые фразы для улучшения поиска
        self.key_phrases = {
            "ПИД-регулятор": ["ПИД", "PID", "регулятор", "контроллер", "управление", "обратная связь"],
            "закон Ома": ["закон ома", "ом", "сопротивление", "напряжение", "ток", "U=I*R"],
            "Python программирование": ["python", "питон", "программирование", "код", "алгоритм"],
            "дипольная антенна": ["дипольная", "антенна", "радио", "передача", "полуволновая"],
            "система автоматизации": ["автоматизация", "система", "управление", "контроль"],
            "ПЛК программирование": ["ПЛК", "PLC", "ladder", "лестница", "программирование"]
        }

    def connect(self):
        """Подключение к базе данных"""
        try:
            self.connection = sqlite3.connect(self.db_path)
            return True
        except sqlite3.Error as e:
            print(f"❌ Ошибка подключения к базе данных: {e}")
            return False

    def create_synonyms_table(self):
        """Создание таблицы синонимов"""
        try:
            cursor = self.connection.cursor()
            
            # Создаем таблицу синонимов
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS synonyms (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    term TEXT NOT NULL,
                    synonym TEXT NOT NULL,
                    category TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(term, synonym)
                )
            """)
            
            # Создаем индексы
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_synonyms_term ON synonyms(term)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_synonyms_synonym ON synonyms(synonym)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_synonyms_category ON synonyms(category)")
            
            self.connection.commit()
            print("✅ Таблица синонимов создана")
            return True
            
        except sqlite3.Error as e:
            print(f"❌ Ошибка создания таблицы синонимов: {e}")
            return False

    def populate_synonyms(self):
        """Заполнение таблицы синонимов"""
        try:
            cursor = self.connection.cursor()
            
            # Очищаем таблицу
            cursor.execute("DELETE FROM synonyms")
            
            # Добавляем синонимы
            for term, synonyms_list in self.synonyms.items():
                for synonym in synonyms_list:
                    # Определяем категорию
                    category = self._get_category(term)
                    
                    cursor.execute("""
                        INSERT OR IGNORE INTO synonyms (term, synonym, category)
                        VALUES (?, ?, ?)
                    """, (term.lower(), synonym.lower(), category))
            
            self.connection.commit()
            print(f"✅ Добавлено {len(self.synonyms)} терминов с синонимами")
            return True
            
        except sqlite3.Error as e:
            print(f"❌ Ошибка заполнения синонимов: {e}")
            return False

    def _get_category(self, term: str) -> str:
        """Определение категории термина"""
        term_lower = term.lower()
        
        if any(keyword in term_lower for keyword in ["пид", "pid", "регулятор", "контроллер"]):
            return "controllers"
        elif any(keyword in term_lower for keyword in ["электро", "электрик", "ом", "напряжение"]):
            return "electrical"
        elif any(keyword in term_lower for keyword in ["python", "питон", "программирование", "код"]):
            return "programming"
        elif any(keyword in term_lower for keyword in ["антенна", "радио", "диполь"]):
            return "radiomechanics"
        elif any(keyword in term_lower for keyword in ["автоматизация", "система", "управление"]):
            return "automation"
        else:
            return "general"

    def improve_document_content(self):
        """Улучшение содержимого документов с помощью синонимов"""
        try:
            cursor = self.connection.cursor()
            
            # Получаем все документы
            cursor.execute("SELECT id, content, category FROM documents")
            documents = cursor.fetchall()
            
            improved_count = 0
            
            for doc_id, content, category in documents:
                # Добавляем синонимы к содержимому
                enhanced_content = self._enhance_content_with_synonyms(content, category)
                
                if enhanced_content != content:
                    cursor.execute("""
                        UPDATE documents 
                        SET content = ?, enhanced_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                    """, (enhanced_content, doc_id))
                    improved_count += 1
            
            self.connection.commit()
            print(f"✅ Улучшено {improved_count} документов")
            return True
            
        except sqlite3.Error as e:
            print(f"❌ Ошибка улучшения документов: {e}")
            return False

    def _enhance_content_with_synonyms(self, content: str, category: str) -> str:
        """Добавление синонимов к содержимому документа"""
        enhanced_content = content
        
        # Получаем синонимы для данной категории
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT term, synonym FROM synonyms 
            WHERE category = ? OR category = 'general'
        """, (category,))
        
        synonyms_data = cursor.fetchall()
        
        # Добавляем синонимы в конец документа
        if synonyms_data:
            synonyms_text = "\n\n<!-- Синонимы для улучшения поиска -->\n"
            synonyms_text += "Ключевые термины: "
            
            terms = set()
            for term, synonym in synonyms_data:
                if term in content.lower():
                    terms.add(term)
                    terms.add(synonym)
            
            synonyms_text += ", ".join(sorted(terms))
            enhanced_content += synonyms_text
        
        return enhanced_content

    def create_search_improvements(self):
        """Создание улучшений для поиска"""
        try:
            cursor = self.connection.cursor()
            
            # Создаем таблицу для улучшенного поиска
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS search_improvements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    original_query TEXT NOT NULL,
                    enhanced_query TEXT NOT NULL,
                    category TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Добавляем улучшенные запросы
            improvements = [
                ("ПИД-регулятор", "ПИД PID регулятор контроллер управление обратная связь", "controllers"),
                ("закон Ома", "закон ома ом сопротивление напряжение ток U=I*R", "electrical"),
                ("Python код", "python питон программирование код алгоритм", "programming"),
                ("антенна", "антенна радио радиоволны передача дипольная", "radiomechanics"),
                ("автоматизация", "автоматизация система управление контроль", "automation")
            ]
            
            for original, enhanced, category in improvements:
                cursor.execute("""
                    INSERT OR IGNORE INTO search_improvements (original_query, enhanced_query, category)
                    VALUES (?, ?, ?)
                """, (original, enhanced, category))
            
            self.connection.commit()
            print("✅ Улучшения поиска созданы")
            return True
            
        except sqlite3.Error as e:
            print(f"❌ Ошибка создания улучшений поиска: {e}")
            return False

    def get_statistics(self):
        """Получение статистики улучшений"""
        try:
            cursor = self.connection.cursor()
            
            # Статистика синонимов
            cursor.execute("SELECT COUNT(*) FROM synonyms")
            synonyms_count = cursor.fetchone()[0]
            
            # Статистика документов
            cursor.execute("SELECT COUNT(*) FROM documents")
            documents_count = cursor.fetchone()[0]
            
            # Статистика по категориям
            cursor.execute("""
                SELECT category, COUNT(*) 
                FROM synonyms 
                GROUP BY category 
                ORDER BY COUNT(*) DESC
            """)
            categories_stats = cursor.fetchall()
            
            print(f"\n📊 СТАТИСТИКА УЛУЧШЕНИЙ:")
            print(f"   📝 Документов в базе: {documents_count}")
            print(f"   🔗 Синонимов добавлено: {synonyms_count}")
            print(f"   📂 Категории синонимов:")
            
            for category, count in categories_stats:
                print(f"      - {category}: {count} синонимов")
            
            return True
            
        except sqlite3.Error as e:
            print(f"❌ Ошибка получения статистики: {e}")
            return False

    def close(self):
        """Закрытие соединения"""
        if self.connection:
            self.connection.close()

def main():
    """Основная функция"""
    print("=== УЛУЧШЕНИЕ ВЕКТОРНОГО ПОИСКА RUBIN AI ===\n")
    
    improver = VectorSearchImprover()
    
    if not improver.connect():
        return
    
    print("1. Создание таблицы синонимов...")
    improver.create_synonyms_table()
    
    print("2. Заполнение синонимов...")
    improver.populate_synonyms()
    
    print("3. Улучшение содержимого документов...")
    improver.improve_document_content()
    
    print("4. Создание улучшений поиска...")
    improver.create_search_improvements()
    
    print("5. Получение статистики...")
    improver.get_statistics()
    
    improver.close()
    
    print(f"\n🎉 УЛУЧШЕНИЯ ЗАВЕРШЕНЫ!")
    print(f"   Система готова к перезапуску с улучшенным поиском")

if __name__ == "__main__":
    main()

















