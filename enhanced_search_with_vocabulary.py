#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Улучшенная система поиска с использованием технического словаря
"""

import sqlite3
import re
from typing import List, Dict, Tuple, Set
from datetime import datetime

class EnhancedSearchWithVocabulary:
    """Класс для улучшенного поиска с использованием словаря"""
    
    def __init__(self, db_path: str = "rubin_ai_v2.db"):
        self.db_path = db_path
        self.connection = None
        self.connect_database()
    
    def connect_database(self):
        """Подключение к базе данных"""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.execute("PRAGMA foreign_keys = ON")
            print("✅ Подключение к базе данных установлено")
        except Exception as e:
            print(f"❌ Ошибка подключения к БД: {e}")
            raise
    
    def get_synonyms(self, term: str) -> List[str]:
        """Получение синонимов для термина"""
        try:
            cursor = self.connection.cursor()
            
            # Ищем синонимы в обеих таблицах
            cursor.execute("""
                SELECT DISTINCT synonym FROM technical_synonyms 
                WHERE main_term = ? OR synonym = ?
                UNION
                SELECT DISTINCT synonym FROM synonyms 
                WHERE term = ? OR synonym = ?
            """, (term, term, term, term))
            
            synonyms = [row[0] for row in cursor.fetchall()]
            return synonyms
            
        except Exception as e:
            print(f"❌ Ошибка получения синонимов: {e}")
            return []
    
    def expand_query_with_synonyms(self, query: str) -> Dict[str, List[str]]:
        """Расширение запроса синонимами"""
        try:
            # Разбиваем запрос на слова
            words = re.findall(r'\b\w+\b', query.lower())
            
            expanded_terms = {}
            
            for word in words:
                if len(word) > 2:  # Игнорируем короткие слова
                    synonyms = self.get_synonyms(word)
                    if synonyms:
                        expanded_terms[word] = synonyms
            
            return expanded_terms
            
        except Exception as e:
            print(f"❌ Ошибка расширения запроса: {e}")
            return {}
    
    def search_documents_with_synonyms(self, query: str, limit: int = 10) -> List[Dict]:
        """Поиск документов с использованием синонимов"""
        try:
            cursor = self.connection.cursor()
            
            # Расширяем запрос синонимами
            expanded_terms = self.expand_query_with_synonyms(query)
            
            # Создаем поисковые термины
            search_terms = [query]
            for term, synonyms in expanded_terms.items():
                search_terms.extend(synonyms)
            
            # Удаляем дубликаты
            search_terms = list(set(search_terms))
            
            # Создаем SQL запрос с OR условиями
            placeholders = ' OR '.join(['content LIKE ?' for _ in search_terms])
            sql_query = f"""
                SELECT DISTINCT 
                    id, file_name, content, category, tags, difficulty_level,
                    CASE 
                        WHEN content LIKE ? THEN 3
                        WHEN content LIKE ? THEN 2
                        ELSE 1
                    END as relevance_score
                FROM documents 
                WHERE {placeholders}
                ORDER BY relevance_score DESC, id DESC
                LIMIT ?
            """
            
            # Подготавливаем параметры
            params = []
            for term in search_terms:
                params.append(f'%{term}%')
            
            # Добавляем основные термины для релевантности
            params.append(f'%{query}%')
            params.append(f'%{query}%')
            params.append(limit)
            
            cursor.execute(sql_query, params)
            results = cursor.fetchall()
            
            # Форматируем результаты
            formatted_results = []
            for row in results:
                formatted_results.append({
                    'id': row[0],
                    'file_name': row[1],
                    'content': row[2][:500] + '...' if len(row[2]) > 500 else row[2],
                    'category': row[3],
                    'tags': row[4],
                    'difficulty_level': row[5],
                    'relevance_score': row[6],
                    'matched_terms': self.find_matched_terms(row[2], search_terms)
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ Ошибка поиска документов: {e}")
            return []
    
    def find_matched_terms(self, content: str, search_terms: List[str]) -> List[str]:
        """Поиск совпавших терминов в контенте"""
        matched = []
        content_lower = content.lower()
        
        for term in search_terms:
            if term.lower() in content_lower:
                matched.append(term)
        
        return matched
    
    def get_category_suggestions(self, query: str) -> List[str]:
        """Получение предложений по категориям"""
        try:
            cursor = self.connection.cursor()
            
            # Ищем категории, связанные с запросом
            cursor.execute("""
                SELECT DISTINCT category, COUNT(*) as count
                FROM technical_synonyms 
                WHERE main_term LIKE ? OR synonym LIKE ?
                GROUP BY category
                ORDER BY count DESC
                LIMIT 5
            """, (f'%{query}%', f'%{query}%'))
            
            categories = [row[0] for row in cursor.fetchall()]
            return categories
            
        except Exception as e:
            print(f"❌ Ошибка получения категорий: {e}")
            return []
    
    def get_related_terms(self, term: str) -> List[str]:
        """Получение связанных терминов"""
        try:
            cursor = self.connection.cursor()
            
            # Ищем термины в той же категории
            cursor.execute("""
                SELECT DISTINCT main_term
                FROM technical_synonyms 
                WHERE category = (
                    SELECT category FROM technical_synonyms 
                    WHERE main_term = ? OR synonym = ? 
                    LIMIT 1
                )
                AND main_term != ?
                LIMIT 10
            """, (term, term, term))
            
            related = [row[0] for row in cursor.fetchall()]
            return related
            
        except Exception as e:
            print(f"❌ Ошибка получения связанных терминов: {e}")
            return []
    
    def analyze_query_complexity(self, query: str) -> Dict[str, any]:
        """Анализ сложности запроса"""
        try:
            words = re.findall(r'\b\w+\b', query.lower())
            
            # Получаем синонимы для анализа
            expanded_terms = self.expand_query_with_synonyms(query)
            
            analysis = {
                'word_count': len(words),
                'unique_terms': len(set(words)),
                'synonym_coverage': len(expanded_terms),
                'total_synonyms': sum(len(syns) for syns in expanded_terms.values()),
                'complexity_score': 0,
                'suggested_categories': self.get_category_suggestions(query)
            }
            
            # Вычисляем оценку сложности
            if analysis['word_count'] > 5:
                analysis['complexity_score'] += 2
            if analysis['synonym_coverage'] > 3:
                analysis['complexity_score'] += 2
            if analysis['total_synonyms'] > 10:
                analysis['complexity_score'] += 1
            
            return analysis
            
        except Exception as e:
            print(f"❌ Ошибка анализа запроса: {e}")
            return {}
    
    def test_search_functionality(self):
        """Тестирование функциональности поиска"""
        print("\n🧪 ТЕСТИРОВАНИЕ ПОИСКОВОЙ ФУНКЦИОНАЛЬНОСТИ")
        print("=" * 50)
        
        test_queries = [
            "ПИД регулятор",
            "автоматизация производства",
            "электротехника",
            "программирование python",
            "датчики и сенсоры",
            "система управления"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Тестирование запроса: '{query}'")
            
            # Анализ запроса
            analysis = self.analyze_query_complexity(query)
            print(f"  📊 Анализ: {analysis['word_count']} слов, {analysis['synonym_coverage']} терминов с синонимами")
            
            # Расширение синонимами
            expanded = self.expand_query_with_synonyms(query)
            if expanded:
                print(f"  🔗 Найдены синонимы:")
                for term, synonyms in expanded.items():
                    print(f"    • {term}: {', '.join(synonyms[:3])}")
            
            # Поиск документов
            results = self.search_documents_with_synonyms(query, limit=3)
            print(f"  📄 Найдено документов: {len(results)}")
            
            for i, result in enumerate(results, 1):
                print(f"    {i}. {result['file_name']} (релевантность: {result['relevance_score']})")
                if result['matched_terms']:
                    print(f"       Совпадения: {', '.join(result['matched_terms'][:3])}")
            
            # Предложения по категориям
            categories = self.get_category_suggestions(query)
            if categories:
                print(f"  📋 Предлагаемые категории: {', '.join(categories)}")
    
    def close_connection(self):
        """Закрытие соединения с базой данных"""
        if self.connection:
            self.connection.close()
            print("✅ Соединение с БД закрыто")

def main():
    """Основная функция"""
    print("🚀 ТЕСТИРОВАНИЕ УЛУЧШЕННОГО ПОИСКА С СЛОВАРЕМ")
    print("=" * 60)
    
    searcher = EnhancedSearchWithVocabulary()
    
    try:
        # Тестируем функциональность
        searcher.test_search_functionality()
        
        print("\n🎉 ТЕСТИРОВАНИЕ ЗАВЕРШЕНО!")
        print(f"📅 Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
    finally:
        searcher.close_connection()

if __name__ == "__main__":
    main()






















