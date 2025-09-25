#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Rubin AI - Контекстуальный поиск в интернете
Автоматически определяет, когда нужен поиск в интернете, и ищет релевантную информацию
"""

import requests
import sqlite3
import json
import os
import datetime
import time
import re
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
from typing import Dict, List, Any, Optional, Tuple
import hashlib

class RubinContextualInternetSearch:
    """Класс для контекстуального поиска в интернете"""
    
    def __init__(self, db_path: str = "rubin_knowledge_base.db"):
        self.db_path = db_path
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
        
        # Ключевые слова, указывающие на необходимость поиска в интернете
        self.internet_search_triggers = {
            'новые_технологии': ['новый', 'современный', 'последний', 'актуальный', 'тренд', 'инновация'],
            'конкретные_вопросы': ['как настроить', 'как подключить', 'как использовать', 'как работает'],
            'технические_спецификации': ['характеристики', 'параметры', 'спецификация', 'технические данные'],
            'обновления': ['обновление', 'версия', 'патч', 'исправление', 'новые функции'],
            'сравнения': ['сравнить', 'разница', 'отличия', 'лучше', 'хуже', 'против'],
            'примеры_кода': ['пример кода', 'код', 'программирование', 'синтаксис', 'функция'],
            'документация': ['документация', 'руководство', 'инструкция', 'мануал', 'справка']
        }
        
        # Кэш для поисковых запросов
        self.search_cache = {}
        self.cache_file = "contextual_search_cache.json"
        self.load_cache()
    
    def load_cache(self):
        """Загружает кэш поиска"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.search_cache = json.load(f)
        except Exception as e:
            print(f"Ошибка загрузки кэша: {e}")
            self.search_cache = {}
    
    def save_cache(self):
        """Сохраняет кэш поиска"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.search_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Ошибка сохранения кэша: {e}")
    
    def needs_internet_search(self, message: str) -> Tuple[bool, str, str]:
        """Определяет, нужен ли поиск в интернете"""
        message_lower = message.lower()
        
        # Проверяем триггеры
        for category, triggers in self.internet_search_triggers.items():
            for trigger in triggers:
                if trigger in message_lower:
                    # Определяем тип поиска
                    search_type = self.determine_search_type(message, category)
                    return True, category, search_type
        
        # Проверяем, есть ли ответ в локальной базе знаний
        if self.has_local_answer(message):
            return False, "local", "local"
        
        # Если нет локального ответа, ищем в интернете
        return True, "general", "general"
    
    def determine_search_type(self, message: str, category: str) -> str:
        """Определяет тип поиска"""
        message_lower = message.lower()
        
        if category == 'новые_технологии':
            return "technology"
        elif category == 'конкретные_вопросы':
            return "howto"
        elif category == 'технические_спецификации':
            return "specifications"
        elif category == 'обновления':
            return "updates"
        elif category == 'сравнения':
            return "comparison"
        elif category == 'примеры_кода':
            return "code_examples"
        elif category == 'документация':
            return "documentation"
        else:
            return "general"
    
    def has_local_answer(self, message: str) -> bool:
        """Проверяет, есть ли ответ в локальной базе знаний"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Ищем по ключевым словам
            words = message.lower().split()
            search_terms = [word for word in words if len(word) > 3]
            
            if not search_terms:
                return False
            
            # Создаем поисковый запрос
            search_query = " OR ".join([f"title LIKE '%{term}%' OR content LIKE '%{term}%' OR keywords LIKE '%{term}%'" for term in search_terms])
            
            cursor.execute(f"""
                SELECT COUNT(*) FROM knowledge_base 
                WHERE {search_query}
            """)
            
            count = cursor.fetchone()[0]
            conn.close()
            
            return count > 0
            
        except Exception as e:
            print(f"Ошибка проверки локальной базы: {e}")
            return False
    
    def generate_search_query(self, message: str, search_type: str) -> str:
        """Генерирует оптимизированный поисковый запрос"""
        message_lower = message.lower()
        
        # Базовый запрос
        base_query = message
        
        # Оптимизируем запрос в зависимости от типа
        if search_type == "technology":
            base_query += " новейшие технологии 2024"
        elif search_type == "howto":
            base_query += " инструкция руководство"
        elif search_type == "specifications":
            base_query += " технические характеристики параметры"
        elif search_type == "updates":
            base_query += " обновления новости"
        elif search_type == "comparison":
            base_query += " сравнение отличия"
        elif search_type == "code_examples":
            base_query += " примеры кода программирование"
        elif search_type == "documentation":
            base_query += " документация официальная"
        
        return base_query
    
    def search_internet(self, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """Ищет информацию в интернете"""
        # Проверяем кэш
        cache_key = hashlib.md5(query.encode()).hexdigest()
        if cache_key in self.search_cache:
            print(f"📋 Используем кэшированные результаты")
            return self.search_cache[cache_key]
        
        print(f"🔍 Поиск в интернете: {query}")
        
        # Используем DuckDuckGo для поиска
        try:
            encoded_query = quote_plus(query)
            search_url = f"https://duckduckgo.com/html/?q={encoded_query}"
            
            response = self.session.get(search_url, timeout=10)
            if response.status_code == 200:
                results = self.parse_duckduckgo_results(response.text)
                
                # Сохраняем в кэш
                self.search_cache[cache_key] = results
                self.save_cache()
                
                return results[:max_results]
                
        except Exception as e:
            print(f"❌ Ошибка поиска: {e}")
        
        return []
    
    def parse_duckduckgo_results(self, html: str) -> List[Dict[str, Any]]:
        """Парсит результаты DuckDuckGo"""
        results = []
        try:
            soup = BeautifulSoup(html, 'html.parser')
            search_results = soup.find_all('div', class_='result')
            
            for result in search_results:
                try:
                    # Заголовок и ссылка
                    title_elem = result.find('a', class_='result__a')
                    if not title_elem:
                        continue
                    
                    title = title_elem.get_text().strip()
                    url = title_elem.get('href', '')
                    
                    # Описание
                    desc_elem = result.find('a', class_='result__snippet')
                    snippet = desc_elem.get_text().strip() if desc_elem else ""
                    
                    if url and title:
                        results.append({
                            'title': title,
                            'url': url,
                            'snippet': snippet,
                            'source': 'duckduckgo',
                            'timestamp': datetime.datetime.now().isoformat()
                        })
                        
                except Exception as e:
                    continue
                    
        except Exception as e:
            print(f"❌ Ошибка парсинга результатов: {e}")
        
        return results
    
    def download_and_analyze_content(self, url: str, title: str) -> Optional[Dict[str, Any]]:
        """Скачивает и анализирует содержимое страницы"""
        try:
            print(f"📥 Анализируем: {title}")
            
            response = self.session.get(url, timeout=15)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Удаляем скрипты и стили
                for script in soup(["script", "style", "nav", "footer", "header"]):
                    script.decompose()
                
                # Извлекаем основной контент
                content_selectors = [
                    'article', 'main', '.content', '.post-content', 
                    '.entry-content', '.article-content', 'div[role="main"]'
                ]
                
                content = ""
                for selector in content_selectors:
                    content_elem = soup.select_one(selector)
                    if content_elem:
                        content = content_elem.get_text()
                        break
                
                if not content:
                    content = soup.get_text()
                
                # Очищаем текст
                lines = (line.strip() for line in content.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                content = ' '.join(chunk for chunk in chunks if chunk)
                
                # Ограничиваем размер
                content = content[:3000]
                
                if len(content) > 200:
                    return {
                        'title': title,
                        'content': content,
                        'url': url,
                        'timestamp': datetime.datetime.now().isoformat()
                    }
                
        except Exception as e:
            print(f"❌ Ошибка анализа {url}: {e}")
        
        return None
    
    def process_user_message(self, message: str) -> Dict[str, Any]:
        """Обрабатывает сообщение пользователя с контекстуальным поиском"""
        print(f"💬 Обрабатываем сообщение: {message}")
        
        # Определяем, нужен ли поиск в интернете
        needs_search, category, search_type = self.needs_internet_search(message)
        
        if not needs_search:
            return {
                'needs_internet_search': False,
                'search_category': category,
                'search_type': search_type,
                'local_answer_available': True,
                'internet_results': [],
                'analyzed_content': None
            }
        
        print(f"🌐 Нужен поиск в интернете: {category} ({search_type})")
        
        # Генерируем поисковый запрос
        search_query = self.generate_search_query(message, search_type)
        
        # Ищем в интернете
        search_results = self.search_internet(search_query, max_results=2)
        
        if not search_results:
            return {
                'needs_internet_search': True,
                'search_category': category,
                'search_type': search_type,
                'local_answer_available': False,
                'internet_results': [],
                'analyzed_content': None
            }
        
        # Анализируем наиболее релевантный результат
        analyzed_content = None
        if search_results:
            best_result = search_results[0]
            analyzed_content = self.download_and_analyze_content(
                best_result['url'], best_result['title']
            )
        
        return {
            'needs_internet_search': True,
            'search_category': category,
            'search_type': search_type,
            'local_answer_available': False,
            'internet_results': search_results,
            'analyzed_content': analyzed_content
        }
    
    def save_knowledge_from_internet(self, analyzed_content: Dict[str, Any], original_message: str) -> bool:
        """Сохраняет знания, полученные из интернета"""
        if not analyzed_content:
            return False
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Определяем категорию
            category = self.detect_category(analyzed_content['content'], analyzed_content['title'])
            
            # Извлекаем ключевые слова
            keywords = self.extract_keywords(analyzed_content['content'], analyzed_content['title'])
            
            # Проверяем, не существует ли уже такая запись
            cursor.execute("""
                SELECT COUNT(*) FROM knowledge_base 
                WHERE title = ? AND source_url = ?
            """, (analyzed_content['title'], analyzed_content['url']))
            
            if cursor.fetchone()[0] == 0:
                # Вставляем новую запись
                cursor.execute("""
                    INSERT INTO knowledge_base 
                    (title, content, category, tags, keywords, created_at, usage_count, relevance_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    analyzed_content['title'],
                    analyzed_content['content'],
                    category,
                    ', '.join(keywords[:5]),
                    ', '.join(keywords),
                    analyzed_content['timestamp'],
                    0,
                    0.9
                ))
                
                conn.commit()
                conn.close()
                
                print(f"💾 Сохранены знания: {analyzed_content['title']}")
                return True
            else:
                print(f"⚠️ Знания уже существуют: {analyzed_content['title']}")
                return False
                
        except Exception as e:
            print(f"❌ Ошибка сохранения знаний: {e}")
            return False
    
    def detect_category(self, content: str, title: str) -> str:
        """Определяет категорию по содержимому"""
        text = f"{title} {content}".lower()
        
        if any(word in text for word in ['математика', 'формула', 'уравнение', 'интеграл', 'производная']):
            return 'mathematics'
        elif any(word in text for word in ['физика', 'механика', 'электричество', 'магнетизм']):
            return 'physics'
        elif any(word in text for word in ['программирование', 'код', 'алгоритм', 'python', 'c++']):
            return 'programming'
        elif any(word in text for word in ['автоматизация', 'plc', 'pmac', 'контроллер']):
            return 'automation'
        elif any(word in text for word in ['электроника', 'схема', 'транзистор', 'резистор']):
            return 'electronics'
        else:
            return 'general'
    
    def extract_keywords(self, content: str, title: str) -> List[str]:
        """Извлекает ключевые слова"""
        text = f"{title} {content}".lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        
        stop_words = {'и', 'в', 'на', 'с', 'по', 'для', 'от', 'до', 'из', 'к', 'о', 'у', 'за', 'под', 'над', 'при', 'через', 'без', 'между', 'среди', 'около', 'вокруг', 'внутри', 'вне', 'перед', 'после', 'во', 'со', 'об', 'про', 'что', 'как', 'где', 'когда', 'почему', 'зачем', 'который', 'которая', 'которое', 'которые', 'это', 'этот', 'эта', 'это', 'эти', 'тот', 'та', 'то', 'те', 'он', 'она', 'оно', 'они', 'мы', 'вы', 'я', 'ты', 'он', 'она', 'оно', 'они', 'мой', 'моя', 'мое', 'мои', 'твой', 'твоя', 'твое', 'твои', 'его', 'ее', 'их', 'наш', 'наша', 'наше', 'наши', 'ваш', 'ваша', 'ваше', 'ваши'}
        
        keywords = [word for word in words if len(word) > 3 and word not in stop_words]
        
        word_count = {}
        for word in keywords:
            word_count[word] = word_count.get(word, 0) + 1
        
        sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_words[:10]]

def main():
    """Основная функция для тестирования"""
    print("🌐 SMART RUBIN AI - КОНТЕКСТУАЛЬНЫЙ ПОИСК В ИНТЕРНЕТЕ")
    print("=" * 60)
    
    # Создаем систему контекстуального поиска
    searcher = RubinContextualInternetSearch()
    
    # Тестовые сообщения
    test_messages = [
        "Как настроить Python для машинного обучения?",
        "Новые технологии в автоматизации 2024",
        "Сравнение PLC и PMAC контроллеров",
        "Что такое производная функции?",
        "Как подключить датчик температуры к Arduino?"
    ]
    
    for message in test_messages:
        print(f"\n💬 Тест: {message}")
        print("-" * 50)
        
        # Обрабатываем сообщение
        result = searcher.process_user_message(message)
        
        print(f"🔍 Нужен поиск в интернете: {result['needs_internet_search']}")
        print(f"📂 Категория: {result['search_category']}")
        print(f"🎯 Тип поиска: {result['search_type']}")
        print(f"📚 Локальный ответ доступен: {result['local_answer_available']}")
        
        if result['internet_results']:
            print(f"🌐 Найдено результатов: {len(result['internet_results'])}")
            for i, res in enumerate(result['internet_results'], 1):
                print(f"   {i}. {res['title']}")
        
        if result['analyzed_content']:
            print(f"📄 Проанализирован контент: {result['analyzed_content']['title']}")
            
            # Сохраняем знания
            if searcher.save_knowledge_from_internet(result['analyzed_content'], message):
                print("💾 Знания сохранены в базу!")
        
        time.sleep(2)  # Пауза между тестами

if __name__ == "__main__":
    main()
