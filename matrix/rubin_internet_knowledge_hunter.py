#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Rubin AI - Охотник за знаниями в интернете
Автоматически ищет и скачивает информацию по контексту для пополнения базы знаний
"""

import requests
import sqlite3
import json
import os
import datetime
import time
import re
from urllib.parse import quote_plus, urljoin, urlparse
from bs4 import BeautifulSoup
from typing import Dict, List, Any, Optional, Tuple
import hashlib

class RubinInternetKnowledgeHunter:
    """Класс для поиска и скачивания знаний из интернета"""
    
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
        
        # Поисковые системы
        self.search_engines = {
            'google': 'https://www.google.com/search?q={}',
            'bing': 'https://www.bing.com/search?q={}',
            'duckduckgo': 'https://duckduckgo.com/?q={}',
            'yandex': 'https://yandex.ru/search/?text={}'
        }
        
        # Кэш для избежания повторных запросов
        self.cache = {}
        self.cache_file = "search_cache.json"
        self.load_cache()
    
    def load_cache(self):
        """Загружает кэш поиска"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
        except Exception as e:
            print(f"Ошибка загрузки кэша: {e}")
            self.cache = {}
    
    def save_cache(self):
        """Сохраняет кэш поиска"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Ошибка сохранения кэша: {e}")
    
    def search_internet(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Ищет информацию в интернете"""
        # Проверяем кэш
        cache_key = hashlib.md5(query.encode()).hexdigest()
        if cache_key in self.cache:
            print(f"📋 Используем кэшированные результаты для: {query}")
            return self.cache[cache_key]
        
        print(f"🔍 Поиск в интернете: {query}")
        results = []
        
        # Пробуем разные поисковики
        for engine_name, search_url in self.search_engines.items():
            try:
                print(f"🔄 Поиск через {engine_name}")
                encoded_query = quote_plus(query)
                url = search_url.format(encoded_query)
                
                response = self.session.get(url, timeout=10)
                if response.status_code == 200:
                    search_results = self.parse_search_results(response.text, engine_name)
                    results.extend(search_results)
                    
                    if len(results) >= max_results:
                        break
                        
                time.sleep(1)  # Пауза между запросами
                
            except Exception as e:
                print(f"❌ Ошибка поиска через {engine_name}: {e}")
                continue
        
        # Ограничиваем количество результатов
        results = results[:max_results]
        
        # Сохраняем в кэш
        self.cache[cache_key] = results
        self.save_cache()
        
        return results
    
    def parse_search_results(self, html: str, engine: str) -> List[Dict[str, Any]]:
        """Парсит результаты поиска"""
        results = []
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            if engine == 'google':
                results = self.parse_google_results(soup)
            elif engine == 'bing':
                results = self.parse_bing_results(soup)
            elif engine == 'duckduckgo':
                results = self.parse_duckduckgo_results(soup)
            elif engine == 'yandex':
                results = self.parse_yandex_results(soup)
                
        except Exception as e:
            print(f"❌ Ошибка парсинга результатов {engine}: {e}")
        
        return results
    
    def parse_google_results(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Парсит результаты Google"""
        results = []
        try:
            # Ищем результаты поиска
            search_results = soup.find_all('div', class_='g')
            
            for result in search_results:
                try:
                    # Заголовок
                    title_elem = result.find('h3')
                    title = title_elem.get_text() if title_elem else "Без заголовка"
                    
                    # Ссылка
                    link_elem = result.find('a')
                    url = link_elem.get('href') if link_elem else ""
                    
                    # Описание
                    desc_elem = result.find('span', class_='aCOpRe')
                    if not desc_elem:
                        desc_elem = result.find('div', class_='VwiC3b')
                    snippet = desc_elem.get_text() if desc_elem else ""
                    
                    if url and title:
                        results.append({
                            'title': title,
                            'url': url,
                            'snippet': snippet,
                            'source': 'google',
                            'timestamp': datetime.datetime.now().isoformat()
                        })
                        
                except Exception as e:
                    continue
                    
        except Exception as e:
            print(f"❌ Ошибка парсинга Google: {e}")
        
        return results
    
    def parse_bing_results(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Парсит результаты Bing"""
        results = []
        try:
            search_results = soup.find_all('li', class_='b_algo')
            
            for result in search_results:
                try:
                    title_elem = result.find('h2')
                    title = title_elem.get_text() if title_elem else "Без заголовка"
                    
                    link_elem = result.find('a')
                    url = link_elem.get('href') if link_elem else ""
                    
                    desc_elem = result.find('p')
                    snippet = desc_elem.get_text() if desc_elem else ""
                    
                    if url and title:
                        results.append({
                            'title': title,
                            'url': url,
                            'snippet': snippet,
                            'source': 'bing',
                            'timestamp': datetime.datetime.now().isoformat()
                        })
                        
                except Exception as e:
                    continue
                    
        except Exception as e:
            print(f"❌ Ошибка парсинга Bing: {e}")
        
        return results
    
    def parse_duckduckgo_results(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Парсит результаты DuckDuckGo"""
        results = []
        try:
            search_results = soup.find_all('div', class_='result')
            
            for result in search_results:
                try:
                    title_elem = result.find('a', class_='result__a')
                    title = title_elem.get_text() if title_elem else "Без заголовка"
                    url = title_elem.get('href') if title_elem else ""
                    
                    desc_elem = result.find('a', class_='result__snippet')
                    snippet = desc_elem.get_text() if desc_elem else ""
                    
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
            print(f"❌ Ошибка парсинга DuckDuckGo: {e}")
        
        return results
    
    def parse_yandex_results(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Парсит результаты Yandex"""
        results = []
        try:
            search_results = soup.find_all('div', class_='serp-item')
            
            for result in search_results:
                try:
                    title_elem = result.find('h2')
                    if not title_elem:
                        title_elem = result.find('a', class_='link')
                    title = title_elem.get_text() if title_elem else "Без заголовка"
                    
                    link_elem = result.find('a', class_='link')
                    url = link_elem.get('href') if link_elem else ""
                    
                    desc_elem = result.find('div', class_='text-container')
                    snippet = desc_elem.get_text() if desc_elem else ""
                    
                    if url and title:
                        results.append({
                            'title': title,
                            'url': url,
                            'snippet': snippet,
                            'source': 'yandex',
                            'timestamp': datetime.datetime.now().isoformat()
                        })
                        
                except Exception as e:
                    continue
                    
        except Exception as e:
            print(f"❌ Ошибка парсинга Yandex: {e}")
        
        return results
    
    def download_page_content(self, url: str) -> Optional[str]:
        """Скачивает содержимое страницы"""
        try:
            print(f"📥 Скачиваем страницу: {url}")
            
            response = self.session.get(url, timeout=15)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Удаляем скрипты и стили
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Извлекаем текст
                text = soup.get_text()
                
                # Очищаем текст
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                
                return text[:5000]  # Ограничиваем размер
                
        except Exception as e:
            print(f"❌ Ошибка скачивания {url}: {e}")
        
        return None
    
    def extract_knowledge_from_content(self, content: str, url: str, title: str) -> Optional[Dict[str, Any]]:
        """Извлекает знания из содержимого страницы"""
        try:
            if not content or len(content) < 100:
                return None
            
            # Определяем категорию по содержимому
            category = self.detect_category(content, title)
            
            # Извлекаем ключевые слова
            keywords = self.extract_keywords(content, title)
            
            # Создаем структурированное знание
            knowledge = {
                'title': title,
                'content': content,
                'category': category,
                'tags': ', '.join(keywords[:5]),  # Первые 5 ключевых слов
                'keywords': ', '.join(keywords),
                'source_url': url,
                'source_type': 'internet',
                'created_at': datetime.datetime.now().isoformat(),
                'usage_count': 0,
                'relevance_score': 0.8
            }
            
            return knowledge
            
        except Exception as e:
            print(f"❌ Ошибка извлечения знаний: {e}")
            return None
    
    def detect_category(self, content: str, title: str) -> str:
        """Определяет категорию по содержимому"""
        content_lower = content.lower()
        title_lower = title.lower()
        
        # Математика
        if any(word in content_lower for word in ['математика', 'формула', 'уравнение', 'интеграл', 'производная', 'алгебра', 'геометрия']):
            return 'mathematics'
        
        # Физика
        if any(word in content_lower for word in ['физика', 'механика', 'электричество', 'магнетизм', 'оптика', 'термодинамика']):
            return 'physics'
        
        # Программирование
        if any(word in content_lower for word in ['программирование', 'код', 'алгоритм', 'python', 'c++', 'java', 'javascript']):
            return 'programming'
        
        # Автоматизация
        if any(word in content_lower for word in ['автоматизация', 'plc', 'pmac', 'контроллер', 'датчик', 'привод']):
            return 'automation'
        
        # Электроника
        if any(word in content_lower for word in ['электроника', 'схема', 'транзистор', 'резистор', 'конденсатор']):
            return 'electronics'
        
        return 'general'
    
    def extract_keywords(self, content: str, title: str) -> List[str]:
        """Извлекает ключевые слова из текста"""
        # Объединяем заголовок и содержимое
        text = f"{title} {content}".lower()
        
        # Удаляем знаки препинания
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Разбиваем на слова
        words = text.split()
        
        # Фильтруем короткие слова и стоп-слова
        stop_words = {'и', 'в', 'на', 'с', 'по', 'для', 'от', 'до', 'из', 'к', 'о', 'у', 'за', 'под', 'над', 'при', 'через', 'без', 'между', 'среди', 'около', 'вокруг', 'внутри', 'вне', 'перед', 'после', 'во', 'со', 'об', 'про', 'что', 'как', 'где', 'когда', 'почему', 'зачем', 'который', 'которая', 'которое', 'которые', 'это', 'этот', 'эта', 'это', 'эти', 'тот', 'та', 'то', 'те', 'он', 'она', 'оно', 'они', 'мы', 'вы', 'я', 'ты', 'он', 'она', 'оно', 'они', 'мой', 'моя', 'мое', 'мои', 'твой', 'твоя', 'твое', 'твои', 'его', 'ее', 'их', 'наш', 'наша', 'наше', 'наши', 'ваш', 'ваша', 'ваше', 'ваши'}
        
        keywords = [word for word in words if len(word) > 3 and word not in stop_words]
        
        # Подсчитываем частоту
        word_count = {}
        for word in keywords:
            word_count[word] = word_count.get(word, 0) + 1
        
        # Сортируем по частоте
        sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        
        return [word for word, count in sorted_words[:10]]  # Топ 10 ключевых слов
    
    def hunt_knowledge(self, topic: str, max_pages: int = 3) -> List[Dict[str, Any]]:
        """Основная функция охоты за знаниями"""
        print(f"🎯 Охота за знаниями по теме: {topic}")
        
        # Ищем в интернете
        search_results = self.search_internet(topic, max_results=max_pages * 2)
        
        if not search_results:
            print("❌ Не найдено результатов поиска")
            return []
        
        print(f"📋 Найдено {len(search_results)} результатов поиска")
        
        # Скачиваем и обрабатываем страницы
        knowledge_items = []
        
        for i, result in enumerate(search_results[:max_pages]):
            try:
                print(f"📥 Обрабатываем {i+1}/{min(max_pages, len(search_results))}: {result['title']}")
                
                # Скачиваем содержимое
                content = self.download_page_content(result['url'])
                
                if content:
                    # Извлекаем знания
                    knowledge = self.extract_knowledge_from_content(
                        content, result['url'], result['title']
                    )
                    
                    if knowledge:
                        knowledge_items.append(knowledge)
                        print(f"✅ Извлечены знания: {knowledge['title']}")
                    else:
                        print(f"⚠️ Не удалось извлечь знания из: {result['title']}")
                
                time.sleep(2)  # Пауза между запросами
                
            except Exception as e:
                print(f"❌ Ошибка обработки {result['title']}: {e}")
                continue
        
        print(f"🎉 Извлечено {len(knowledge_items)} знаний")
        return knowledge_items
    
    def save_knowledge_to_database(self, knowledge_items: List[Dict[str, Any]]) -> int:
        """Сохраняет знания в базу данных"""
        if not knowledge_items:
            return 0
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            saved_count = 0
            
            for knowledge in knowledge_items:
                try:
                    # Проверяем, не существует ли уже такая запись
                    cursor.execute("""
                        SELECT COUNT(*) FROM knowledge_base 
                        WHERE title = ? AND source_url = ?
                    """, (knowledge['title'], knowledge['source_url']))
                    
                    if cursor.fetchone()[0] == 0:
                        # Вставляем новую запись
                        cursor.execute("""
                            INSERT INTO knowledge_base 
                            (title, content, category, tags, keywords, created_at, usage_count, relevance_score)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            knowledge['title'],
                            knowledge['content'],
                            knowledge['category'],
                            knowledge['tags'],
                            knowledge['keywords'],
                            knowledge['created_at'],
                            knowledge['usage_count'],
                            knowledge['relevance_score']
                        ))
                        saved_count += 1
                        print(f"💾 Сохранено: {knowledge['title']}")
                    else:
                        print(f"⚠️ Уже существует: {knowledge['title']}")
                        
                except Exception as e:
                    print(f"❌ Ошибка сохранения {knowledge['title']}: {e}")
                    continue
            
            conn.commit()
            conn.close()
            
            return saved_count
            
        except Exception as e:
            print(f"❌ Ошибка работы с базой данных: {e}")
            return 0
    
    def auto_hunt_knowledge(self, topics: List[str]) -> Dict[str, int]:
        """Автоматически охотится за знаниями по списку тем"""
        print("🚀 АВТОМАТИЧЕСКАЯ ОХОТА ЗА ЗНАНИЯМИ")
        print("=" * 50)
        
        results = {}
        
        for topic in topics:
            print(f"\n🎯 Тема: {topic}")
            print("-" * 30)
            
            # Охотимся за знаниями
            knowledge_items = self.hunt_knowledge(topic, max_pages=2)
            
            if knowledge_items:
                # Сохраняем в базу
                saved_count = self.save_knowledge_to_database(knowledge_items)
                results[topic] = saved_count
                print(f"✅ Сохранено {saved_count} знаний по теме '{topic}'")
            else:
                results[topic] = 0
                print(f"❌ Не найдено знаний по теме '{topic}'")
            
            time.sleep(3)  # Пауза между темами
        
        return results

def main():
    """Основная функция"""
    print("🌐 SMART RUBIN AI - ОХОТНИК ЗА ЗНАНИЯМИ В ИНТЕРНЕТЕ")
    print("=" * 60)
    
    # Создаем охотника за знаниями
    hunter = RubinInternetKnowledgeHunter()
    
    # Список тем для поиска
    topics = [
        "программирование Python основы",
        "математика производные интегралы",
        "физика законы Ньютона механика",
        "автоматизация PLC программирование",
        "электроника схемы транзисторы"
    ]
    
    # Запускаем автоматическую охоту
    results = hunter.auto_hunt_knowledge(topics)
    
    # Показываем результаты
    print("\n🎯 ИТОГОВЫЕ РЕЗУЛЬТАТЫ:")
    print("=" * 30)
    
    total_saved = 0
    for topic, count in results.items():
        print(f"📚 {topic}: {count} знаний")
        total_saved += count
    
    print(f"\n🎉 ВСЕГО СОХРАНЕНО: {total_saved} знаний")
    
    if total_saved > 0:
        print("\n✅ База знаний Smart Rubin AI успешно пополнена!")
        print("🧠 Теперь Rubin может давать более точные ответы по найденным темам!")
    else:
        print("\n⚠️ Не удалось найти и сохранить новые знания")
        print("💡 Попробуйте изменить поисковые запросы или проверить интернет-соединение")

if __name__ == "__main__":
    main()
