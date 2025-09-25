#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Расширенные функции Rubin AI
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import sqlite3
import hashlib

class RubinEnhancements:
    """Класс расширенных возможностей Rubin AI"""
    
    def __init__(self):
        self.cache_db = "rubin_cache.db"
        self.user_db = "rubin_users.db"
        self.analytics_db = "rubin_analytics.db"
        self.init_databases()
    
    def init_databases(self):
        """Инициализация баз данных"""
        # Кэш ответов
        with sqlite3.connect(self.cache_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_hash TEXT UNIQUE,
                    response TEXT,
                    timestamp DATETIME,
                    usage_count INTEGER DEFAULT 1
                )
            """)
        
        # Пользователи
        with sqlite3.connect(self.user_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE,
                    preferences TEXT,
                    created_at DATETIME,
                    last_active DATETIME
                )
            """)
        
        # Аналитика
        with sqlite3.connect(self.analytics_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    endpoint TEXT,
                    response_time REAL,
                    success BOOLEAN,
                    user_agent TEXT,
                    timestamp DATETIME
                )
            """)
    
    def cache_response(self, query: str, response: str) -> str:
        """Кэширование ответа"""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        
        with sqlite3.connect(self.cache_db) as conn:
            # Проверяем, есть ли уже такой запрос
            cursor = conn.execute(
                "SELECT usage_count FROM cache WHERE query_hash = ?", 
                (query_hash,)
            )
            result = cursor.fetchone()
            
            if result:
                # Увеличиваем счетчик использования
                conn.execute(
                    "UPDATE cache SET usage_count = usage_count + 1 WHERE query_hash = ?",
                    (query_hash,)
                )
            else:
                # Добавляем новый ответ в кэш
                conn.execute(
                    "INSERT INTO cache (query_hash, response, timestamp) VALUES (?, ?, ?)",
                    (query_hash, response, datetime.now())
                )
        
        return query_hash
    
    def get_cached_response(self, query: str) -> Optional[str]:
        """Получение кэшированного ответа"""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        
        with sqlite3.connect(self.cache_db) as conn:
            cursor = conn.execute(
                "SELECT response FROM cache WHERE query_hash = ?", 
                (query_hash,)
            )
            result = cursor.fetchone()
            return result[0] if result else None
    
    def log_analytics(self, endpoint: str, response_time: float, success: bool, user_agent: str):
        """Логирование аналитики"""
        with sqlite3.connect(self.analytics_db) as conn:
            conn.execute(
                """INSERT INTO analytics (endpoint, response_time, success, user_agent, timestamp) 
                   VALUES (?, ?, ?, ?, ?)""",
                (endpoint, response_time, success, user_agent, datetime.now())
            )
    
    def get_analytics_summary(self, days: int = 7) -> Dict[str, Any]:
        """Получение сводки аналитики"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.analytics_db) as conn:
            # Общая статистика
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_requests,
                    AVG(response_time) as avg_response_time,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_requests
                FROM analytics 
                WHERE timestamp >= ?
            """, (cutoff_date,))
            
            stats = cursor.fetchone()
            
            # Топ эндпоинты
            cursor = conn.execute("""
                SELECT endpoint, COUNT(*) as count
                FROM analytics 
                WHERE timestamp >= ?
                GROUP BY endpoint
                ORDER BY count DESC
                LIMIT 10
            """, (cutoff_date,))
            
            top_endpoints = cursor.fetchall()
            
            return {
                "period_days": days,
                "total_requests": stats[0],
                "average_response_time": round(stats[1], 3) if stats[1] else 0,
                "success_rate": round(stats[2] / stats[0] * 100, 2) if stats[0] > 0 else 0,
                "top_endpoints": [{"endpoint": ep[0], "count": ep[1]} for ep in top_endpoints]
            }
    
    def personalize_response(self, user_id: str, base_response: str, query: str) -> str:
        """Персонализация ответа"""
        # Получаем предпочтения пользователя
        with sqlite3.connect(self.user_db) as conn:
            cursor = conn.execute(
                "SELECT preferences FROM users WHERE username = ?", 
                (user_id,)
            )
            result = cursor.fetchone()
            
            if result:
                preferences = json.loads(result[0])
                
                # Применяем персонализацию
                if preferences.get("language") == "ru":
                    # Добавляем русский контекст
                    base_response = f"🇷🇺 {base_response}"
                
                if preferences.get("style") == "formal":
                    # Формальный стиль
                    base_response = base_response.replace("ты", "Вы")
                
                if preferences.get("detail_level") == "brief":
                    # Краткий ответ
                    if len(base_response) > 200:
                        base_response = base_response[:200] + "..."
        
        return base_response
    
    def create_user(self, username: str, preferences: Dict[str, Any]) -> bool:
        """Создание пользователя"""
        try:
            with sqlite3.connect(self.user_db) as conn:
                conn.execute(
                    """INSERT INTO users (username, preferences, created_at, last_active) 
                       VALUES (?, ?, ?, ?)""",
                    (username, json.dumps(preferences), datetime.now(), datetime.now())
                )
            return True
        except sqlite3.IntegrityError:
            return False  # Пользователь уже существует
    
    def update_user_preferences(self, username: str, preferences: Dict[str, Any]) -> bool:
        """Обновление предпочтений пользователя"""
        try:
            with sqlite3.connect(self.user_db) as conn:
                conn.execute(
                    "UPDATE users SET preferences = ?, last_active = ? WHERE username = ?",
                    (json.dumps(preferences), datetime.now(), username)
                )
            return True
        except Exception:
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Статистика кэша"""
        with sqlite3.connect(self.cache_db) as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_entries,
                    SUM(usage_count) as total_usage,
                    AVG(usage_count) as avg_usage,
                    MAX(usage_count) as max_usage
                FROM cache
            """)
            
            stats = cursor.fetchone()
            
            # Самые популярные запросы
            cursor = conn.execute("""
                SELECT response, usage_count
                FROM cache
                ORDER BY usage_count DESC
                LIMIT 5
            """)
            
            popular = cursor.fetchall()
            
            return {
                "total_entries": stats[0],
                "total_usage": stats[1],
                "average_usage": round(stats[2], 2) if stats[2] else 0,
                "max_usage": stats[3],
                "popular_queries": [{"response": p[0][:100] + "...", "usage": p[1]} for p in popular]
            }
    
    def cleanup_old_cache(self, days: int = 30):
        """Очистка старого кэша"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.cache_db) as conn:
            cursor = conn.execute(
                "DELETE FROM cache WHERE timestamp < ? AND usage_count < 3",
                (cutoff_date,)
            )
            return cursor.rowcount

# Пример использования
if __name__ == "__main__":
    enhancer = RubinEnhancements()
    
    # Создаем пользователя
    enhancer.create_user("test_user", {
        "language": "ru",
        "style": "informal",
        "detail_level": "detailed"
    })
    
    # Кэшируем ответ
    query = "Что такое машинное обучение?"
    response = "Машинное обучение — это раздел искусственного интеллекта..."
    cache_id = enhancer.cache_response(query, response)
    
    # Получаем кэшированный ответ
    cached = enhancer.get_cached_response(query)
    print(f"Кэшированный ответ: {cached}")
    
    # Логируем аналитику
    enhancer.log_analytics("/api/v2/chat", 1.5, True, "Mozilla/5.0")
    
    # Получаем статистику
    analytics = enhancer.get_analytics_summary()
    cache_stats = enhancer.get_cache_stats()
    
    print(f"Аналитика: {analytics}")
    print(f"Статистика кэша: {cache_stats}")











