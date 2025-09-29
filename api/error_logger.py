"""
Модуль автоматического логирования ошибок Rubin AI
Анализирует паттерны ошибок и предсказывает возможные сбои
"""

import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import hashlib
import re

class ErrorLogger:
    def __init__(self, db_path: str = "rubin_errors.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._init_database()
    
    def _init_database(self):
        """Инициализация базы данных для логирования ошибок"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Таблица ошибок
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS errors (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        error_type TEXT NOT NULL,
                        error_message TEXT NOT NULL,
                        module TEXT NOT NULL,
                        severity INTEGER DEFAULT 1,
                        context TEXT,
                        stack_trace TEXT,
                        error_hash TEXT UNIQUE,
                        resolved BOOLEAN DEFAULT FALSE,
                        resolution_notes TEXT
                    )
                ''')
                
                # Таблица паттернов ошибок
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS error_patterns (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        pattern_hash TEXT UNIQUE,
                        pattern_description TEXT,
                        frequency INTEGER DEFAULT 1,
                        first_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
                        last_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
                        severity_score REAL DEFAULT 0.0,
                        auto_resolution TEXT
                    )
                ''')
                
                # Таблица предсказаний
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        module TEXT NOT NULL,
                        prediction_type TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        predicted_time DATETIME,
                        actual_time DATETIME,
                        accuracy REAL
                    )
                ''')
                
                conn.commit()
                self.logger.info("✅ База данных ошибок инициализирована")
                
        except Exception as e:
            self.logger.error(f"❌ Ошибка инициализации базы данных: {e}")
    
    def log_error(self, error_type: str, error_message: str, module: str, 
                  severity: int = 1, context: str = "", stack_trace: str = ""):
        """Логирование ошибки с анализом паттернов"""
        try:
            # Создаем хеш ошибки для дедупликации
            error_hash = self._create_error_hash(error_type, error_message, module)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Проверяем, существует ли уже такая ошибка
                cursor.execute('''
                    SELECT id, frequency FROM error_patterns 
                    WHERE pattern_hash = ?
                ''', (error_hash,))
                
                result = cursor.fetchone()
                
                if result:
                    # Обновляем частоту и время последнего появления
                    pattern_id, frequency = result
                    cursor.execute('''
                        UPDATE error_patterns 
                        SET frequency = frequency + 1, last_seen = CURRENT_TIMESTAMP
                        WHERE id = ?
                    ''', (pattern_id,))
                else:
                    # Создаем новый паттерн
                    cursor.execute('''
                        INSERT INTO error_patterns 
                        (pattern_hash, pattern_description, frequency, severity_score)
                        VALUES (?, ?, 1, ?)
                    ''', (error_hash, f"{error_type} in {module}", severity))
                
                # Логируем саму ошибку
                cursor.execute('''
                    INSERT INTO errors 
                    (error_type, error_message, module, severity, context, stack_trace, error_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (error_type, error_message, module, severity, context, stack_trace, error_hash))
                
                conn.commit()
                
                # Анализируем паттерн для предсказания
                self._analyze_pattern(error_hash, module)
                
                self.logger.info(f"✅ Ошибка залогирована: {error_type} в {module}")
                
        except Exception as e:
            self.logger.error(f"❌ Ошибка при логировании: {e}")
    
    def _create_error_hash(self, error_type: str, error_message: str, module: str) -> str:
        """Создание хеша ошибки для дедупликации"""
        # Нормализуем сообщение об ошибке
        normalized_message = re.sub(r'\d+', 'N', error_message)
        normalized_message = re.sub(r'[a-f0-9]{8,}', 'HASH', normalized_message)
        
        content = f"{error_type}:{normalized_message}:{module}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _analyze_pattern(self, error_hash: str, module: str):
        """Анализ паттерна ошибки для предсказания"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Получаем статистику по модулю
                cursor.execute('''
                    SELECT COUNT(*) as total_errors,
                           AVG(severity) as avg_severity,
                           COUNT(DISTINCT error_type) as unique_types
                    FROM errors 
                    WHERE module = ? AND timestamp > datetime('now', '-24 hours')
                ''', (module,))
                
                stats = cursor.fetchone()
                
                if stats and stats[0] > 0:
                    total_errors, avg_severity, unique_types = stats
                    
                    # Вычисляем риск сбоя
                    risk_score = self._calculate_risk_score(total_errors, avg_severity, unique_types)
                    
                    # Предсказываем возможный сбой
                    if risk_score > 0.7:
                        self._predict_failure(module, risk_score)
                        
        except Exception as e:
            self.logger.error(f"❌ Ошибка анализа паттерна: {e}")
    
    def _calculate_risk_score(self, total_errors: int, avg_severity: float, unique_types: int) -> float:
        """Вычисление риска сбоя на основе статистики"""
        # Нормализуем метрики
        error_score = min(total_errors / 10.0, 1.0)  # Максимум 10 ошибок = 1.0
        severity_score = avg_severity / 5.0  # Максимум 5 = 1.0
        diversity_score = min(unique_types / 5.0, 1.0)  # Максимум 5 типов = 1.0
        
        # Взвешенная сумма
        risk_score = (error_score * 0.4 + severity_score * 0.4 + diversity_score * 0.2)
        return min(risk_score, 1.0)
    
    def _predict_failure(self, module: str, confidence: float):
        """Предсказание возможного сбоя модуля"""
        try:
            predicted_time = datetime.now() + timedelta(hours=1)  # Предсказываем сбой через час
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO predictions 
                    (module, prediction_type, confidence, predicted_time)
                    VALUES (?, 'failure', ?, ?)
                ''', (module, confidence, predicted_time))
                conn.commit()
                
            self.logger.warning(f"⚠️ Предсказан возможный сбой модуля {module} с вероятностью {confidence:.2f}")
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка предсказания: {e}")
    
    def get_error_statistics(self, hours: int = 24) -> Dict:
        """Получение статистики ошибок за указанный период"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Общая статистика
                cursor.execute('''
                    SELECT 
                        COUNT(*) as total_errors,
                        COUNT(DISTINCT module) as affected_modules,
                        AVG(severity) as avg_severity,
                        COUNT(DISTINCT error_type) as unique_types
                    FROM errors 
                    WHERE timestamp > datetime('now', '-{} hours')
                '''.format(hours))
                
                stats = cursor.fetchone()
                
                # Топ ошибок по модулям
                cursor.execute('''
                    SELECT module, COUNT(*) as error_count
                    FROM errors 
                    WHERE timestamp > datetime('now', '-{} hours')
                    GROUP BY module
                    ORDER BY error_count DESC
                    LIMIT 5
                '''.format(hours))
                
                top_modules = cursor.fetchall()
                
                # Топ типов ошибок
                cursor.execute('''
                    SELECT error_type, COUNT(*) as error_count
                    FROM errors 
                    WHERE timestamp > datetime('now', '-{} hours')
                    GROUP BY error_type
                    ORDER BY error_count DESC
                    LIMIT 5
                '''.format(hours))
                
                top_types = cursor.fetchall()
                
                return {
                    'total_errors': stats[0] or 0,
                    'affected_modules': stats[1] or 0,
                    'avg_severity': stats[2] or 0,
                    'unique_types': stats[3] or 0,
                    'top_modules': [{'module': m[0], 'count': m[1]} for m in top_modules],
                    'top_types': [{'type': t[0], 'count': t[1]} for t in top_types]
                }
                
        except Exception as e:
            self.logger.error(f"❌ Ошибка получения статистики: {e}")
            return {}
    
    def get_predictions(self) -> List[Dict]:
        """Получение активных предсказаний"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT module, prediction_type, confidence, predicted_time
                    FROM predictions 
                    WHERE actual_time IS NULL
                    ORDER BY predicted_time ASC
                ''')
                
                predictions = cursor.fetchall()
                return [
                    {
                        'module': p[0],
                        'type': p[1],
                        'confidence': p[2],
                        'predicted_time': p[3]
                    }
                    for p in predictions
                ]
                
        except Exception as e:
            self.logger.error(f"❌ Ошибка получения предсказаний: {e}")
            return []
    
    def mark_prediction_resolved(self, prediction_id: int, actual_time: datetime = None):
        """Отметка предсказания как выполненного"""
        try:
            if actual_time is None:
                actual_time = datetime.now()
                
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE predictions 
                    SET actual_time = ?, accuracy = 1.0
                    WHERE id = ?
                ''', (actual_time, prediction_id))
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"❌ Ошибка обновления предсказания: {e}")

# Глобальный экземпляр логгера ошибок
error_logger = ErrorLogger()






















