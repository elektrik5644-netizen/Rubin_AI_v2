#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Примеры кода на Python для различных задач
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import sqlite3
import requests

# =============================================================================
# МАТЕМАТИЧЕСКИЕ ВЫЧИСЛЕНИЯ
# =============================================================================

def solve_quadratic_equation(a, b, c):
    """
    Решение квадратного уравнения ax² + bx + c = 0
    """
    discriminant = b**2 - 4*a*c
    
    if discriminant > 0:
        x1 = (-b + math.sqrt(discriminant)) / (2*a)
        x2 = (-b - math.sqrt(discriminant)) / (2*a)
        return x1, x2
    elif discriminant == 0:
        x = -b / (2*a)
        return x, x
    else:
        real_part = -b / (2*a)
        imag_part = math.sqrt(-discriminant) / (2*a)
        return complex(real_part, imag_part), complex(real_part, -imag_part)

def calculate_pid_output(error, prev_error, integral, kp, ki, kd, dt):
    """
    Расчет выхода ПИД-регулятора
    """
    # Пропорциональная составляющая
    proportional = kp * error
    
    # Интегральная составляющая
    integral += error * dt
    integral_term = ki * integral
    
    # Дифференциальная составляющая
    derivative = (error - prev_error) / dt
    derivative_term = kd * derivative
    
    # Общий выход
    output = proportional + integral_term + derivative_term
    
    return output, integral

# =============================================================================
# РАБОТА С ДАННЫМИ
# =============================================================================

class DataProcessor:
    """Класс для обработки данных"""
    
    def __init__(self):
        self.data = []
        self.statistics = {}
    
    def add_data(self, value):
        """Добавление данных"""
        self.data.append(value)
    
    def calculate_statistics(self):
        """Расчет статистики"""
        if not self.data:
            return None
        
        self.statistics = {
            'count': len(self.data),
            'mean': np.mean(self.data),
            'median': np.median(self.data),
            'std': np.std(self.data),
            'min': np.min(self.data),
            'max': np.max(self.data),
            'range': np.max(self.data) - np.min(self.data)
        }
        return self.statistics
    
    def filter_outliers(self, threshold=2):
        """Фильтрация выбросов"""
        if not self.data:
            return []
        
        mean = np.mean(self.data)
        std = np.std(self.data)
        
        filtered_data = []
        for value in self.data:
            if abs(value - mean) <= threshold * std:
                filtered_data.append(value)
        
        return filtered_data
    
    def plot_data(self, title="Данные"):
        """Построение графика"""
        if not self.data:
            print("Нет данных для построения графика")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.data, 'b-', linewidth=2, label='Данные')
        plt.axhline(y=np.mean(self.data), color='r', linestyle='--', label='Среднее')
        plt.xlabel('Время')
        plt.ylabel('Значение')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()

# =============================================================================
# РАБОТА С БАЗОЙ ДАННЫХ
# =============================================================================

class DatabaseManager:
    """Менеджер базы данных"""
    
    def __init__(self, db_name):
        self.db_name = db_name
        self.connection = None
    
    def connect(self):
        """Подключение к базе данных"""
        try:
            self.connection = sqlite3.connect(self.db_name)
            print(f"Подключение к базе данных {self.db_name} установлено")
            return True
        except sqlite3.Error as e:
            print(f"Ошибка подключения к базе данных: {e}")
            return False
    
    def create_table(self, table_name, columns):
        """Создание таблицы"""
        if not self.connection:
            print("Нет подключения к базе данных")
            return False
        
        try:
            cursor = self.connection.cursor()
            columns_str = ', '.join(columns)
            query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_str})"
            cursor.execute(query)
            self.connection.commit()
            print(f"Таблица {table_name} создана успешно")
            return True
        except sqlite3.Error as e:
            print(f"Ошибка создания таблицы: {e}")
            return False
    
    def insert_data(self, table_name, data):
        """Вставка данных"""
        if not self.connection:
            print("Нет подключения к базе данных")
            return False
        
        try:
            cursor = self.connection.cursor()
            placeholders = ', '.join(['?' for _ in data])
            query = f"INSERT INTO {table_name} VALUES ({placeholders})"
            cursor.execute(query, data)
            self.connection.commit()
            return True
        except sqlite3.Error as e:
            print(f"Ошибка вставки данных: {e}")
            return False
    
    def select_data(self, table_name, condition=None):
        """Выборка данных"""
        if not self.connection:
            print("Нет подключения к базе данных")
            return []
        
        try:
            cursor = self.connection.cursor()
            query = f"SELECT * FROM {table_name}"
            if condition:
                query += f" WHERE {condition}"
            cursor.execute(query)
            return cursor.fetchall()
        except sqlite3.Error as e:
            print(f"Ошибка выборки данных: {e}")
            return []
    
    def close(self):
        """Закрытие подключения"""
        if self.connection:
            self.connection.close()
            print("Подключение к базе данных закрыто")

# =============================================================================
# СЕТЕВЫЕ ОПЕРАЦИИ
# =============================================================================

class NetworkManager:
    """Менеджер сетевых операций"""
    
    def __init__(self, base_url):
        self.base_url = base_url
        self.session = requests.Session()
    
    def get_request(self, endpoint, params=None):
        """GET запрос"""
        try:
            url = f"{self.base_url}/{endpoint}"
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Ошибка GET запроса: {e}")
            return None
    
    def post_request(self, endpoint, data=None):
        """POST запрос"""
        try:
            url = f"{self.base_url}/{endpoint}"
            response = self.session.post(url, json=data)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Ошибка POST запроса: {e}")
            return None
    
    def check_connection(self):
        """Проверка соединения"""
        try:
            response = self.session.get(self.base_url, timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

# =============================================================================
# КОНФИГУРАЦИЯ И НАСТРОЙКИ
# =============================================================================

class ConfigManager:
    """Менеджер конфигурации"""
    
    def __init__(self, config_file):
        self.config_file = config_file
        self.config = {}
    
    def load_config(self):
        """Загрузка конфигурации"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            print(f"Конфигурация загружена из {self.config_file}")
            return True
        except FileNotFoundError:
            print(f"Файл конфигурации {self.config_file} не найден")
            return False
        except json.JSONDecodeError as e:
            print(f"Ошибка парсинга JSON: {e}")
            return False
    
    def save_config(self):
        """Сохранение конфигурации"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
            print(f"Конфигурация сохранена в {self.config_file}")
            return True
        except Exception as e:
            print(f"Ошибка сохранения конфигурации: {e}")
            return False
    
    def get(self, key, default=None):
        """Получение значения по ключу"""
        return self.config.get(key, default)
    
    def set(self, key, value):
        """Установка значения по ключу"""
        self.config[key] = value

# =============================================================================
# ЛОГИРОВАНИЕ
# =============================================================================

import logging

class Logger:
    """Класс для логирования"""
    
    def __init__(self, name, log_file=None, level=logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Очистка существующих обработчиков
        self.logger.handlers.clear()
        
        # Форматтер
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Обработчик для консоли
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Обработчик для файла
        if log_file:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def info(self, message):
        """Информационное сообщение"""
        self.logger.info(message)
    
    def warning(self, message):
        """Предупреждение"""
        self.logger.warning(message)
    
    def error(self, message):
        """Ошибка"""
        self.logger.error(message)
    
    def debug(self, message):
        """Отладочное сообщение"""
        self.logger.debug(message)

# =============================================================================
# ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ
# =============================================================================

def main():
    """Основная функция с примерами использования"""
    
    print("=== ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ ===\n")
    
    # 1. Решение квадратного уравнения
    print("1. Решение квадратного уравнения x² - 5x + 6 = 0:")
    x1, x2 = solve_quadratic_equation(1, -5, 6)
    print(f"   x1 = {x1}, x2 = {x2}\n")
    
    # 2. ПИД-регулятор
    print("2. ПИД-регулятор:")
    error = 10.0
    prev_error = 12.0
    integral = 5.0
    kp, ki, kd = 2.0, 0.1, 0.05
    dt = 0.1
    
    output, new_integral = calculate_pid_output(
        error, prev_error, integral, kp, ki, kd, dt
    )
    print(f"   Выход ПИД-регулятора: {output:.2f}\n")
    
    # 3. Обработка данных
    print("3. Обработка данных:")
    processor = DataProcessor()
    
    # Добавление тестовых данных
    test_data = [1, 2, 3, 4, 5, 100, 6, 7, 8, 9, 10]  # 100 - выброс
    for value in test_data:
        processor.add_data(value)
    
    # Расчет статистики
    stats = processor.calculate_statistics()
    print(f"   Статистика: {stats}")
    
    # Фильтрация выбросов
    filtered = processor.filter_outliers()
    print(f"   Отфильтрованные данные: {filtered}\n")
    
    # 4. Работа с базой данных
    print("4. Работа с базой данных:")
    db = DatabaseManager("test.db")
    
    if db.connect():
        # Создание таблицы
        columns = ["id INTEGER PRIMARY KEY", "name TEXT", "value REAL", "timestamp TEXT"]
        db.create_table("measurements", columns)
        
        # Вставка данных
        timestamp = datetime.now().isoformat()
        db.insert_data("measurements", (1, "sensor1", 25.5, timestamp))
        
        # Выборка данных
        data = db.select_data("measurements")
        print(f"   Данные из БД: {data}")
        
        db.close()
    
    # 5. Сетевые операции
    print("\n5. Сетевые операции:")
    network = NetworkManager("http://localhost:8084")
    
    if network.check_connection():
        print("   Соединение установлено")
        # Пример GET запроса
        # response = network.get_request("api/health")
        # print(f"   Ответ сервера: {response}")
    else:
        print("   Соединение не установлено")
    
    # 6. Конфигурация
    print("\n6. Конфигурация:")
    config = ConfigManager("config.json")
    
    # Создание тестовой конфигурации
    config.set("database", {"host": "localhost", "port": 5432})
    config.set("api", {"base_url": "http://localhost:8084"})
    config.set("logging", {"level": "INFO", "file": "app.log"})
    
    if config.save_config():
        print("   Конфигурация сохранена")
    
    if config.load_config():
        print(f"   Загруженная конфигурация: {config.config}")
    
    # 7. Логирование
    print("\n7. Логирование:")
    logger = Logger("test_app", "test.log")
    logger.info("Приложение запущено")
    logger.warning("Это предупреждение")
    logger.error("Это ошибка")
    logger.debug("Отладочная информация")
    
    print("\n=== ПРИМЕРЫ ЗАВЕРШЕНЫ ===")

if __name__ == "__main__":
    main()






















