#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
План улучшений системы Rubin AI
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, List, Set

class RubinAIImprover:
    def __init__(self, db_path="rubin_ai_v2.db"):
        self.db_path = db_path
        self.connection = None
        
    def connect(self):
        """Подключение к базе данных"""
        try:
            self.connection = sqlite3.connect(self.db_path)
            return True
        except Exception as e:
            print(f"❌ Ошибка подключения к БД: {e}")
            return False
    
    def close(self):
        """Закрытие соединения"""
        if self.connection:
            self.connection.close()
    
    def add_pid_documents(self):
        """Добавление документов по ПИД-регуляторам"""
        print("🔧 Добавление документов по ПИД-регуляторам...")
        
        pid_documents = [
            {
                'title': 'ПИД-регуляторы: Теория и практика',
                'content': '''
# ПИД-регуляторы: Теория и практика

## Основы ПИД-регулирования

ПИД-регулятор (Proportional-Integral-Derivative) - это устройство автоматического управления, которое использует три компонента для достижения желаемого значения регулируемой величины.

### Компоненты ПИД-регулятора:

1. **Пропорциональный компонент (P)**:
   - Реагирует на текущую ошибку
   - Формула: P = Kp × e(t)
   - Где Kp - коэффициент пропорциональности, e(t) - ошибка

2. **Интегральный компонент (I)**:
   - Устраняет статическую ошибку
   - Формула: I = Ki × ∫e(t)dt
   - Где Ki - коэффициент интеграции

3. **Дифференциальный компонент (D)**:
   - Предсказывает будущую ошибку
   - Формула: D = Kd × de(t)/dt
   - Где Kd - коэффициент дифференцирования

### Общая формула ПИД-регулятора:
u(t) = Kp × e(t) + Ki × ∫e(t)dt + Kd × de(t)/dt

## Настройка ПИД-регулятора

### Метод Зиглера-Николса:
1. Установить Ki = 0, Kd = 0
2. Увеличивать Kp до появления устойчивых колебаний
3. Записать критический коэффициент Kc и период колебаний Tc
4. Рассчитать параметры:
   - Kp = 0.6 × Kc
   - Ki = 2 × Kp / Tc
   - Kd = Kp × Tc / 8

### Практические рекомендации:
- Начинать с малых значений коэффициентов
- Настраивать по одному параметру
- Использовать осциллограф для контроля
- Учитывать инерционность системы

## Применение в промышленности

### Системы управления температурой:
- Печи и сушилки
- Теплицы
- Системы отопления

### Системы управления давлением:
- Компрессоры
- Насосы
- Пневматические системы

### Системы управления уровнем:
- Резервуары
- Баки
- Системы водоснабжения

## Преимущества ПИД-регуляторов:
- Высокая точность
- Быстрая реакция
- Универсальность
- Простота реализации

## Недостатки:
- Сложность настройки
- Чувствительность к шумам
- Нестабильность при неправильной настройке
                ''',
                'category': 'controllers',
                'metadata': {
                    'description': 'Подробное руководство по ПИД-регуляторам',
                    'author': 'Rubin AI',
                    'version': '2.0',
                    'tags': ['ПИД', 'регулятор', 'автоматизация', 'управление']
                }
            },
            {
                'title': 'Практические примеры настройки ПИД-регуляторов',
                'content': '''
# Практические примеры настройки ПИД-регуляторов

## Пример 1: Система управления температурой печи

### Параметры системы:
- Объем печи: 1 м³
- Мощность нагревателя: 5 кВт
- Целевая температура: 200°C
- Время нагрева: 30 минут

### Начальные настройки:
- Kp = 2.0
- Ki = 0.1
- Kd = 0.5

### Процесс настройки:
1. Установить Kp = 2.0, Ki = 0, Kd = 0
2. Наблюдать за переходным процессом
3. При перерегулировании > 20% уменьшить Kp
4. При медленном достижении уставки увеличить Kp
5. Добавить интегральную составляющую для устранения статической ошибки
6. Добавить дифференциальную составляющую для уменьшения перерегулирования

### Финальные параметры:
- Kp = 1.5
- Ki = 0.05
- Kd = 0.3

## Пример 2: Система управления давлением

### Параметры системы:
- Объем системы: 0.5 м³
- Производительность компрессора: 100 л/мин
- Целевое давление: 6 бар
- Время набора давления: 5 минут

### Настройка:
1. Начать с Kp = 1.0
2. При колебаниях уменьшить Kp до 0.7
3. Добавить Ki = 0.02 для устранения статической ошибки
4. Добавить Kd = 0.1 для стабилизации

### Результат:
- Статическая ошибка: < 0.1 бар
- Время установления: < 2 минут
- Перерегулирование: < 5%

## Пример 3: Система управления уровнем

### Параметры системы:
- Диаметр бака: 2 м
- Высота бака: 3 м
- Производительность насоса: 50 л/мин
- Целевой уровень: 2 м

### Особенности настройки:
- Учесть нелинейность системы
- Использовать адаптивные алгоритмы
- Учитывать задержки в системе

### Рекомендуемые параметры:
- Kp = 0.8
- Ki = 0.01
- Kd = 0.2

## Диагностика проблем

### Перерегулирование:
- Уменьшить Kp
- Увеличить Kd
- Проверить инерционность системы

### Медленная реакция:
- Увеличить Kp
- Уменьшить Kd
- Проверить производительность исполнительного устройства

### Статическая ошибка:
- Увеличить Ki
- Проверить настройки интегратора
- Учесть ограничения системы

### Нестабильность:
- Уменьшить все коэффициенты
- Проверить качество сигналов
- Учесть задержки в системе
                ''',
                'category': 'controllers',
                'metadata': {
                    'description': 'Практические примеры настройки ПИД-регуляторов',
                    'author': 'Rubin AI',
                    'version': '2.0',
                    'tags': ['ПИД', 'настройка', 'примеры', 'практика']
                }
            }
        ]
        
        cursor = self.connection.cursor()
        added_count = 0
        
        for doc in pid_documents:
            try:
                cursor.execute('''
                    INSERT INTO documents (file_name, content, category, file_path, file_size, 
                                         file_type, metadata, tags, created_at, updated_at, 
                                         difficulty_level, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    doc['title'],
                    doc['content'],
                    doc['category'],
                    f"pid_guide_{added_count + 1}.txt",
                    len(doc['content']),
                    'txt',
                    json.dumps(doc['metadata']),
                    json.dumps(doc['metadata'].get('tags', [])),
                    datetime.now().isoformat(),
                    datetime.now().isoformat(),
                    'medium',
                    datetime.now().isoformat()
                ))
                added_count += 1
                print(f"✅ Добавлен документ: {doc['title']}")
            except Exception as e:
                print(f"❌ Ошибка добавления документа: {e}")
        
        self.connection.commit()
        print(f"📊 Добавлено {added_count} документов по ПИД-регуляторам")
        return added_count
    
    def expand_synonyms(self):
        """Расширение словаря синонимов"""
        print("🔍 Расширение словаря синонимов...")
        
        new_synonyms = {
            # ПИД-регуляторы
            "ПИД-регулятор": ["PID-регулятор", "пид-регулятор", "pid-регулятор", "регулятор", "контроллер", "управление", "автоматика"],
            "PID-регулятор": ["ПИД-регулятор", "пид-регулятор", "pid-регулятор", "регулятор", "контроллер", "управление", "автоматика"],
            "пид-регулятор": ["ПИД-регулятор", "PID-регулятор", "pid-регулятор", "регулятор", "контроллер", "управление", "автоматика"],
            "pid-регулятор": ["ПИД-регулятор", "PID-регулятор", "пид-регулятор", "регулятор", "контроллер", "управление", "автоматика"],
            
            # Автоматизация
            "автоматизация": ["автоматика", "автоматизирование", "механизация", "роботизация", "управление", "контроль"],
            "автоматика": ["автоматизация", "автоматизирование", "механизация", "роботизация", "управление", "контроль"],
            "автоматизирование": ["автоматизация", "автоматика", "механизация", "роботизация", "управление", "контроль"],
            "механизация": ["автоматизация", "автоматика", "автоматизирование", "роботизация", "управление", "контроль"],
            "роботизация": ["автоматизация", "автоматика", "автоматизирование", "механизация", "управление", "контроль"],
            
            # ПЛК
            "ПЛК": ["PLC", "plc", "программируемый логический контроллер", "контроллер", "автоматика", "управление"],
            "PLC": ["ПЛК", "plc", "программируемый логический контроллер", "контроллер", "автоматика", "управление"],
            "plc": ["ПЛК", "PLC", "программируемый логический контроллер", "контроллер", "автоматика", "управление"],
            "программируемый логический контроллер": ["ПЛК", "PLC", "plc", "контроллер", "автоматика", "управление"],
            
            # SCADA
            "SCADA": ["скада", "диспетчерская система", "система мониторинга", "система управления", "автоматика"],
            "скада": ["SCADA", "диспетчерская система", "система мониторинга", "система управления", "автоматика"],
            "диспетчерская система": ["SCADA", "скада", "система мониторинга", "система управления", "автоматика"],
            "система мониторинга": ["SCADA", "скада", "диспетчерская система", "система управления", "автоматика"],
            
            # Датчики
            "датчик": ["сенсор", "измеритель", "преобразователь", "измерительный прибор", "контроль"],
            "сенсор": ["датчик", "измеритель", "преобразователь", "измерительный прибор", "контроль"],
            "измеритель": ["датчик", "сенсор", "преобразователь", "измерительный прибор", "контроль"],
            "преобразователь": ["датчик", "сенсор", "измеритель", "измерительный прибор", "контроль"],
            
            # Исполнительные устройства
            "исполнительное устройство": ["привод", "актуатор", "исполнительный механизм", "устройство управления"],
            "привод": ["исполнительное устройство", "актуатор", "исполнительный механизм", "устройство управления"],
            "актуатор": ["исполнительное устройство", "привод", "исполнительный механизм", "устройство управления"],
            "исполнительный механизм": ["исполнительное устройство", "привод", "актуатор", "устройство управления"],
            
            # Протоколы связи
            "Modbus": ["модбас", "протокол связи", "промышленная сеть", "коммуникация"],
            "модбас": ["Modbus", "протокол связи", "промышленная сеть", "коммуникация"],
            "Profibus": ["профибас", "протокол связи", "промышленная сеть", "коммуникация"],
            "профибас": ["Profibus", "протокол связи", "промышленная сеть", "коммуникация"],
            "Ethernet/IP": ["эзернет айпи", "протокол связи", "промышленная сеть", "коммуникация"],
            "эзернет айпи": ["Ethernet/IP", "протокол связи", "промышленная сеть", "коммуникация"]
        }
        
        cursor = self.connection.cursor()
        added_count = 0
        
        for term, synonyms in new_synonyms.items():
            for synonym in synonyms:
                try:
                    cursor.execute('''
                        INSERT OR IGNORE INTO synonyms (term, synonym, category, created_at)
                        VALUES (?, ?, ?, ?)
                    ''', (term, synonym, 'automation', datetime.now().isoformat()))
                    added_count += 1
                except Exception as e:
                    print(f"❌ Ошибка добавления синонима: {e}")
        
        self.connection.commit()
        print(f"📊 Добавлено {added_count} новых синонимов")
        return added_count
    
    def optimize_search_parameters(self):
        """Оптимизация параметров векторного поиска"""
        print("⚙️ Оптимизация параметров векторного поиска...")
        
        # Создаем таблицу для хранения параметров поиска
        cursor = self.connection.cursor()
        
        try:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS search_parameters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    parameter_name TEXT UNIQUE NOT NULL,
                    parameter_value TEXT NOT NULL,
                    description TEXT,
                    updated_at TEXT NOT NULL
                )
            ''')
            
            # Добавляем оптимизированные параметры
            parameters = [
                ('vector_search_threshold', '0.7', 'Порог схожести для векторного поиска'),
                ('max_search_results', '20', 'Максимальное количество результатов поиска'),
                ('text_search_threshold', '0.6', 'Порог схожести для текстового поиска'),
                ('ranking_weights', '{"vector": 0.4, "text": 0.3, "metadata": 0.3}', 'Веса для ранжирования результатов'),
                ('min_content_length', '100', 'Минимальная длина контента для индексации'),
                ('max_content_length', '50000', 'Максимальная длина контента для индексации'),
                ('synonym_expansion', 'true', 'Использовать расширение синонимов'),
                ('fuzzy_search', 'true', 'Использовать нечеткий поиск'),
                ('category_boost', '1.2', 'Коэффициент усиления для категории'),
                ('recent_document_boost', '1.1', 'Коэффициент усиления для недавних документов')
            ]
            
            for param_name, param_value, description in parameters:
                cursor.execute('''
                    INSERT OR REPLACE INTO search_parameters 
                    (parameter_name, parameter_value, description, updated_at)
                    VALUES (?, ?, ?, ?)
                ''', (param_name, param_value, description, datetime.now().isoformat()))
            
            self.connection.commit()
            print(f"✅ Добавлено {len(parameters)} параметров поиска")
            return len(parameters)
            
        except Exception as e:
            print(f"❌ Ошибка оптимизации параметров: {e}")
            return 0
    
    def enhance_metadata(self):
        """Улучшение метаданных документов"""
        print("📊 Улучшение метаданных документов...")
        
        cursor = self.connection.cursor()
        
        try:
            # Добавляем новые поля в таблицу документов
            cursor.execute('''
                ALTER TABLE documents ADD COLUMN tags TEXT DEFAULT '[]'
            ''')
        except:
            pass  # Поле уже существует
        
        try:
            cursor.execute('''
                ALTER TABLE documents ADD COLUMN difficulty_level TEXT DEFAULT 'medium'
            ''')
        except:
            pass  # Поле уже существует
        
        try:
            cursor.execute('''
                ALTER TABLE documents ADD COLUMN last_updated TEXT DEFAULT ''
            ''')
        except:
            pass  # Поле уже существует
        
        # Обновляем метаданные существующих документов
        cursor.execute('SELECT id, category, file_name FROM documents')
        documents = cursor.fetchall()
        
        updated_count = 0
        for doc_id, category, file_name in documents:
            # Определяем теги на основе категории и заголовка
            tags = []
            if 'electrical' in category.lower():
                tags.extend(['электротехника', 'электрика', 'энергетика'])
            if 'programming' in category.lower():
                tags.extend(['программирование', 'код', 'разработка'])
            if 'controllers' in category.lower():
                tags.extend(['контроллеры', 'автоматизация', 'ПЛК'])
            if 'radiomechanics' in category.lower():
                tags.extend(['радиотехника', 'связь', 'антенны'])
            if 'automation' in category.lower():
                tags.extend(['автоматизация', 'управление', 'системы'])
            
            # Определяем уровень сложности
            difficulty = 'medium'
            if any(word in file_name.lower() for word in ['базовый', 'основы', 'введение']):
                difficulty = 'beginner'
            elif any(word in file_name.lower() for word in ['продвинутый', 'сложный', 'экспертный']):
                difficulty = 'advanced'
            
            # Обновляем документ
            cursor.execute('''
                UPDATE documents 
                SET tags = ?, difficulty_level = ?, last_updated = ?
                WHERE id = ?
            ''', (json.dumps(tags), difficulty, datetime.now().isoformat(), doc_id))
            
            updated_count += 1
        
        self.connection.commit()
        print(f"✅ Обновлено {updated_count} документов с улучшенными метаданными")
        return updated_count
    
    def create_update_scheduler(self):
        """Создание системы регулярных обновлений"""
        print("🔄 Создание системы регулярных обновлений...")
        
        cursor = self.connection.cursor()
        
        try:
            # Создаем таблицу для отслеживания обновлений
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS update_schedule (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_name TEXT UNIQUE NOT NULL,
                    last_run TEXT,
                    next_run TEXT,
                    interval_hours INTEGER DEFAULT 24,
                    is_active BOOLEAN DEFAULT 1,
                    description TEXT
                )
            ''')
            
            # Добавляем задачи обновления
            tasks = [
                ('vector_index_update', 'Обновление векторных индексов', 6),
                ('synonym_expansion', 'Расширение словаря синонимов', 12),
                ('metadata_optimization', 'Оптимизация метаданных', 24),
                ('search_parameter_tuning', 'Настройка параметров поиска', 48),
                ('document_quality_check', 'Проверка качества документов', 72)
            ]
            
            for task_name, description, interval in tasks:
                cursor.execute('''
                    INSERT OR IGNORE INTO update_schedule 
                    (task_name, description, interval_hours, is_active)
                    VALUES (?, ?, ?, 1)
                ''', (task_name, description, interval))
            
            self.connection.commit()
            print(f"✅ Создано {len(tasks)} задач регулярного обновления")
            return len(tasks)
            
        except Exception as e:
            print(f"❌ Ошибка создания системы обновлений: {e}")
            return 0
    
    def run_all_improvements(self):
        """Запуск всех улучшений"""
        print("🚀 Запуск всех улучшений системы Rubin AI...\n")
        
        if not self.connect():
            return False
        
        results = {}
        
        # 1. Добавление документов по ПИД-регуляторам
        results['pid_documents'] = self.add_pid_documents()
        
        # 2. Расширение синонимов
        results['synonyms'] = self.expand_synonyms()
        
        # 3. Оптимизация параметров поиска
        results['search_params'] = self.optimize_search_parameters()
        
        # 4. Улучшение метаданных
        results['metadata'] = self.enhance_metadata()
        
        # 5. Создание системы обновлений
        results['scheduler'] = self.create_update_scheduler()
        
        self.close()
        
        # Итоговый отчет
        print("\n" + "="*50)
        print("📊 ИТОГОВЫЙ ОТЧЕТ ПО УЛУЧШЕНИЯМ")
        print("="*50)
        print(f"📄 Документов по ПИД-регуляторам: {results['pid_documents']}")
        print(f"🔍 Новых синонимов: {results['synonyms']}")
        print(f"⚙️ Параметров поиска: {results['search_params']}")
        print(f"📊 Обновленных метаданных: {results['metadata']}")
        print(f"🔄 Задач обновления: {results['scheduler']}")
        
        total_improvements = sum(results.values())
        print(f"\n✅ Всего улучшений: {total_improvements}")
        
        return results

def main():
    """Основная функция"""
    improver = RubinAIImprover()
    results = improver.run_all_improvements()
    
    if results:
        print("\n🎉 Все улучшения успешно применены!")
        print("🔄 Рекомендуется перезапустить систему для применения изменений")
    else:
        print("\n❌ Произошли ошибки при применении улучшений")

if __name__ == "__main__":
    main()
