#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Rubin AI Server с интеграцией интернет-поиска
Автоматически ищет информацию в интернете по контексту
"""

import json
import time
import sqlite3
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading
import os
import sys

# Добавляем текущую директорию в путь для импорта модулей
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Импортируем модуль контекстуального поиска
try:
    from rubin_contextual_internet_search import RubinContextualInternetSearch
    INTERNET_SEARCH_AVAILABLE = True
except ImportError:
    print("⚠️ Модуль интернет-поиска недоступен")
    INTERNET_SEARCH_AVAILABLE = False

class SmartRubinAIWithInternetHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Инициализируем систему интернет-поиска
        if INTERNET_SEARCH_AVAILABLE:
            self.internet_searcher = RubinContextualInternetSearch()
        else:
            self.internet_searcher = None
        
        super().__init__(*args, **kwargs)
    
    def init_database(self):
        """Инициализация базы данных"""
        try:
            conn = sqlite3.connect('rubin_ai.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message TEXT NOT NULL,
                    response TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    internet_search_used BOOLEAN DEFAULT FALSE,
                    search_category TEXT,
                    knowledge_saved BOOLEAN DEFAULT FALSE
                )
            ''')
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Ошибка инициализации БД: {e}")
            return False
    
    def save_message_to_db(self, message, response, internet_search_used=False, search_category=None, knowledge_saved=False):
        """Сохранение сообщения в базу данных"""
        try:
            conn = sqlite3.connect('rubin_ai.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO messages (message, response, internet_search_used, search_category, knowledge_saved) 
                VALUES (?, ?, ?, ?, ?)
            ''', (message, response, internet_search_used, search_category, knowledge_saved))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Ошибка сохранения в БД: {e}")
            return False
    
    def search_knowledge_base(self, message):
        """Поиск в базе знаний"""
        try:
            conn = sqlite3.connect('rubin_knowledge_base.db')
            cursor = conn.cursor()
            
            # Ищем по ключевым словам
            words = message.lower().split()
            search_terms = [word for word in words if len(word) > 3]
            
            if not search_terms:
                return None
            
            # Создаем поисковый запрос
            search_conditions = []
            for term in search_terms:
                search_conditions.append(f"title LIKE '%{term}%' OR content LIKE '%{term}%' OR keywords LIKE '%{term}%'")
            
            search_query = " OR ".join(search_conditions)
            
            cursor.execute(f'''
                SELECT title, content, category, tags 
                FROM knowledge_base 
                WHERE {search_query}
                ORDER BY relevance_score DESC, usage_count DESC
                LIMIT 3
            ''')
            
            results = cursor.fetchall()
            conn.close()
            
            if results:
                return results[0]  # Возвращаем наиболее релевантный результат
            
            return None
            
        except Exception as e:
            print(f"Ошибка поиска в БЗ: {e}")
            return None
    
    def generate_smart_response(self, message):
        """Генерирует умный ответ с интернет-поиском"""
        message_lower = message.lower()
        
        # Специфические обработчики (без изменений)
        if any(word in message_lower for word in ["контактор", "реle"]) and any(word in message_lower for word in ["одно", "то же", "это", "равно"]):
            return """⚡ **Техническая неточность в вопросе!**

❌ **Неправильно:** Контакторы и реле - это одно и то же устройство

✅ **Правильно:** Контакторы и реле - это **разные устройства** с разными характеристиками

**Основные различия:**

**Реле:**
• **Мощность** - до 10 А, низковольтные цепи
• **Назначение** - управление, сигнализация, логика
• **Контакты** - маломощные, для слаботочных цепей
• **Применение** - схемы управления, автоматика
• **Размер** - компактные, модульные

**Контакторы:**
• **Мощность** - от 10 А до сотен ампер
• **Назначение** - коммутация силовых цепей
• **Контакты** - мощные, для силовых нагрузок
• **Применение** - управление двигателями, нагревателями
• **Размер** - крупные, с дугогасительными камерами

**Конструктивные особенности:**

**Реле:**
• **Катушка** - 5-24 В DC/AC
• **Контакты** - NO, NC, переключающие
• **Дуга** - не требует гашения
• **Срок службы** - миллионы переключений

**Контакторы:**
• **Катушка** - 24-380 В AC/DC
• **Контакты** - силовые, с дугогасительными камерами
• **Дуга** - требует гашения (магнитное, воздушное)
• **Срок службы** - сотни тысяч переключений

**В промышленной автоматизации:**
• **PLC** - реле для логики, контакторы для силовых цепей
• **Схемы управления** - реле для сигналов, контакторы для двигателей
• **Защита** - реле контроля, контакторы с тепловыми реле

**Выбор устройства:**
• **Реле** - для управления, сигнализации, слаботочных цепей
• **Контакторы** - для коммутации мощных нагрузок, двигателей

Нужна помощь с выбором или подключением конкретного устройства?"""

        # Частотные преобразователи
        if any(word in message_lower for word in ["частотн", "преобразовател", "инвертор", "vfd", "частотник"]):
            return """⚡ **Частотные преобразователи - основа современной автоматизации!**

**Принцип работы:**
• **Выпрямление** - AC → DC (диодный мост)
• **Фильтрация** - сглаживание пульсаций (конденсаторы)
• **Инвертирование** - DC → AC (IGBT транзисторы)
• **ШИМ модуляция** - формирование синусоиды

**Основные функции:**
• **Плавный пуск** - снижение пусковых токов
• **Регулировка скорости** - 0-100% от номинальной
• **Реверс** - изменение направления вращения
• **Торможение** - динамическое и рекуперативное

**Типы управления:**
• **Скалярное (U/f)** - простое, для насосов/вентиляторов
• **Векторное** - точное, для сервоприводов
• **Прямое управление моментом** - максимальная точность

**Параметры настройки:**
• **Номинальная мощность** - соответствие двигателю
• **Частота** - 0-400 Гц (обычно 0-50 Гц)
• **Напряжение** - 220В/380В/660В
• **Ток** - номинальный и пусковой

**Защитные функции:**
• **Перегрузка** - защита от превышения тока
• **Перегрев** - температурная защита
• **Короткое замыкание** - мгновенное отключение
• **Обрыв фазы** - контроль питающей сети

**Применение:**
• **Насосы** - экономия энергии до 50%
• **Вентиляторы** - плавная регулировка
• **Конвейеры** - точное позиционирование
• **Станки** - регулировка скорости шпинделя

**Подключение:**
• **Вход** - 3-фазное питание 380В
• **Выход** - к двигателю (U, V, W)
• **Управление** - аналоговые/цифровые сигналы
• **Обратная связь** - энкодеры, тахогенераторы

**Настройка через HMI:**
• **Параметры двигателя** - мощность, ток, частота
• **Характеристики нагрузки** - момент, инерция
• **Режимы работы** - ручной/автоматический
• **Защиты** - уставки срабатывания

Какой аспект частотных преобразователей вас интересует?"""

        # Энкодеры
        if "энкодер" in message_lower:
            return """🔍 **Энкодеры - глаза автоматизации!**

**Типы энкодеров:**

**Инкрементальные:**
• **Принцип** - импульсы A, B, Z
• **Разрешение** - 100-10000 имп/об
• **Выход** - TTL, HTL, RS422
• **Применение** - скорость, направление

**Абсолютные:**
• **Принцип** - уникальный код позиции
• **Разрешение** - 12-24 бита (4096-16M позиций)
• **Выход** - SSI, Profibus, Ethernet
• **Применение** - точное позиционирование

**Оптические:**
• **Принцип** - светодиод + фотодиод
• **Точность** - высокая, до 0.01°
• **Скорость** - до 10000 об/мин
• **Надежность** - зависит от чистоты

**Магнитные:**
• **Принцип** - магнитное поле + датчик Холла
• **Точность** - средняя, до 0.1°
• **Скорость** - до 6000 об/мин
• **Надежность** - высокая, не боится грязи

**Индуктивные:**
• **Принцип** - электромагнитная индукция
• **Точность** - высокая, до 0.001°
• **Скорость** - до 30000 об/мин
• **Надежность** - очень высокая

**Подключение к PLC:**
• **Цифровые входы** - для инкрементальных
• **Специальные модули** - для абсолютных
• **Счетчики** - подсчет импульсов
• **Интерфейсы** - SSI, Profibus, Ethernet

**Настройка в PMAC:**
• **Разрешение** - количество импульсов на оборот
• **Направление** - прямая/обратная связь
• **Фильтрация** - подавление помех
• **Калибровка** - точная настройка

**Диагностика проблем:**
• **Нет сигнала** - проверка питания, кабелей
• **Неточность** - калибровка, загрязнение
• **Помехи** - экранирование, заземление
• **Износ** - замена подшипников, диска

**Выбор энкодера:**
• **Точность** - требуемое разрешение
• **Скорость** - максимальные обороты
• **Среда** - температура, влажность, вибрация
• **Интерфейс** - совместимость с контроллером

Какой тип энкодера или проблема вас интересует?"""

        # Поиск в локальной базе знаний
        knowledge_result = self.search_knowledge_base(message)
        if knowledge_result:
            title, content, category, tags = knowledge_result
            return f"""🧠 **Найдено в базе знаний:**

**📚 {title}**
*Категория: {category}*

{content}

*Теги: {tags}*

Нужна дополнительная информация по этой теме?"""

        # Интернет-поиск для новых/актуальных вопросов
        if self.internet_searcher:
            print("🌐 Проверяем необходимость интернет-поиска...")
            
            # Обрабатываем сообщение с контекстуальным поиском
            search_result = self.internet_searcher.process_user_message(message)
            
            if search_result['needs_internet_search']:
                print(f"🔍 Выполняем поиск в интернете: {search_result['search_category']}")
                
                if search_result['internet_results']:
                    response = f"""🌐 **Информация из интернета:**

🔍 **По вашему запросу:** "{message}"
📂 **Категория поиска:** {search_result['search_category']}

**Найденные источники:**

"""
                    
                    for i, result in enumerate(search_result['internet_results'], 1):
                        response += f"{i}. **{result['title']}**\n"
                        response += f"   🔗 {result['url']}\n"
                        response += f"   📝 {result['snippet']}\n\n"
                    
                    # Если есть проанализированный контент, добавляем его
                    if search_result['analyzed_content']:
                        analyzed = search_result['analyzed_content']
                        response += f"""📄 **Детальная информация:**

**{analyzed['title']}**

{analyzed['content'][:1000]}{'...' if len(analyzed['content']) > 1000 else ''}

🔗 **Источник:** {analyzed['url']}

"""
                        
                        # Сохраняем знания в базу
                        if self.internet_searcher.save_knowledge_from_internet(analyzed, message):
                            response += "💾 **Знания сохранены в базу для будущих ответов!**\n\n"
                    
                    response += """⚠️ **Важно:** Информация получена из интернета и может быть актуальной.
🔗 Для получения более детальной информации перейдите по ссылкам выше."""
                    
                    # Сохраняем сообщение с флагом интернет-поиска
                    self.save_message_to_db(
                        message, response, 
                        internet_search_used=True,
                        search_category=search_result['search_category'],
                        knowledge_saved=bool(search_result['analyzed_content'])
                    )
                    
                    return response
                else:
                    return f"""🤔 **Интересный вопрос:** "{message}"

🌐 **Попытка поиска в интернете не дала результатов.**

🎯 **Моя специализация:**
• Промышленная автоматизация
• Программирование PLC и PMAC
• Работа с датчиками и приводами
• Анализ и оптимизация кода

💡 **Для лучшего ответа уточните:**
• Конкретную техническую задачу
• Тип оборудования или системы
• Язык программирования (если применимо)
• Контекст применения

🔧 **Примеры конкретных вопросов:**
• "Как настроить термопару типа K в системе Siemens?"
• "Программа PLC для управления сервоприводом"
• "Анализ кода Python для сбора данных с датчиков"
• "Настройка PMAC для 3-осевого станка"

**Задайте более конкретный вопрос, и я дам детальный технический ответ!**"""

        # Fallback ответ
        return f"""🤖 **Понял ваш запрос:** "{message}"

🎯 **Моя специализация:**
• Промышленная автоматизация
• Программирование PLC и PMAC
• Работа с датчиками и приводами
• Анализ и оптимизация кода

💡 **Для лучшего ответа уточните:**
• Конкретную техническую задачу
• Тип оборудования или системы
• Есть ли код для анализа?
• Какие параметры важны?

🔧 **Примеры конкретных вопросов:**
• "Как настроить термопару типа K в системе Siemens?"
• "Программа PLC для управления сервоприводом"
• "Анализ кода Python для сбора данных с датчиками"
• "Настройка PMAC для 3-осевого станка"

**Задайте более конкретный вопрос, и я дам детальный технический ответ!**"""
    
    def do_POST(self):
        """Обработка POST запросов"""
        if self.path == '/api/chat':
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))
                
                message = data.get('message', '')
                if not message:
                    self.send_error_response('Сообщение не может быть пустым')
                    return
                
                print(f"Получено сообщение: {message}")
                
                # Генерируем ответ
                response = self.generate_smart_response(message)
                
                # Отправляем ответ
                self.send_json_response({
                    'response': response,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                })
                
            except Exception as e:
                print(f"Ошибка обработки POST: {e}")
                self.send_error_response(f'Ошибка обработки запроса: {str(e)}')
        else:
            self.send_error_response('Неизвестный endpoint')
    
    def do_GET(self):
        """Обработка GET запросов"""
        if self.path == '/health':
            self.send_json_response({
                'status': 'ok',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'internet_search_available': INTERNET_SEARCH_AVAILABLE,
                'pid': os.getpid()
            })
        elif self.path == '/api/stats':
            self.send_stats_response()
        else:
            self.send_error_response('Неизвестный endpoint')
    
    def send_stats_response(self):
        """Отправляет статистику"""
        try:
            conn = sqlite3.connect('rubin_ai.db')
            cursor = conn.cursor()
            
            # Общая статистика
            cursor.execute('SELECT COUNT(*) FROM messages')
            total_messages = cursor.fetchone()[0]
            
            # Статистика по интернет-поиску
            cursor.execute('SELECT COUNT(*) FROM messages WHERE internet_search_used = 1')
            internet_searches = cursor.fetchone()[0]
            
            # Статистика по сохранению знаний
            cursor.execute('SELECT COUNT(*) FROM messages WHERE knowledge_saved = 1')
            knowledge_saved = cursor.fetchone()[0]
            
            # Категории поиска
            cursor.execute('SELECT search_category, COUNT(*) FROM messages WHERE search_category IS NOT NULL GROUP BY search_category')
            search_categories = dict(cursor.fetchall())
            
            conn.close()
            
            self.send_json_response({
                'total_messages': total_messages,
                'internet_searches': internet_searches,
                'knowledge_saved': knowledge_saved,
                'search_categories': search_categories,
                'internet_search_available': INTERNET_SEARCH_AVAILABLE
            })
            
        except Exception as e:
            self.send_error_response(f'Ошибка получения статистики: {str(e)}')
    
    def send_json_response(self, data):
        """Отправляет JSON ответ"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
        response = json.dumps(data, ensure_ascii=False, indent=2)
        self.wfile.write(response.encode('utf-8'))
    
    def send_error_response(self, error_message):
        """Отправляет ответ с ошибкой"""
        self.send_response(400)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        error_response = json.dumps({'error': error_message}, ensure_ascii=False)
        self.wfile.write(error_response.encode('utf-8'))
    
    def log_message(self, format, *args):
        """Отключаем логирование в консоль"""
        pass

def run_server(port=8083):
    """Запуск сервера"""
    print("🚀 Запуск Smart Rubin AI Server с интернет-поиском...")
    print(f"📡 Порт: {port}")
    print(f"🌐 Интернет-поиск: {'✅ Доступен' if INTERNET_SEARCH_AVAILABLE else '❌ Недоступен'}")
    print("=" * 60)
    
    # Инициализируем базу данных
    handler = SmartRubinAIWithInternetHandler
    temp_handler = handler(None, None, None)
    if temp_handler.init_database():
        print("✅ База данных инициализирована")
    else:
        print("❌ Ошибка инициализации базы данных")
        return
    
    try:
        server = HTTPServer(('localhost', port), handler)
        print(f"🎉 Сервер запущен на http://localhost:{port}")
        print("📋 Доступные endpoints:")
        print(f"   POST http://localhost:{port}/api/chat - чат с AI")
        print(f"   GET  http://localhost:{port}/health - статус сервера")
        print(f"   GET  http://localhost:{port}/api/stats - статистика")
        print("\n🛑 Для остановки нажмите Ctrl+C")
        print("=" * 60)
        
        server.serve_forever()
        
    except KeyboardInterrupt:
        print("\n🛑 Остановка сервера...")
        server.shutdown()
        print("✅ Сервер остановлен")
    except Exception as e:
        print(f"❌ Ошибка сервера: {e}")

if __name__ == "__main__":
    run_server()
