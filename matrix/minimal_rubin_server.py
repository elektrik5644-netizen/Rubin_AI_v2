"""
Минимальный сервер Rubin AI для быстрого запуска
"""

import json
import time
import sqlite3
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading

class RubinAIHandler(BaseHTTPRequestHandler):
    def init_database(self):
        """Инициализация базы данных"""
        try:
            conn = sqlite3.connect('rubin_ai.db')
            cursor = conn.cursor()
            
            # Создаем таблицу сообщений
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message TEXT NOT NULL,
                    response TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Ошибка инициализации БД: {e}")
            return False
    
    def save_message_to_db(self, message, response):
        """Сохранение сообщения в базу данных"""
        try:
            conn = sqlite3.connect('rubin_ai.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO messages (message, response) 
                VALUES (?, ?)
            ''', (message, response))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Ошибка сохранения в БД: {e}")
            return False
    
    def do_GET(self):
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            response = {
                "status": "healthy",
                "message": "Rubin AI Matrix Simple is running!",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "version": "2.0.0"
            }
            self.wfile.write(json.dumps(response, ensure_ascii=False).encode('utf-8'))
            
        elif parsed_path.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            response = {
                "message": "Rubin AI Matrix Simple v2.0",
                "status": "running",
                "endpoints": ["/health", "/api/chat", "/api/code/analyze"]
            }
            self.wfile.write(json.dumps(response, ensure_ascii=False).encode('utf-8'))
            
        else:
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            response = {"error": "Endpoint not found"}
            self.wfile.write(json.dumps(response).encode('utf-8'))
    
    def do_POST(self):
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/api/chat':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                message = data.get('message', '')
                
                # Простая логика ответов
                response_text = self.generate_chat_response(message)
                
                # Сохраняем в базу данных
                self.save_message_to_db(message, response_text)
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                response = {
                    "response": response_text,
                    "session_id": "default",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "processing_time": 0.1
                }
                self.wfile.write(json.dumps(response, ensure_ascii=False).encode('utf-8'))
                
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                response = {"error": f"Chat processing failed: {str(e)}"}
                self.wfile.write(json.dumps(response).encode('utf-8'))
        
        elif parsed_path.path == '/api/code/analyze':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                code = data.get('code', '')
                language = data.get('language', 'python')
                
                # Простой анализ кода
                analysis_result = self.analyze_code(code, language)
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                response = {
                    "language": language,
                    "analysis_type": "full",
                    "issues": analysis_result.get("issues", []),
                    "recommendations": analysis_result.get("recommendations", []),
                    "quality_score": analysis_result.get("quality_score", 0),
                    "results": analysis_result,
                    "processing_time": 0.2,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
                }
                self.wfile.write(json.dumps(response, ensure_ascii=False).encode('utf-8'))
                
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                response = {"error": f"Code analysis failed: {str(e)}"}
                self.wfile.write(json.dumps(response).encode('utf-8'))
        
        else:
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            response = {"error": "Endpoint not found"}
            self.wfile.write(json.dumps(response).encode('utf-8'))
    
    def do_OPTIONS(self):
        """Обработка OPTIONS запросов для CORS"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.send_header('Access-Control-Max-Age', '86400')
        self.end_headers()
    
    def generate_chat_response(self, message):
        """Генерация ответа чата"""
        message_lower = message.lower()
        
        # Приветствие
        if "привет" in message_lower or "hello" in message_lower:
            return "Привет! Я Rubin AI. Готов помочь с программированием и промышленной автоматизацией!"
        
        # Состояние системы
        elif "как дела" in message_lower:
            return "Отлично! Система работает стабильно. Чем могу помочь?"
        
        # Термопары
        elif "термопар" in message_lower:
            if "сопротивлени" in message_lower:
                return """❌ Неточность в вопросе! 

Термопары НЕ работают на принципе изменения сопротивления. 

🔬 **Принцип работы термопар:**
• Термопары работают на принципе **термоэлектрического эффекта** (эффект Зеебека)
• При нагреве места соединения двух разных металлов возникает ЭДС
• Измеряется не сопротивление, а **напряжение** (милливольты)
• Чем больше разность температур, тем больше напряжение

🌡️ **Типы термопар:**
• Тип K (хромель-алюмель): -200°C до +1200°C
• Тип J (железо-константан): -200°C до +750°C
• Тип T (медь-константан): -200°C до +350°C

💡 **Для измерения сопротивления используются:**
• RTD (Resistance Temperature Detectors) - платиновые термометры
• Термисторы - полупроводниковые датчики

Нужна помощь с настройкой термопар в системе автоматизации?"""
        
        # Python и переменные
        elif "python" in message_lower and ("переменн" in message_lower or "объявл" in message_lower):
            return """✅ **Правильно!** В Python переменные нужно объявлять перед использованием.

🐍 **Особенности Python:**
• Python - язык с **динамической типизацией**
• Переменные создаются при первом присваивании
• Тип определяется автоматически

📝 **Примеры объявления:**
```python
# Правильно - объявляем перед использованием
name = "Rubin AI"
age = 25
temperature = 23.5
is_active = True

# Неправильно - использование без объявления
print(undefined_var)  # NameError: name 'undefined_var' is not defined
```

🔧 **Лучшие практики:**
• Используйте описательные имена: `sensor_temperature` вместо `t`
• Инициализируйте переменные: `counter = 0`
• Проверяйте существование: `if 'var' in locals():`

Нужна помощь с конкретным кодом Python?"""
        
        # Python общий
        elif "python" in message_lower:
            return """🐍 **Python - отличный выбор для автоматизации!**

🚀 **Преимущества Python:**
• Простой синтаксис
• Богатые библиотеки для промышленности
• Отличная поддержка PLC и PMAC
• Быстрая разработка

📚 **Популярные библиотеки:**
• `pymodbus` - работа с Modbus
• `opcua` - OPC UA клиенты
• `numpy` - математические вычисления
• `pandas` - обработка данных

🔧 **Для анализа кода используйте:** /api/code/analyze

Нужна помощь с конкретной задачей?"""
        
        # PLC программирование
        elif "plc" in message_lower or "плц" in message_lower:
            return """🏭 **PLC программирование - моя специализация!**

⚡ **Языки программирования PLC:**
• **Ladder Logic (LD)** - релейная логика
• **Structured Text (ST)** - текстовый язык
• **Function Block Diagram (FBD)** - функциональные блоки
• **Instruction List (IL)** - список инструкций

🔧 **Популярные производители:**
• Siemens (S7-1200, S7-1500)
• Allen-Bradley (CompactLogix, ControlLogix)
• Schneider Electric (Modicon)
• Omron (CP1, CJ2)

💡 **Могу помочь с:**
• Написанием программ
• Диагностикой ошибок
• Оптимизацией кода
• Интеграцией с PMAC

Есть конкретная задача?"""
        
        # PMAC контроллеры
        elif "pmac" in message_lower:
            return """🎯 **PMAC контроллеры - это моя область!**

⚙️ **PMAC (Programmable Multi-Axis Controller):**
• Высокоточное управление движением
• До 32 осей одновременно
• Встроенная математика
• Реальное время

🔧 **Основные функции:**
• Позиционирование
• Интерполяция траекторий
• Синхронизация осей
• Обратная связь

📝 **Языки программирования:**
• **Motion Programs** - программы движения
• **PLC Programs** - логика управления
• **Background Programs** - фоновые задачи

💡 **Могу помочь с:**
• Настройкой осей
• Программированием движения
• Диагностикой
• Оптимизацией

Какая задача стоит?"""
        
        # Анализ кода
        elif "анализ" in message_lower or "анализ кода" in message_lower:
            return """🔍 **Анализ кода - одна из моих основных функций!**

📊 **Что я анализирую:**
• **Python** - синтаксис, стиль, безопасность
• **C/C++** - память, производительность
• **SQL** - оптимизация запросов
• **PLC** - логика, эффективность
• **PMAC** - программы движения

🎯 **Типы анализа:**
• Синтаксические ошибки
• Проблемы производительности
• Уязвимости безопасности
• Рекомендации по улучшению

🚀 **Использование:**
1. Загрузите код через интерфейс
2. Выберите язык программирования
3. Получите детальный отчет

Готов проанализировать ваш код!"""
        
        # Помощь
        elif "помощь" in message_lower or "help" in message_lower:
            return """🆘 **Доступные функции Rubin AI:**

🔧 **Программирование:**
• Анализ кода (Python, C, SQL, PLC, PMAC)
• Генерация кода
• Отладка и оптимизация

🏭 **Промышленная автоматизация:**
• Программирование PLC
• Настройка PMAC контроллеров
• Работа с датчиками (термопары, RTD)
• Диагностика оборудования

📊 **Анализ и диагностика:**
• Анализ производительности
• Поиск ошибок
• Рекомендации по улучшению

💡 **Примеры вопросов:**
• "Как настроить термопару типа K?"
• "Анализ кода Python"
• "Программа PLC для управления двигателем"
• "Настройка PMAC для 3-осевого станка"

Чем конкретно могу помочь?"""
        
        # По умолчанию - умный ответ
        else:
            # Анализируем ключевые слова в сообщении
            keywords = []
            if any(word in message_lower for word in ["датчик", "сенсор", "измерен"]):
                keywords.append("датчики")
            if any(word in message_lower for word in ["двигатель", "мотор", "привод"]):
                keywords.append("двигатели")
            if any(word in message_lower for word in ["программ", "код", "скрипт"]):
                keywords.append("программирование")
            if any(word in message_lower for word in ["ошибка", "проблема", "не работает"]):
                keywords.append("диагностика")
            
            if keywords:
                return f"""🔍 **Анализ вашего вопроса:** {', '.join(keywords)}

💡 **Могу помочь с:**
• Техническими решениями
• Анализом кода
• Диагностикой проблем
• Программированием PLC/PMAC

📝 **Для более точного ответа уточните:**
• Какой тип оборудования?
• Какая конкретная задача?
• Есть ли код для анализа?

**Ваш вопрос:** "{message}"

Готов дать детальный ответ!"""
            else:
                return f"""🤖 **Понял ваш запрос:** "{message}"

🎯 **Я специализируюсь на:**
• Промышленной автоматизации
• Программировании PLC и PMAC
• Анализе кода
• Работе с датчиками и приводами

💡 **Для лучшего ответа уточните:**
• Конкретную техническую задачу
• Тип оборудования
• Язык программирования

Чем конкретно могу помочь?"""
    
    def analyze_code(self, code, language):
        """Простой анализ кода"""
        issues = []
        recommendations = []
        quality_score = 85.0
        
        if language.lower() == "python":
            if "import *" in code:
                issues.append({
                    "type": "warning",
                    "message": "Использование 'import *' не рекомендуется",
                    "severity": "medium"
                })
                recommendations.append("Используйте конкретные импорты")
            
            if "eval(" in code:
                issues.append({
                    "type": "security",
                    "message": "Использование eval() может быть небезопасно",
                    "severity": "high"
                })
                recommendations.append("Избегайте использования eval()")
            
            if len(code.split('\n')) > 50:
                issues.append({
                    "type": "quality",
                    "message": "Код довольно длинный",
                    "severity": "low"
                })
                recommendations.append("Рассмотрите разбиение на функции")
        
        elif language.lower() == "c":
            if "printf(" in code and "stdio.h" not in code:
                issues.append({
                    "type": "error",
                    "message": "Использование printf() без подключения stdio.h",
                    "severity": "high"
                })
                recommendations.append("Добавьте #include <stdio.h>")
                quality_score -= 15
            
            if "malloc(" in code and "stdlib.h" not in code:
                issues.append({
                    "type": "error", 
                    "message": "Использование malloc() без подключения stdlib.h",
                    "severity": "high"
                })
                recommendations.append("Добавьте #include <stdlib.h>")
                quality_score -= 15
                
        elif language.lower() == "sql":
            if "SELECT *" in code.upper():
                issues.append({
                    "type": "warning",
                    "message": "Использование SELECT * может быть неэффективным",
                    "severity": "medium"
                })
                recommendations.append("Указывайте конкретные колонки вместо *")
                quality_score -= 5
                
            if "WHERE" not in code.upper() and "SELECT" in code.upper():
                issues.append({
                    "type": "warning",
                    "message": "SELECT без WHERE может быть неэффективным",
                    "severity": "medium"
                })
                recommendations.append("Добавьте условие WHERE для фильтрации")
                quality_score -= 5
                
        elif language.lower() in ["ladder", "st", "fbd"]:
            if "TON" not in code and "TOF" not in code:
                recommendations.append("Рассмотрите использование таймеров")
            quality_score = 80.0
        
        quality_score = max(60, quality_score - len(issues) * 5)
        
        return {
            "issues": issues,
            "quality_score": quality_score,
            "recommendations": recommendations,
            "security_report": {"level": "low", "issues": []},
            "summary": {
                "total_issues": len(issues),
                "security_issues": len([i for i in issues if i.get("type") == "security"]),
                "code_length": len(code.split('\n')),
                "language": language
            }
        }
    
    def log_message(self, format, *args):
        """Отключение логов для чистоты вывода"""
        pass

def init_database():
    """Инициализация базы данных"""
    try:
        conn = sqlite3.connect('rubin_ai.db')
        cursor = conn.cursor()
        
        # Создаем таблицу сообщений
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message TEXT NOT NULL,
                response TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Ошибка инициализации БД: {e}")
        return False

def run_server(port=8083):
    """Запуск сервера"""
    # Инициализируем базу данных
    if init_database():
        print("✅ База данных инициализирована")
    else:
        print("❌ Ошибка инициализации базы данных")
    
    server_address = ('', port)
    httpd = HTTPServer(server_address, RubinAIHandler)
    
    print(f"🚀 Rubin AI Matrix Simple запущен!")
    print(f"🌐 Сервер доступен по адресу: http://localhost:{port}")
    print(f"📊 Проверка здоровья: http://localhost:{port}/health")
    print(f"💬 API чат: http://localhost:{port}/api/chat")
    print(f"🔍 Анализ кода: http://localhost:{port}/api/code/analyze")
    print(f"⏹️  Для остановки нажмите Ctrl+C")
    print("-" * 50)
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Сервер остановлен")
        httpd.server_close()

if __name__ == "__main__":
    run_server()
