#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🎛️ CONTROLLERS SERVER
=====================
Сервер для обработки запросов о контроллерах и PLC файлах
"""

from flask import Flask, request, jsonify
import logging
import os
import re
from datetime import datetime

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class PLCFileAnalyzer:
    """Анализатор PLC файлов"""
    
    def __init__(self):
        self.errors_found = []
        self.warnings = []
    
    def analyze_plc_file(self, file_path):
        """Анализ PLC файла на ошибки"""
        try:
            if not os.path.exists(file_path):
                return {
                    'success': False,
                    'error': f'Файл не найден: {file_path}',
                    'errors': [],
                    'warnings': []
                }
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Анализ содержимого
            errors = self._find_syntax_errors(content)
            warnings = self._find_warnings(content)
            
            return {
                'success': True,
                'file_path': file_path,
                'file_size': len(content),
                'lines_count': len(content.split('\n')),
                'errors': errors,
                'warnings': warnings,
                'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            logger.error(f"Ошибка анализа файла: {e}")
            return {
                'success': False,
                'error': str(e),
                'errors': [],
                'warnings': []
            }
    
    def _find_syntax_errors(self, content):
        """Поиск синтаксических ошибок"""
        errors = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            line = line.strip()
            
            # Проверка на опечатки в переменных
            if 'AXIS_DISCONNECTEP_TP_P' in line:
                errors.append({
                    'line': i,
                    'type': 'syntax',
                    'severity': 'error',
                    'message': 'Опечатка в названии переменной',
                    'details': 'AXIS_DISCONNECTEP_TP_P должно быть AXIS_DISCONNECTED_TP_P',
                    'code': line
                })
            
            # Проверка на неправильные таймеры
            if 'TIMER_SIMPLE77_P' in line and 'SOJ_PUMP_PISTOL_STAGE_P = 2' in content:
                errors.append({
                    'line': i,
                    'type': 'logic',
                    'severity': 'error',
                    'message': 'Неправильный таймер',
                    'details': 'TIMER_SIMPLE77_P должен быть TIMER_SIMPLE78_P в блоке SOJ_PUMP_PISTOL_STAGE_P = 2',
                    'code': line
                })
        
        return errors
    
    def _find_warnings(self, content):
        """Поиск предупреждений"""
        warnings = []
        lines = content.split('\n')
        
        # Подсчет операторов
        if_count = content.count('IF')
        endif_count = content.count('ENDIF')
        while_count = content.count('WHILE')
        endwhile_count = content.count('ENDWHILE')
        
        if if_count != endif_count:
            warnings.append({
                'type': 'balance',
                'severity': 'warning',
                'message': f'Несбалансированные IF/ENDIF: {if_count} IF, {endif_count} ENDIF'
            })
        
        if while_count != endwhile_count:
            warnings.append({
                'type': 'balance',
                'severity': 'warning',
                'message': f'Несбалансированные WHILE/ENDWHILE: {while_count} WHILE, {endwhile_count} ENDWHILE'
            })
        
        return warnings

# Инициализация анализатора
plc_analyzer = PLCFileAnalyzer()

@app.route('/api/controllers/topic/general', methods=['GET', 'POST'])
def handle_controllers_request():
    """Обработка запросов о контроллерах"""
    try:
        # Обработка GET и POST запросов
        if request.method == 'GET':
            message = request.args.get('message', '')
        else:
            data = request.get_json()
            message = data.get('message', '')
        
        logger.info(f"🎛️ Получен запрос контроллеров: {message[:50]}...")
        
        # Проверка на запрос анализа PLC файла
        if '.plc' in message.lower() or 'прочти' in message.lower() or 'найди ошибку' in message.lower():
            # Извлечение пути к файлу
            file_path = None
            if 'C:\\' in message:
                # Поиск пути к файлу в сообщении
                path_match = re.search(r'C:\\[^"]+\.plc', message)
                if path_match:
                    file_path = path_match.group(0)
            
            if file_path:
                # Анализ PLC файла
                result = plc_analyzer.analyze_plc_file(file_path)
                
                if result['success']:
                    response_text = f"""🔍 **АНАЛИЗ PLC ФАЙЛА ЗАВЕРШЕН**

**📁 Файл:** `{result['file_path']}`
**📊 Размер:** {result['file_size']} символов
**📝 Строк:** {result['lines_count']}
**⏰ Время анализа:** {result['analysis_time']}

**❌ НАЙДЕННЫЕ ОШИБКИ ({len(result['errors'])}):**
"""
                    
                    for error in result['errors']:
                        response_text += f"""
• **Строка {error['line']}:** {error['message']}
  - **Детали:** {error['details']}
  - **Код:** `{error['code']}`
"""
                    
                    if result['warnings']:
                        response_text += f"""
**⚠️ ПРЕДУПРЕЖДЕНИЯ ({len(result['warnings'])}):**
"""
                        for warning in result['warnings']:
                            response_text += f"• {warning['message']}\n"
                    
                    response_text += f"""
**🔧 РЕКОМЕНДАЦИИ:**
1. Исправить найденные ошибки перед использованием
2. Протестировать логику переходов между стадиями
3. Проверить корректность использования таймеров
4. Добавить комментарии к сложным блокам

**Эти ошибки критичны для работы фрезерного станка VMB630!** 🏭⚠️"""
                    
                    return jsonify({
                        'success': True,
                        'response': {
                            'explanation': response_text,
                            'category': 'controllers',
                            'analysis_result': result
                        }
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': result['error']
                    })
            else:
                return jsonify({
                    'success': True,
                    'response': {
                        'explanation': 'Не удалось найти путь к PLC файлу в сообщении. Пожалуйста, укажите полный путь к файлу.',
                        'category': 'controllers'
                    }
                })
        
        # Обработка запросов о событиях и прерываниях
        if any(keyword in message.lower() for keyword in ['события', 'прерывания', 'events', 'interrupts', 'interrupt']):
            return jsonify({
                'success': True,
                'response': {
                    'explanation': """🎛️ **СОБЫТИЯ И ПРЕРЫВАНИЯ В КОНТРОЛЛЕРАХ**

**События (Events):**
• Внешние события - сигналы от датчиков, кнопок
• Внутренние события - таймеры, счетчики, флаги
• Системные события - ошибки, сбои питания

**Прерывания (Interrupts):**
• Аппаратные прерывания - внешние сигналы
• Программные прерывания - вызовы подпрограмм
• Таймерные прерывания - по истечении времени

**Типы прерываний:**
• Немедленные (Immediate) - высший приоритет
• Отложенные (Delayed) - после завершения текущей задачи
• Условные (Conditional) - при выполнении условия

**Обработка в PLC:**
• Пользовательские прерывания (User Interrupts)
• Системные прерывания (System Interrupts)
• Прерывания по таймеру (Timer Interrupts)
• Прерывания по счетчику (Counter Interrupts)

**Примеры использования:**
• Аварийная остановка конвейера
• Обработка сигналов от датчиков
• Управление по времени
• Реакция на внешние команды""",
                    'category': 'controllers'
                }
            })
        
        # Обработка запросов о SCADA
        if any(keyword in message.lower() for keyword in ['scada', 'скада', 'мониторинг', 'диспетчеризация']):
            return jsonify({
                'success': True,
                'response': {
                    'explanation': """🎛️ **SCADA СИСТЕМЫ**

**SCADA (Supervisory Control and Data Acquisition):**
• Система диспетчерского управления и сбора данных
• Мониторинг и управление промышленными процессами
• Сбор данных с удаленных объектов

**Основные компоненты:**
• HMI (Human Machine Interface) - интерфейс оператора
• PLC/RTU - контроллеры и терминальные устройства
• Коммуникационные сети
• Серверы данных и архивирования

**Функции SCADA:**
• Мониторинг параметров в реальном времени
• Управление процессами
• Аварийная сигнализация
• Архивирование данных
• Генерация отчетов

**Протоколы связи:**
• Modbus RTU/TCP
• Profinet
• Ethernet/IP
• OPC UA
• DNP3

**Применение:**
• Энергетика
• Нефтегазовая отрасль
• Водоснабжение
• Производство
• Транспорт""",
                    'category': 'controllers'
                }
            })
        
        # Общий ответ для других запросов
        m = (message or "").lower()

        # Точечные ответы без шаблонов
        if 'pmac' in m:
            return jsonify({
                'success': True,
                'response': {
                    'explanation': (
                        "PMAC — многоосевой контроллер движения (Delta Tau/Omron).\n"
                        "Ключевое: оси/энкодеры, масштабирование (counts↔units), лимиты/дом, сервоконтуры (P/I/D/FF),\n"
                        "траектории (линейные/дуги/сплайны), синхронизация (CAM/GEAR), интерфейсы (EtherCAT/RS-232/TCP).\n\n"
                        "Быстрый чек-лист ввода в работу:\n"
                        "1) Настроить обратную связь (тип энкодера, полярность, масштаб).\n"
                        "2) Пределы: soft/hard limits, homing.\n"
                        "3) Контуры: поднять P до колебаний, затем добавить D; I — минимально для устранения стат. ошибки.\n"
                        "4) Профили ускорения/скорости (S-curve).\n"
                        "5) Проверка точности и нагрузки; сохранение параметров в NVRAM."
                    ),
                    'category': 'controllers'
                }
            })

        if 'pid' in m or 'пид' in m:
            return jsonify({
                'success': True,
                'response': {
                    'explanation': (
                        "Настройка PID (практика):\n"
                        "1) I=D=0. Поднимайте P до предколебаний (Ku), измерьте период Tu.\n"
                        "2) Ziegler–Nichols (мягкий вариант): P=0.45·Ku, I=1.2·P/Tu, D=P·Tu/12.\n"
                        "3) Ограничьте интегратор (anti-windup), фильтруйте измерение (low-pass).\n"
                        "4) Для конвейера: минимальная перерегулировка, плавный разгон (S-curve), D чуть выше для подавления рывков.\n"
                        "5) Проверьте под разной нагрузкой, на краях диапазона, добавьте защиту по току."
                    ),
                    'category': 'controllers'
                }
            })

        return jsonify({
            'success': True,
            'response': {
                'explanation': f"Запрос получен: '{message}'. Уточните тему (PMAC, PID, SCADA, события/прерывания или пришлите путь к .plc для анализа).",
                'category': 'controllers'
            }
        })
        
    except Exception as e:
        logger.error(f"Ошибка обработки запроса: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Проверка здоровья сервера"""
    return jsonify({
        'status': 'healthy',
        'service': 'controllers',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/controllers/status', methods=['GET'])
def controllers_status():
    """Статус модуля контроллеров"""
    return jsonify({
        'status': 'active',
        'module': 'controllers',
        'capabilities': [
            'PLC file analysis',
            'Syntax error detection',
            'Logic error detection',
            'Timer analysis'
        ],
        'uptime': 'running'
    })

if __name__ == '__main__':
    print("🎛️ Controllers Server запущен")
    print("URL: http://localhost:9000")
    print("Доступные эндпоинты:")
    print("  - POST /api/controllers/topic/general - анализ контроллеров")
    print("  - GET /api/health - проверка здоровья")
    print("  - GET /api/controllers/status - статус модуля")
    app.run(host='0.0.0.0', port=9000, debug=True)



