#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rubin AI - Математический сервер
Решает математические задачи без шаблонных ответов
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import logging
import json
import os
from datetime import datetime

# Настройка логирования (перенесено выше)
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Попытка импорта нейронной сети
try:
    from neural_rubin import get_neural_rubin
    NEURAL_NETWORK_AVAILABLE = True
    logger.info("🧠 Нейронная сеть доступна!")
except ImportError as e:
    NEURAL_NETWORK_AVAILABLE = False
    logger.warning(f"⚠️ Нейронная сеть недоступна: {e}")
    logger.info("💡 Для установки запустите: python install_ml_dependencies.py")

# Попытка импорта улучшенного диспетчера
try:
    from intelligent_dispatcher import get_intelligent_dispatcher
    ENHANCED_DISPATCHER_AVAILABLE = True
    enhanced_dispatcher = get_intelligent_dispatcher()
    logger.info("✅ Улучшенный диспетчер доступен!")
except ImportError as e:
    ENHANCED_DISPATCHER_AVAILABLE = False
    enhanced_dispatcher = None
    logger.warning(f"⚠️ Улучшенный диспетчер недоступен: {e}")
    logger.info("💡 Используется fallback логика")

# Логирование уже настроено выше

app = Flask(__name__)
CORS(app)

# Простая система "обучения" - сохранение диалогов
def log_conversation(question, answer, success=True):
    """Логирует диалоги для анализа и 'обучения'"""
    try:
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'answer': answer[:200] + '...' if len(answer) > 200 else answer,
            'success': success,
            'question_length': len(question),
            'answer_length': len(answer)
        }
        
        # Сохраняем в файл
        with open('rubin_learning_log.json', 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    except Exception as e:
        logger.error(f"Ошибка логирования: {e}")

def get_learning_stats():
    """Получает статистику 'обучения'"""
    try:
        if not os.path.exists('rubin_learning_log.json'):
            return {'total_questions': 0, 'successful_answers': 0}
        
        total = 0
        successful = 0
        
        with open('rubin_learning_log.json', 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    total += 1
                    if entry.get('success', True):
                        successful += 1
        
        return {
            'total_questions': total,
            'successful_answers': successful,
            'success_rate': f"{(successful/total*100):.1f}%" if total > 0 else "0%"
        }
    except Exception as e:
        logger.error(f"Ошибка получения статистики: {e}")
        return {'total_questions': 0, 'successful_answers': 0}

def show_thinking_process(message):
    """Показывает процесс 'мышления' Rubin AI"""
    steps = []
    
    # Шаг 1: Анализ входного текста
    steps.append(f"🔍 **Шаг 1:** Получил вопрос: '{message}'")
    
    # Шаг 2: Нормализация
    normalized = message.lower().strip()
    steps.append(f"📝 **Шаг 2:** Нормализовал текст: '{normalized}'")
    
    # Шаг 3: Поиск ключевых слов
    keywords_found = []
    math_keywords = ['транзистор', 'диод', 'резистор', 'скорость', 'угол', '+', '-', '*', '/']
    for keyword in math_keywords:
        if keyword in normalized:
            keywords_found.append(keyword)
    
    steps.append(f"🎯 **Шаг 3:** Найдены ключевые слова: {keywords_found}")
    
    # Шаг 4: Определение категории
    if any(word in normalized for word in ['транзистор', 'диод', 'резистор']):
        category = 'электроника'
    elif any(word in normalized for word in ['скорость', 'путь', 'время']):
        category = 'физика'
    elif any(word in normalized for word in ['+', '-', '*', '/']):
        category = 'математика'
    elif 'угол' in normalized:
        category = 'геометрия'
    else:
        category = 'общие вопросы'
    
    steps.append(f"📂 **Шаг 4:** Определил категорию: '{category}'")
    
    # Шаг 5: Выбор алгоритма
    if category == 'математика':
        algorithm = 'Арифметические вычисления'
    elif category == 'физика':
        algorithm = 'Формулы движения'
    elif category == 'электроника':
        algorithm = 'Техническая база знаний'
    else:
        algorithm = 'Поиск в статических ответах'
    
    steps.append(f"⚙️ **Шаг 5:** Выбрал алгоритм: '{algorithm}'")
    steps.append(f"✅ **Шаг 6:** Генерирую ответ по выбранному правилу")
    
    return "\n".join(steps)

def solve_math(message):
    """Решает математические задачи"""
    message = message.strip()
    
    # Простая арифметика: 2+4, 5*3, 10-2, 8/2
    pattern = r'^(\d+)\s*([+\-*/])\s*(\d+)$'
    match = re.match(pattern, message)
    
    if match:
        num1 = int(match.group(1))
        op = match.group(2)
        num2 = int(match.group(3))
        
        if op == '+':
            result = num1 + num2
            operation = "сложение"
        elif op == '-':
            result = num1 - num2
            operation = "вычитание"
        elif op == '*':
            result = num1 * num2
            operation = "умножение"
        elif op == '/':
            if num2 == 0:
                return "❌ Деление на ноль невозможно!"
            result = num1 / num2
            operation = "деление"
        
        return f"""🧮 **Решение:**

**Задача:** {num1} {op} {num2}
**Операция:** {operation}
**Ответ:** {result}

✅ **Результат: {result}**"""
    
    # Квадратные уравнения: x² - 5x + 6 = 0, x^2 + 3x - 4 = 0
    quadratic_pattern = r'x[²²^2]\s*([+\-]?)\s*(\d*)\s*x\s*([+\-]?)\s*(\d*)\s*=\s*(\d+)'
    quadratic_match = re.search(quadratic_pattern, message)
    
    if quadratic_match:
        try:
            # Извлекаем коэффициенты
            a_sign = quadratic_match.group(1) or '+'
            b_coeff = quadratic_match.group(2) or '1'
            c_sign = quadratic_match.group(3) or '+'
            c_coeff = quadratic_match.group(4) or '0'
            right_side = int(quadratic_match.group(5))
            
            # Преобразуем в стандартную форму ax² + bx + c = 0
            a = 1  # коэффициент при x² всегда 1 в нашем случае
            b = int(a_sign + b_coeff) if b_coeff else 0
            c = int(c_sign + c_coeff) if c_coeff else 0
            
            # Переносим правую часть влево
            c = c - right_side
            
            # Вычисляем дискриминант
            discriminant = b**2 - 4*a*c
            
            if discriminant > 0:
                # Два различных корня
                x1 = (-b + discriminant**0.5) / (2*a)
                x2 = (-b - discriminant**0.5) / (2*a)
                
                return f"""📐 **Решение квадратного уравнения:**
                
**Уравнение:** x² {f'+ {b}x' if b > 0 else f'- {abs(b)}x' if b < 0 else ''} {f'+ {c}' if c > 0 else f'- {abs(c)}' if c < 0 else ''} = 0
**Коэффициенты:** a = {a}, b = {b}, c = {c}
**Дискриминант:** D = b² - 4ac = {b}² - 4×{a}×{c} = {discriminant}

**Корни:**
• x₁ = (-b + √D) / 2a = (-{b} + √{discriminant}) / 2 = {x1:.2f}
• x₂ = (-b - √D) / 2a = (-{b} - √{discriminant}) / 2 = {x2:.2f}

✅ **Ответ: x₁ = {x1:.2f}, x₂ = {x2:.2f}**"""
            
            elif discriminant == 0:
                # Один корень
                x = -b / (2*a)
                
                return f"""📐 **Решение квадратного уравнения:**
                
**Уравнение:** x² {f'+ {b}x' if b > 0 else f'- {abs(b)}x' if b < 0 else ''} {f'+ {c}' if c > 0 else f'- {abs(c)}' if c < 0 else ''} = 0
**Коэффициенты:** a = {a}, b = {b}, c = {c}
**Дискриминант:** D = b² - 4ac = {b}² - 4×{a}×{c} = 0

**Корень:** x = -b / 2a = -{b} / 2 = {x:.2f}

✅ **Ответ: x = {x:.2f}**"""
            
            else:
                # Нет действительных корней
                return f"""📐 **Решение квадратного уравнения:**
                
**Уравнение:** x² {f'+ {b}x' if b > 0 else f'- {abs(b)}x' if b < 0 else ''} {f'+ {c}' if c > 0 else f'- {abs(c)}' if c < 0 else ''} = 0
**Коэффициенты:** a = {a}, b = {b}, c = {c}
**Дискриминант:** D = b² - 4ac = {b}² - 4×{a}×{c} = {discriminant} < 0

❌ **Ответ: Уравнение не имеет действительных корней**"""
                
        except Exception as e:
            logger.error(f"Ошибка решения квадратного уравнения: {e}")
    
    # Текстовые задачи с яблоками
    if 'яблок' in message.lower() and ('осталось' in message.lower() or 'укатилось' in message.lower()):
        numbers = re.findall(r'\d+', message)
        if len(numbers) >= 2:
            total = int(numbers[0])
            taken = int(numbers[1])
            remaining = total - taken
            return f"""🍎 **Решение задачи:**

**Было:** {total} яблок
**Взяли/укатилось:** {taken}
**Осталось:** {total} - {taken} = {remaining}

✅ **Ответ: {remaining} яблок{'а' if remaining > 1 else 'о'}**"""
    
    # Геометрия - деление углов
    if ('луч' in message.lower() and 'делит' in message.lower() and 'угол' in message.lower()) or ('больше другого' in message.lower() and 'угол' in message.lower()):
        # Ищем числа в задаче
        numbers = re.findall(r'\d+', message)
        
        if len(numbers) >= 2:  # есть разность и хотя бы один угол
            difference = int(numbers[0])  # на сколько больше (90°)
            angles_to_solve = []
            
            # Ищем углы для решения (а) 120; б) 170; в) 180)
            for i in range(1, len(numbers)):
                total_angle = int(numbers[i])
                if total_angle > difference:  # проверяем, что угол больше разности
                    angles_to_solve.append(total_angle)
            
            if angles_to_solve:
                solutions = []
                for total_angle in angles_to_solve:
                    # Составляем уравнение: x + (x + 90) = total_angle
                    # 2x + 90 = total_angle
                    # 2x = total_angle - 90
                    # x = (total_angle - 90) / 2
                    
                    smaller_angle = (total_angle - difference) / 2
                    larger_angle = smaller_angle + difference
                    
                    solutions.append({
                        'total': total_angle,
                        'smaller': smaller_angle,
                        'larger': larger_angle
                    })
                
                result = f"""📐 **Геометрическое решение (деление угла):**

**Условие:** Луч делит угол на два угла, один из которых на {difference}° больше другого.

**Составляем уравнение:**
• Пусть меньший угол = x
• Тогда больший угол = x + {difference}°
• Сумма углов: x + (x + {difference}°) = общий угол
• Упрощаем: 2x + {difference}° = общий угол
• Находим x: x = (общий угол - {difference}°) / 2

**Решения:**
"""
                
                for i, sol in enumerate(solutions):
                    letter = chr(ord('а') + i)  # а, б, в...
                    result += f"""
**{letter}) При угле {sol['total']}°:**
• 2x + {difference}° = {sol['total']}°
• 2x = {sol['total']}° - {difference}° = {sol['total'] - difference}°
• x = {sol['total'] - difference}° / 2 = {sol['smaller']}°
• Меньший угол: {sol['smaller']}°
• Больший угол: {sol['smaller']}° + {difference}° = {sol['larger']}°
• Проверка: {sol['smaller']}° + {sol['larger']}° = {sol['total']}° ✓
"""
                
                result += f"\n✅ **Ответы:** "
                for i, sol in enumerate(solutions):
                    letter = chr(ord('а') + i)
                    result += f"{letter}) {sol['smaller']}° и {sol['larger']}°; "
                
                return result
    
    # Геометрия - смежные углы
    if 'смежными' in message.lower() and 'углы' in message.lower():
        angle_match = re.search(r'(\d+)\s*°?', message)
        if angle_match:
            angle_sum = int(angle_match.group(1))
            if angle_sum == 180:
                return f"""📐 **Геометрическое решение:**

**Сумма углов:** {angle_sum}°
**Правило:** Смежные углы в сумме дают 180°
**Сравнение:** {angle_sum}° = 180°

✅ **Ответ: ДА, углы являются смежными**"""
            else:
                return f"""📐 **Геометрическое решение:**

**Сумма углов:** {angle_sum}°
**Правило:** Смежные углы в сумме дают 180°
**Сравнение:** {angle_sum}° ≠ 180°

✅ **Ответ: НЕТ, углы НЕ являются смежными**"""
    
    # Физика - задачи на движение
    # 1. Расчет скорости (v = s/t)
    if 'скорость' in message.lower() and ('путь' in message.lower() or 'расстояние' in message.lower()):
        distance_match = re.search(r'(\d+)\s*м', message)
        time_match = re.search(r'(\d+)\s*с', message)
        
        if distance_match and time_match:
            distance = int(distance_match.group(1))
            time = int(time_match.group(1))
            velocity_ms = distance / time
            velocity_kmh = velocity_ms * 3.6
            
            return f"""⚡ **Физическое решение:**

**Дано:**
• Расстояние: {distance} м
• Время: {time} с

**Формула:** v = s/t
**Расчет:** v = {distance}/{time} = {velocity_ms:.1f} м/с
**В км/ч:** {velocity_ms:.1f} × 3.6 = {velocity_kmh:.1f} км/ч

✅ **Ответ: {velocity_ms:.1f} м/с ({velocity_kmh:.1f} км/ч)**"""
    
    # 2. Расчет расстояния (s = v × t)
    if ('расстояние' in message.lower() or 'путь' in message.lower() or 'пролетит' in message.lower()) and 'скорость' in message.lower():
        # Ищем скорость в км/ч
        speed_kmh_match = re.search(r'(\d+)\s*км/ч', message)
        # Ищем время в минутах
        time_min_match = re.search(r'(\d+)\s*мин', message)
        
        if speed_kmh_match and time_min_match:
            speed_kmh = int(speed_kmh_match.group(1))
            time_min = int(time_min_match.group(1))
            
            # Переводим время в часы
            time_hours = time_min / 60
            
            # Вычисляем расстояние
            distance_km = speed_kmh * time_hours
            
            return f"""✈️ **Физическое решение (формула пути):**

**Дано:**
• Скорость: {speed_kmh} км/ч
• Время: {time_min} мин = {time_hours:.2f} ч

**Формула пути:** s = v × t
**Расчет:** s = {speed_kmh} × {time_hours:.2f} = {distance_km:.1f} км

**Пошаговое решение:**
1. Переводим время: {time_min} мин = {time_min}/60 = {time_hours:.2f} ч
2. Применяем формулу: s = v × t
3. Подставляем: s = {speed_kmh} × {time_hours:.2f} = {distance_km:.1f} км

✅ **Ответ: {distance_km:.1f} км**"""
    
    # 3. Расчет времени (t = s/v)
    if 'время' in message.lower() and 'скорость' in message.lower() and ('расстояние' in message.lower() or 'путь' in message.lower()):
        speed_match = re.search(r'(\d+)\s*(км/ч|м/с)', message)
        distance_match = re.search(r'(\d+)\s*(км|м)', message)
        
        if speed_match and distance_match:
            speed = int(speed_match.group(1))
            speed_unit = speed_match.group(2)
            distance = int(distance_match.group(1))
            distance_unit = distance_match.group(2)
            
            # Приводим к одним единицам
            if speed_unit == 'км/ч' and distance_unit == 'км':
                time_hours = distance / speed
                time_min = time_hours * 60
                
                return f"""⏱️ **Физическое решение (расчет времени):**

**Дано:**
• Скорость: {speed} км/ч
• Расстояние: {distance} км

**Формула:** t = s/v
**Расчет:** t = {distance}/{speed} = {time_hours:.2f} ч = {time_min:.0f} мин

✅ **Ответ: {time_hours:.2f} ч ({time_min:.0f} мин)**"""
    
    # Задачи с деревьями в саду
    if 'саду' in message.lower() and 'деревьев' in message.lower():
        numbers = re.findall(r'\d+', message)
        if len(numbers) >= 3:
            total = int(numbers[0])
            apples = int(numbers[1])
            pears = int(numbers[2])
            plums = total - apples - pears
            
            return f"""🌳 **Решение задачи:**

**Всего деревьев:** {total}
**Яблонь:** {apples}
**Груш:** {pears}
**Слив:** {total} - {apples} - {pears} = {plums}

✅ **Ответ: {plums} слив**"""
    
    # Задачи с улиткой (многодневные задачи)
    if 'улитка' in message.lower() and ('день' in message.lower() or 'дня' in message.lower()):
        numbers = re.findall(r'\d+', message)
        if len(numbers) >= 2:
            day1 = int(numbers[0])  # первый день
            increase = int(numbers[1])  # на сколько больше во второй день
            
            day2 = day1 + increase  # второй день
            day3 = day1 + day2      # третий день = первый + второй
            
            return f"""🐌 **Решение задачи с улиткой:**

**Первый день:** {day1} м
**Второй день:** {day1} + {increase} = {day2} м
**Третий день:** {day1} + {day2} = {day3} м

**Пошаговое решение:**
1. Первый день: {day1} м (дано)
2. Второй день: {day1} + {increase} = {day2} м
3. Третий день: столько же, сколько в первые два дня вместе
4. Третий день: {day1} + {day2} = {day3} м

✅ **Ответ: {day3} метров**"""
    
    # Общие многодневные задачи
    if ('первый день' in message.lower() or 'во второй' in message.lower()) and 'третий' in message.lower():
        numbers = re.findall(r'\d+', message)
        if len(numbers) >= 2:
            day1 = int(numbers[0])
            increase = int(numbers[1])
            day2 = day1 + increase
            day3 = day1 + day2
            
            return f"""📅 **Решение многодневной задачи:**

**День 1:** {day1}
**День 2:** {day1} + {increase} = {day2}
**День 3:** {day1} + {day2} = {day3}

✅ **Ответ: {day3}**"""
    
    # Электротехника - основные компоненты и темы
    if any(word in message.lower() for word in ['транзистор', 'диод', 'резистор', 'конденсатор', 'катушка', 'индуктивность', 'защита', 'короткое замыкание', 'предохранитель', 'автомат']):
        
        if 'транзистор' in message.lower():
            return f"""🔌 **Электротехника - Транзистор:**

**Что такое транзистор:**
Транзистор - это полупроводниковый прибор с тремя выводами, предназначенный для усиления и переключения электрических сигналов.

**Основные типы:**
• **Биполярные (BJT)** - управляются током
  - NPN транзисторы
  - PNP транзисторы
• **Полевые (FET)** - управляются напряжением
  - JFET (с p-n переходом)
  - MOSFET (с изолированным затвором)

**Выводы транзистора:**
• **База (B)** - управляющий электрод
• **Коллектор (C)** - выход усиленного сигнала
• **Эмиттер (E)** - общий электрод

**Принцип работы:**
1. Малый ток базы управляет большим током коллектора
2. Коэффициент усиления β = Ic/Ib
3. Может работать в режимах: усиления, насыщения, отсечки

**Применение:**
• Усилители сигналов
• Электронные ключи
• Генераторы колебаний
• Стабилизаторы напряжения

✅ **Транзистор - основа современной электроники!**"""
        
        elif 'диод' in message.lower():
            return f"""🔌 **Электротехника - Диод:**

**Что такое диод:**
Диод - это полупроводниковый прибор с двумя выводами, пропускающий ток только в одном направлении.

**Конструкция:**
• **Анод (+)** - положительный вывод
• **Катод (-)** - отрицательный вывод
• **p-n переход** - граница между областями

**Принцип работы:**
• **Прямое включение** - диод открыт, ток проходит
• **Обратное включение** - диод закрыт, ток не проходит
• **Пороговое напряжение** - ~0.7В для кремниевых диодов

**Основные типы:**
• **Выпрямительные** - для преобразования AC в DC
• **Стабилитроны** - для стабилизации напряжения
• **Светодиоды (LED)** - излучают свет
• **Фотодиоды** - преобразуют свет в ток
• **Варикапы** - переменная емкость

**Применение:**
• Выпрямители в блоках питания
• Защита от обратной полярности
• Ограничители напряжения
• Индикация и освещение (LED)

**Основные параметры:**
• Прямое напряжение (Vf)
• Максимальный прямой ток (If max)
• Обратное напряжение (Vr max)

✅ **Диод - основа выпрямления и защиты цепей!**"""
        
        elif 'резистор' in message.lower():
            return f"""🔌 **Электротехника - Резистор:**

**Что такое резистор:**
Резистор - это пассивный элемент, ограничивающий ток в электрической цепи.

**Основной закон:**
• **Закон Ома:** U = I × R
• **Мощность:** P = I² × R = U²/R

**Типы резисторов:**
• **Постоянные** - фиксированное сопротивление
• **Переменные** - регулируемое сопротивление
• **Подстроечные** - для точной настройки

**Материалы:**
• **Углеродные** - дешевые, низкая точность
• **Металлопленочные** - высокая точность
• **Проволочные** - высокая мощность

**Цветовая маркировка:**
• 4 полосы: сопротивление + точность
• 5 полос: повышенная точность
• 6 полос: + температурный коэффициент

**Применение:**
• Ограничение тока
• Деление напряжения
• Нагрузка в цепях
• Времязадающие цепи (с конденсаторами)

✅ **Резистор - основа управления током!**"""
        
        elif 'защита' in message.lower() and ('короткое замыкание' in message.lower() or 'замыкание' in message.lower()):
            return f"""🛡️ **Защита от короткого замыкания:**

**Что такое короткое замыкание:**
Соединение двух точек цепи с разными потенциалами через очень малое сопротивление, что приводит к резкому увеличению тока.

**Опасности:**
• **Перегрев проводников** - возможен пожар
• **Повреждение оборудования** - выход из строя
• **Поражение электрическим током** - опасность для людей
• **Дуговые разряды** - взрывоопасность

**Средства защиты:**

**1. Предохранители:**
• **Плавкие вставки** - одноразовые, точные
• **Время срабатывания** - мгновенное при КЗ
• **Номиналы** - от мА до кА
• **Применение** - бытовые и промышленные сети

**2. Автоматические выключатели:**
• **Тепловая защита** - от перегрузки
• **Электромагнитная защита** - от КЗ
• **Многократное использование** - можно включать заново
• **Характеристики** - B, C, D (время-токовые)

**3. Дифференциальные автоматы (УЗО):**
• **Защита от утечек** - ток утечки на землю
• **Защита людей** - от поражения током
• **Чувствительность** - 10мА, 30мА, 100мА, 300мА
• **Время срабатывания** - < 30мс

**4. Реле защиты:**
• **Максимально-токовая защита** - от перегрузки
• **Дистанционная защита** - по сопротивлению
• **Дифференциальная защита** - сравнение токов
• **Направленная защита** - по направлению мощности

**5. Ограничители тока:**
• **PTC термисторы** - саморегулирующиеся
• **Токоограничивающие реакторы** - индуктивное сопротивление
• **Электронные ограничители** - активное управление

**Расчет защиты:**
• **Iном.защиты ≥ Iраб.макс** (рабочий ток)
• **Iном.защиты ≤ Iдоп.провода** (допустимый ток провода)
• **Время срабатывания** - зависит от характера нагрузки

**Селективность защиты:**
Защиты должны срабатывать в определенной последовательности - сначала ближайшая к месту КЗ.

✅ **Правильная защита = безопасность и надежность!**"""
    
    # Электротехника - коэффициент мощности
    if 'коэффициент мощности' in message.lower() or 'cos φ' in message.lower():
        return f"""⚡ **Электротехника - Коэффициент мощности:**

**Что такое коэффициент мощности (cos φ):**
Коэффициент мощности - это отношение активной мощности к полной мощности в цепи переменного тока.

**Формула:**
• **cos φ = P / S**
• **P** - активная мощность (Вт)
• **S** - полная мощность (ВА)
• **φ** - угол сдвига фаз между током и напряжением

**Типы мощности:**
• **Активная (P)** - полезная мощность, Вт
• **Реактивная (Q)** - "бесполезная" мощность, ВАр
• **Полная (S)** - общая мощность, ВА
• **S² = P² + Q²** (треугольник мощностей)

**Значения cos φ:**
• **cos φ = 1** - идеальная нагрузка (только активная)
• **cos φ = 0.9-0.95** - хорошо
• **cos φ = 0.7-0.9** - удовлетворительно
• **cos φ < 0.7** - плохо, требует коррекции

**Причины низкого cos φ:**
• Асинхронные двигатели без нагрузки
• Трансформаторы на холостом ходу
• Люминесцентные лампы
• Сварочные аппараты
• Индукционные печи

**Как улучшить cos φ:**

**1. Конденсаторные батареи:**
• Подключение параллельно нагрузке
• Компенсация реактивной мощности
• Автоматическое управление

**2. Синхронные двигатели:**
• Работа с опережающим током
• Компенсация отстающего тока асинхронных машин

**3. Статические компенсаторы:**
• Тиристорные устройства
• Быстрое регулирование

**4. Правильная эксплуатация:**
• Загрузка двигателей на 75-85%
• Отключение ненагруженного оборудования
• Замена старых двигателей на энергоэффективные

**Преимущества высокого cos φ:**
• Снижение потерь в сети
• Уменьшение тока в проводах
• Экономия электроэнергии
• Снижение нагрузки на трансформаторы
• Улучшение качества напряжения

**Расчет компенсации:**
• **Qc = P × (tg φ1 - tg φ2)**
• **C = Qc / (2π × f × U²)**

✅ **Высокий cos φ = экономия и качество!**"""
    
    # Электротехника - законы и формулы
    if any(phrase in message.lower() for phrase in ['закон ома', 'сопротивление', 'напряжение', 'ток']):
        return f"""⚡ **Электротехника - Основные законы:**

**Закон Ома:**
• **U = I × R** (напряжение = ток × сопротивление)
• **I = U / R** (ток = напряжение / сопротивление)
• **R = U / I** (сопротивление = напряжение / ток)

**Мощность:**
• **P = U × I** (мощность = напряжение × ток)
• **P = I² × R** (через ток и сопротивление)
• **P = U² / R** (через напряжение и сопротивление)

**Законы Кирхгофа:**
• **1-й закон:** Сумма токов в узле = 0
• **2-й закон:** Сумма напряжений в контуре = 0

**Соединения резисторов:**
• **Последовательное:** R = R1 + R2 + R3...
• **Параллельное:** 1/R = 1/R1 + 1/R2 + 1/R3...

**Единицы измерения:**
• Напряжение: Вольт (В)
• Ток: Ампер (А)
• Сопротивление: Ом (Ω)
• Мощность: Ватт (Вт)

✅ **Основа всей электротехники!**"""
    
    # Программирование - сравнение языков
    if any(phrase in message.lower() for phrase in ['c++', 'python', 'программирование', 'автоматизация', 'сравни']):
        if 'c++' in message.lower() and 'python' in message.lower():
            return f"""💻 **Программирование - C++ vs Python для автоматизации:**

**C++ для промышленной автоматизации:**

**Преимущества:**
• **Высокая производительность** - компилируемый язык
• **Низкоуровневый доступ** - работа с железом, регистрами
• **Детерминированность** - предсказуемое время выполнения
• **Малое потребление памяти** - важно для встраиваемых систем
• **Реальное время** - жесткие временные требования
• **Интеграция с PLC** - легкая работа с промышленными протоколами

**Недостатки:**
• **Сложность разработки** - много кода для простых задач
• **Время разработки** - долгая отладка
• **Управление памятью** - риск утечек и ошибок
• **Портируемость** - зависимость от платформы

**Применение C++:**
• Драйверы устройств
• Системы реального времени
• ПЛК программирование
• Встраиваемые системы
• Критичные по времени задачи

---

**Python для промышленной автоматизации:**

**Преимущества:**
• **Быстрая разработка** - простой синтаксис
• **Богатые библиотеки** - NumPy, Pandas, Matplotlib
• **Простота изучения** - низкий порог входа
• **Интерпретируемость** - быстрое тестирование
• **Кроссплатформенность** - работает везде
• **Анализ данных** - отличные инструменты

**Недостатки:**
• **Низкая производительность** - интерпретируемый язык
• **Потребление памяти** - больше чем C++
• **GIL ограничения** - проблемы с многопоточностью
• **Не для реального времени** - непредсказуемые задержки

**Применение Python:**
• SCADA системы
• Анализ данных с датчиков
• Веб-интерфейсы для управления
• Прототипирование алгоритмов
• Машинное обучение в производстве

---

**Рекомендации по выбору:**

**Выбирайте C++ для:**
• Управления двигателями и приводами
• Обработки сигналов в реальном времени
• Программирования микроконтроллеров
• Критичных по безопасности систем
• Высокочастотных измерений

**Выбирайте Python для:**
• Мониторинга и диспетчеризации
• Анализа производственных данных
• Создания отчетов и графиков
• Интеграции различных систем
• Машинного обучения для предиктивного обслуживания

**Гибридный подход:**
• **C++** - критичные компоненты управления
• **Python** - анализ данных и интерфейсы
• **Связь через** - TCP/IP, REST API, shared memory

**Популярные комбинации:**
• C++ + Python через pybind11
• C++ для PLC, Python для HMI
• C++ драйверы + Python приложения

✅ **Лучший выбор зависит от конкретной задачи!**"""
    
    # Дополнительные темы
    if 'коэффициент мощности' in message.lower():
        return f"""⚡ **Коэффициент мощности (cos φ):**

**Что это:**
Коэффициент мощности показывает эффективность использования электроэнергии.

**Формула:** cos φ = P / S
• P - активная мощность (Вт)  
• S - полная мощность (ВА)

**Как улучшить:**
1. **Конденсаторы** - компенсация реактивной мощности
2. **Синхронные двигатели** - работа с опережающим cos φ
3. **Активные фильтры** - для нелинейных нагрузок
4. **Правильный выбор оборудования**

✅ **Хороший cos φ = экономия электроэнергии!**"""
    
    # Программирование - алгоритмы управления
    if any(phrase in message.lower() for phrase in ['алгоритм', 'конвейер', 'управление', 'python', 'программирование']):
        if 'конвейер' in message.lower() and 'python' in message.lower():
            return f"""🏭 **Алгоритм управления конвейером на Python:**

```python
import time
import threading
from enum import Enum

class ConveyorState(Enum):
    STOPPED = 0
    RUNNING = 1
    ERROR = 2
    MAINTENANCE = 3

class ConveyorController:
    def __init__(self):
        self.state = ConveyorState.STOPPED
        self.speed = 0  # м/мин
        self.items_count = 0
        self.emergency_stop = False
        self.sensors = {{
            'start_button': False,
            'stop_button': False,
            'emergency': False,
            'overload': False,
            'item_detected': False
        }}
    
    def start_conveyor(self):
        \"\"\"Запуск конвейера\"\"\"
        if self.emergency_stop:
            print("❌ Аварийная остановка активна!")
            return False
        
        if self.sensors['overload']:
            print("❌ Перегрузка конвейера!")
            return False
        
        self.state = ConveyorState.RUNNING
        self.speed = 10  # м/мин
        print("✅ Конвейер запущен")
        return True
    
    def stop_conveyor(self):
        \"\"\"Остановка конвейера\"\"\"
        self.state = ConveyorState.STOPPED
        self.speed = 0
        print("🛑 Конвейер остановлен")
    
    def emergency_stop_conveyor(self):
        \"\"\"Аварийная остановка\"\"\"
        self.emergency_stop = True
        self.state = ConveyorState.ERROR
        self.speed = 0
        print("🚨 АВАРИЙНАЯ ОСТАНОВКА!")
    
    def monitor_sensors(self):
        \"\"\"Мониторинг датчиков\"\"\"
        while True:
            # Проверка аварийной кнопки
            if self.sensors['emergency']:
                self.emergency_stop_conveyor()
            
            # Проверка перегрузки
            if self.sensors['overload'] and self.state == ConveyorState.RUNNING:
                print("⚠️ Перегрузка! Снижаем скорость")
                self.speed = max(5, self.speed - 1)
            
            # Подсчет предметов
            if self.sensors['item_detected']:
                self.items_count += 1
                print(f"📦 Предмет #{self.items_count} обнаружен")
            
            time.sleep(0.1)  # 100мс цикл
    
    def get_status(self):
        \"\"\"Получить статус системы\"\"\"
        return {{
            'state': self.state.name,
            'speed': self.speed,
            'items_processed': self.items_count,
            'emergency_active': self.emergency_stop
        }}

# Использование:
conveyor = ConveyorController()

# Запуск мониторинга в отдельном потоке
monitor_thread = threading.Thread(target=conveyor.monitor_sensors)
monitor_thread.daemon = True
monitor_thread.start()

# Управление конвейером
conveyor.start_conveyor()
time.sleep(5)
conveyor.stop_conveyor()
```

**Ключевые особенности:**
• **Машина состояний** - четкие режимы работы
• **Безопасность** - проверки перед запуском
• **Мониторинг** - постоянный контроль датчиков
• **Многопоточность** - параллельная обработка
• **Логирование** - отслеживание событий

✅ **Готовый алгоритм для промышленного применения!**"""
    
    # Протоколы связи
    if 'modbus' in message.lower() and 'rtu' in message.lower():
        return f"""📡 **Протокол Modbus RTU:**

**Что это:**
Modbus RTU - промышленный протокол связи для обмена данными между устройствами автоматизации.

**Основные характеристики:**
• **RTU** - Remote Terminal Unit (удаленный терминал)
• **Последовательная передача** - RS-485/RS-232
• **Мастер-слейв архитектура** - один ведущий, много ведомых
• **Адресация** - устройства 1-247 (0 - широковещательный)

**Структура кадра:**
```
[Адрес][Функция][Данные][CRC16]
  1 байт  1 байт  N байт  2 байта
```

**Основные функции:**
• **01** - Чтение дискретных выходов (Coils)
• **02** - Чтение дискретных входов (Discrete Inputs)
• **03** - Чтение регистров хранения (Holding Registers)
• **04** - Чтение входных регистров (Input Registers)
• **05** - Запись одного выхода (Single Coil)
• **06** - Запись одного регистра (Single Register)
• **15** - Запись множественных выходов
• **16** - Запись множественных регистров

**Пример запроса (чтение 10 регистров с адреса 0):**
```
Адрес устройства: 01
Код функции: 03
Начальный адрес: 00 00
Количество регистров: 00 0A
CRC16: C5 CD
```

**Преимущества:**
• Простота реализации
• Широкая поддержка
• Надежность передачи (CRC)
• Стандартизация

**Применение:**
• PLC системы
• SCADA системы
• Промышленные сети
• Системы мониторинга

✅ **Modbus RTU - стандарт промышленной автоматизации!**"""
    
    # Закон Ома для полной цепи
    if 'закон ома' in message.lower() and ('полной цепи' in message.lower() or 'полная цепь' in message.lower()):
        return f"""⚡ **Закон Ома для полной цепи:**

**Формула:**
**I = E / (R + r)**

Где:
• **I** - ток в цепи (А)
• **E** - ЭДС источника (В)
• **R** - внешнее сопротивление (Ом)
• **r** - внутреннее сопротивление источника (Ом)

**Физический смысл:**
Ток в замкнутой цепи равен ЭДС источника, деленной на полное сопротивление цепи.

**Напряжение на нагрузке:**
**U = I × R = E × R / (R + r)**

**Мощность в нагрузке:**
**P = I² × R = E² × R / (R + r)²**

**Максимальная мощность:**
Передается в нагрузку при **R = r** (согласование сопротивлений)

**КПД цепи:**
**η = R / (R + r) × 100%**

**Практические примеры:**

**1. Батарейка (E=1.5В, r=1Ом) + лампочка (R=9Ом):**
• I = 1.5 / (9 + 1) = 0.15 А
• U на лампочке = 0.15 × 9 = 1.35 В
• КПД = 9/10 = 90%

**2. Генератор (E=220В, r=0.5Ом) + нагрузка (R=10Ом):**
• I = 220 / (10 + 0.5) = 20.95 А
• U на нагрузке = 20.95 × 10 = 209.5 В
• Потери = 20.95² × 0.5 = 219 Вт

**Важные выводы:**
• При r << R: U ≈ E (идеальный источник)
• При r >> R: большие потери в источнике
• Внутреннее сопротивление снижает КПД

✅ **Основа расчета любой электрической цепи!**"""
    
    if ('c++' in message.lower() and 'python' in message.lower()) or 'сравни' in message.lower():
        return f"""💻 **C++ vs Python для автоматизации:**

**C++ лучше для:**
• Систем реального времени
• Работы с PLC/контроллерами  
• Критичных по времени задач
• Встраиваемых систем

**Python лучше для:**
• Анализа данных
• SCADA систем
• Прототипирования
• Веб-интерфейсов

**Вывод:** Используйте оба! C++ для "железа", Python для анализа."""
    
    return None

@app.route('/')
def index():
    return jsonify({
        'name': 'Rubin AI',
        'version': '2.0',
        'status': 'online',
        'features': ['mathematics', 'physics', 'geometry', 'word_problems']
    })

@app.route('/api/health')
@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'timestamp': '2025-09-18'})

@app.route('/api/stats')
def stats():
    learning_stats = get_learning_stats()
    return jsonify({
        'system': {
            'name': 'Rubin AI',
            'version': '2.0',
            'status': 'running'
        },
        'modules': {
            'math_solver': 'active',
            'chat': 'active',
            'learning_system': 'active'
        },
        'learning': learning_stats
    })

@app.route('/api/learning-stats')
def learning_stats():
    """Endpoint для получения статистики обучения"""
    stats = get_learning_stats()
    
    # Добавляем статистику нейронной сети
    neural_stats = {}
    if NEURAL_NETWORK_AVAILABLE:
        try:
            neural_ai = get_neural_rubin()
            neural_stats = neural_ai.get_neural_stats()
        except Exception as e:
            neural_stats = {'error': str(e)}
    
    return jsonify({
        'message': f'''📊 **Статистика обучения Rubin AI:**

**Обычная статистика:**
• Всего вопросов: {stats['total_questions']}
• Успешных ответов: {stats['successful_answers']}
• Процент успеха: {stats.get('success_rate', '0%')}

**Нейронная сеть:**
• Статус: {'🧠 Активна' if NEURAL_NETWORK_AVAILABLE else '❌ Недоступна'}
• Устройство: {neural_stats.get('device', 'N/A')}
• Параметров в модели: {neural_stats.get('model_parameters', 0):,}
• Диалогов через нейросеть: {neural_stats.get('conversation_count', 0)}

**Возможности:**
• Классификация вопросов
• Обучение на обратной связи
• Генерация ответов
• Сохранение/загрузка модели''',
        'stats': stats,
        'neural_stats': neural_stats,
        'neural_available': NEURAL_NETWORK_AVAILABLE
    })

@app.route('/api/test', methods=['GET'])
def test_endpoint():
    """Тестовый endpoint"""
    return jsonify({'message': 'Тестовый endpoint работает!', 'timestamp': datetime.now().isoformat()})

@app.route('/api/chat', methods=['POST'])
def chat():
    """Основной endpoint для чата с Rubin AI"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'error': 'Сообщение не может быть пустым'}), 400
        
        # Логируем вопрос
        log_conversation(message, "", False)
        
        # Показываем процесс мышления
        thinking = show_thinking_process(message)
        
        # Пытаемся решить математическую задачу
        math_result = solve_math(message)
        
        if math_result:
            # Логируем успешный ответ
            log_conversation(message, math_result, True)
            return jsonify({
                'success': True,
                'response': math_result,
                'thinking_process': thinking,
                'category': 'mathematics'
            })
        
        # Если не математика, используем улучшенный диспетчер
        if ENHANCED_DISPATCHER_AVAILABLE and enhanced_dispatcher:
            try:
                response, module = enhanced_dispatcher.route_request(message, {'message': message})
                if response:
                    log_conversation(message, str(response), True)
                    return jsonify({
                        'success': True,
                        'response': response,
                        'thinking_process': thinking,
                        'category': module,
                        'module': module
                    })
            except Exception as e:
                logger.error(f"Ошибка диспетчера: {e}")
        
        # Fallback ответ
        fallback_response = f"""🤖 **Rubin AI отвечает:**

Я получил ваш вопрос: "{message}"

К сожалению, я пока не могу дать полный ответ на этот вопрос. 
Попробуйте переформулировать вопрос или задать вопрос по:
• Математике (арифметика, геометрия)
• Физике (движение, скорость)
• Электротехнике (резисторы, транзисторы)

💡 **Совет:** Используйте конкретные термины для лучшего понимания."""
        
        log_conversation(message, fallback_response, False)
        return jsonify({
            'success': True,
            'response': fallback_response,
            'thinking_process': thinking,
            'category': 'general'
        })
        
    except Exception as e:
        logger.error(f"Ошибка в чате: {e}")
        return jsonify({'error': f'Внутренняя ошибка сервера: {str(e)}'}), 500

@app.route('/api/neural-feedback', methods=['POST'])
def neural_feedback():
    """Endpoint для обратной связи нейронной сети"""
    if not NEURAL_NETWORK_AVAILABLE:
        return jsonify({'error': 'Нейронная сеть недоступна'}), 400
    
    try:
        data = request.get_json()
        question = data.get('question', '')
        correct_category = data.get('correct_category', '')
        user_rating = data.get('rating', 3)  # 1-5
        
        neural_ai = get_neural_rubin()
        success = neural_ai.learn_from_feedback(question, correct_category, user_rating)
        
        if success:
            neural_ai.save_model()  # Сохраняем обновленную модель
            
        return jsonify({
            'success': success,
            'message': 'Спасибо за обратную связь! Нейронная сеть обучается.' if success else 'Ошибка обучения'
        })
        
    except Exception as e:
        return jsonify({'error': f'Ошибка обратной связи: {str(e)}'}), 500

@app.route('/RubinIDE.html')
def rubin_ide():
    try:
        with open('RubinIDE.html', 'r', encoding='utf-8') as f:
            return f.read(), 200, {'Content-Type': 'text/html; charset=utf-8'}
    except FileNotFoundError:
        return jsonify({'error': 'RubinIDE.html not found'}), 404



if __name__ == '__main__':
    print("Rubin AI (Legacy Server) запущен")
    print("URL: http://localhost:8084")
    app.run(host='0.0.0.0', port=8084, debug=False, threaded=True)