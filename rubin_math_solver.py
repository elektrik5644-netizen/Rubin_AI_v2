#!/usr/bin/env python3
"""
Улучшенный математический решатель для Rubin AI
"""

import re
import math

class RubinMathSolver:
    """Улучшенный математический решатель"""
    
    def __init__(self):
        self.patterns = {
            # Арифметические операции
            'arithmetic': r'(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)',
            
            # Задачи на движение
            'motion': r'(?:скорость|скоростью|со скоростью)\s*(\d+(?:\.\d+)?)\s*(?:км/ч|км/час)',
            'time': r'(?:за|через|время)\s*(\d+(?:\.\d+)?)\s*(?:мин|минут|час|часов|ч)',
            'distance': r'(?:расстояние|путь|проехал|прошел)\s*(\d+(?:\.\d+)?)\s*(?:км|метров|м)',
            
            # Задачи на углы
            'angle': r'угол\s*([А-Я]+)\s*равен\s*(\d+(?:\.\d+)?)°',
            'angle_division': r'луч\s*([А-Я]+)\s*делит\s*угол\s*([А-Я]+)\s*на\s*два\s*угла',
            
            # Задачи на количество
            'quantity': r'(\d+)\s*(?:ящик|ящика|ящиков|штук|штуки|штука)',
            'more_less': r'(?:на|больше|меньше)\s*(\d+)\s*(?:ящик|ящика|ящиков|штук|штуки|штука)',
        }
    
    def solve_arithmetic(self, text):
        """Решение арифметических задач"""
        matches = re.findall(self.patterns['arithmetic'], text)
        if not matches:
            return None
        
        results = []
        for match in matches:
            num1 = float(match[0])
            op = match[1]
            num2 = float(match[2])
            
            if op == '+':
                result = num1 + num2
            elif op == '-':
                result = num1 - num2
            elif op == '*':
                result = num1 * num2
            elif op == '/':
                result = num1 / num2 if num2 != 0 else "Деление на ноль!"
            
            results.append(f"{num1} {op} {num2} = {result}")
        
        return "🧮 **Арифметика:**\n" + "\n".join(results)
    
    def solve_motion_problem(self, text):
        """Решение задач на движение"""
        # Ищем скорость
        speed_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:км/ч|км/час)', text)
        if not speed_match:
            return None
        
        speed = float(speed_match.group(1))
        
        # Ищем время
        time_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:мин|минут)', text)
        if time_match:
            time_minutes = float(time_match.group(1))
            time_hours = time_minutes / 60
        else:
            time_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:час|часов|ч)', text)
            if time_match:
                time_hours = float(time_match.group(1))
            else:
                return None
        
        # Вычисляем расстояние
        distance = speed * time_hours
        
        result = f"""🚗 **Задача на движение:**

**Дано:**
• Скорость: {speed} км/ч
• Время: {time_minutes if 'time_minutes' in locals() else time_hours} {'минут' if 'time_minutes' in locals() else 'часов'}

**Решение:**
Формула пути: S = v × t
S = {speed} × {time_hours} = {distance} км

**Ответ:** {distance} км"""
        
        return result
    
    def solve_angle_problem(self, text):
        """Решение задач на углы"""
        # Ищем информацию об углах
        angle_matches = re.findall(self.patterns['angle'], text)
        if not angle_matches:
            return None
        
        # Ищем информацию о делении угла
        division_match = re.search(self.patterns['angle_division'], text)
        if not division_match:
            return None
        
        # Ищем информацию о том, что один угол на 90° больше другого
        difference_match = re.search(r'на\s*(\d+(?:\.\d+)?)°\s*(?:больше|меньше)', text)
        if not difference_match:
            return None
        
        difference = float(difference_match.group(1))
        
        # Ищем значения углов
        angle_values = []
        for match in angle_matches:
            angle_name = match[0]
            angle_value = float(match[1])
            angle_values.append((angle_name, angle_value))
        
        if not angle_values:
            return None
        
        results = []
        for angle_name, total_angle in angle_values:
            # Если один угол на difference° больше другого
            # Пусть x - меньший угол, тогда x + difference - больший угол
            # x + (x + difference) = total_angle
            # 2x + difference = total_angle
            # x = (total_angle - difference) / 2
            
            smaller_angle = (total_angle - difference) / 2
            larger_angle = smaller_angle + difference
            
            results.append(f"""📐 **Угол {angle_name} = {total_angle}°:**

**Дано:**
• Один угол на {difference}° больше другого
• Сумма углов = {total_angle}°

**Решение:**
Пусть x - меньший угол, тогда (x + {difference}) - больший угол
x + (x + {difference}) = {total_angle}
2x + {difference} = {total_angle}
2x = {total_angle - difference}
x = {smaller_angle}

**Ответ:**
• Меньший угол: {smaller_angle}°
• Больший угол: {larger_angle}°""")
        
        return "\n\n".join(results)
    
    def solve_quantity_problem(self, text):
        """Решение задач на количество"""
        # Ищем количество конфет
        candy_match = re.search(r'(\d+)\s*(?:ящик|ящика|ящиков)\s*(?:конфет|конфеты)', text)
        if not candy_match:
            return None
        
        candy_boxes = int(candy_match.group(1))
        
        # Ищем информацию о печенье
        cookie_match = re.search(r'печенья?\s*-\s*на\s*(\d+)\s*(?:ящик|ящика|ящиков)\s*(?:больше|меньше)', text)
        if not cookie_match:
            return None
        
        cookie_difference = int(cookie_match.group(1))
        cookie_boxes = candy_boxes + cookie_difference
        total_boxes = candy_boxes + cookie_boxes
        
        result = f"""📦 **Задача на количество:**

**Дано:**
• Конфет: {candy_boxes} ящиков
• Печенья: на {cookie_difference} ящиков больше

**Решение:**
• Ящиков конфет: {candy_boxes}
• Ящиков печенья: {candy_boxes} + {cookie_difference} = {cookie_boxes}
• Всего ящиков: {candy_boxes} + {cookie_boxes} = {total_boxes}

**Ответ:** {total_boxes} ящиков"""
        
        return result
    
    def solve(self, text):
        """Основной метод решения математических задач"""
        # Пробуем разные типы задач
        solvers = [
            self.solve_arithmetic,
            self.solve_motion_problem,
            self.solve_angle_problem,
            self.solve_quantity_problem
        ]
        
        for solver in solvers:
            result = solver(text)
            if result:
                return result
        
        return None

# Создаем экземпляр решателя
math_solver = RubinMathSolver()

def solve_math_problem(text):
    """Функция для решения математических задач"""
    return math_solver.solve(text)

if __name__ == "__main__":
    # Тестирование
    test_problems = [
        "2+3",
        "5-7",
        "Средняя скорость самолета Ил-14 равна 412 км/ч. Какое расстояние он пролетит за 15 мин?",
        "Луч ОС делит угол АОВ на два угла, один из которых на 90° больше другого. Найдите эти углы, если угол АОВ равен 120°",
        "В магазин привезли 3 ящика конфет, а печенья - на 3 ящика больше. Сколько всего ящиков привезли в магазин?"
    ]
    
    print("🧪 ТЕСТИРОВАНИЕ МАТЕМАТИЧЕСКОГО РЕШАТЕЛЯ")
    print("=" * 50)
    
    for i, problem in enumerate(test_problems, 1):
        print(f"\n{i}. Задача: {problem}")
        result = solve_math_problem(problem)
        if result:
            print(f"✅ Решение найдено:")
            print(result)
        else:
            print("❌ Решение не найдено")

















