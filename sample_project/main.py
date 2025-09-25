
#!/usr/bin/env python3
"""
Пример проекта для демонстрации Rubin AI
"""

class Calculator:
    """Простой калькулятор"""
    
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        """Сложение двух чисел"""
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def multiply(self, a, b):
        """Умножение двух чисел"""
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result
    
    def get_history(self):
        """Получение истории вычислений"""
        return self.history

def main():
    """Главная функция"""
    calc = Calculator()
    
    # Примеры вычислений
    print("Результат сложения:", calc.add(5, 3))
    print("Результат умножения:", calc.multiply(4, 7))
    print("История:", calc.get_history())

if __name__ == "__main__":
    main()
