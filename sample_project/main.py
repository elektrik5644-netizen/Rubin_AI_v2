
#!/usr/bin/env python3
"""
������ ������� ��� ������������ Rubin AI
"""

class Calculator:
    """������� �����������"""
    
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        """�������� ���� �����"""
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def multiply(self, a, b):
        """��������� ���� �����"""
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result
    
    def get_history(self):
        """��������� ������� ����������"""
        return self.history

def main():
    """������� �������"""
    calc = Calculator()
    
    # ������� ����������
    print("��������� ��������:", calc.add(5, 3))
    print("��������� ���������:", calc.multiply(4, 7))
    print("�������:", calc.get_history())

if __name__ == "__main__":
    main()
