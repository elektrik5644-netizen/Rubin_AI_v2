class TestClass:
    """
    Тестовый класс
    """
    
    def __init__(self, value=0):
        """
        Инициализация TestClass
        
        Args:
            value: начальное значение
        """
        self.value = value
    
    def get_value(self):
        """
        Получить значение
        
        Args:
            
        
        Returns:
            int: текущее значение
        """
        return self.value