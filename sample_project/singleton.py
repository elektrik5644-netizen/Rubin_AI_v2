
"""
������ ���������� �������� Singleton
"""

class DatabaseConnection:
    """Singleton ��� ����������� � ���� ������"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseConnection, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.connection_string = "sqlite:///example.db"
            self._initialized = True
    
    def connect(self):
        """����������� � ���� ������"""
        print(f"����������� � {self.connection_string}")
        return True

# ������ �������������
if __name__ == "__main__":
    db1 = DatabaseConnection()
    db2 = DatabaseConnection()
    
    print("db1 is db2:", db1 is db2)  # True
    db1.connect()
