#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Rubin AI - Система пополнения базы знаний
Автоматическое добавление новых знаний в базу данных
"""

import sqlite3
import json
import requests
import os
import datetime
from typing import Dict, List, Any, Optional

class RubinKnowledgeExpander:
    """Класс для пополнения базы знаний Smart Rubin AI"""
    
    def __init__(self, db_path: str = "rubin_knowledge_base.db"):
        self.db_path = db_path
        self.server_url = "http://localhost:8083"
        
    def add_knowledge_entry(self, title: str, content: str, category: str = "general", 
                          tags: str = "", source: str = "manual", confidence: float = 0.9) -> bool:
        """Добавляет новую запись в базу знаний"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Генерируем уникальный ID
            knowledge_id = f"KNOW_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(title) % 10000}"
            
            # Вставляем запись
            cursor.execute("""
                INSERT INTO knowledge (id, title, content, category, tags, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (knowledge_id, title, content, category, tags, datetime.datetime.now().isoformat()))
            
            conn.commit()
            conn.close()
            
            print(f"✅ Добавлена запись: {title}")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка добавления записи: {e}")
            return False
    
    def add_document(self, title: str, content: str, category: str = "document", 
                    keywords: str = "", file_type: str = "text") -> bool:
        """Добавляет документ в базу документов"""
        try:
            conn = sqlite3.connect("rubin_documents.db")
            cursor = conn.cursor()
            
            # Создаем таблицу если не существует
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT,
                    file_path TEXT,
                    file_type TEXT,
                    file_size INTEGER,
                    content_hash TEXT,
                    content TEXT,
                    metadata TEXT,
                    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 0,
                    tags TEXT,
                    category TEXT,
                    language TEXT DEFAULT 'ru',
                    encoding TEXT DEFAULT 'utf-8'
                )
            """)
            
            # Генерируем хеш контента
            content_hash = str(hash(content))
            
            # Вставляем документ
            cursor.execute("""
                INSERT INTO documents (filename, file_path, file_type, file_size, 
                                     content_hash, content, category, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (title, f"manual/{title}", file_type, len(content), 
                  content_hash, content, category, keywords))
            
            conn.commit()
            conn.close()
            
            print(f"✅ Добавлен документ: {title}")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка добавления документа: {e}")
            return False
    
    def add_formula(self, formula_name: str, formula_text: str, category: str = "mathematics",
                   keywords: str = "", context: str = "", difficulty_level: int = 1) -> bool:
        """Добавляет математическую формулу"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Создаем таблицу формул если не существует
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS formula_index (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    formula_name TEXT,
                    formula_text TEXT,
                    category TEXT,
                    keywords TEXT,
                    context TEXT,
                    difficulty_level INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Вставляем формулу
            cursor.execute("""
                INSERT INTO formula_index (formula_name, formula_text, category, 
                                         keywords, context, difficulty_level)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (formula_name, formula_text, category, keywords, context, difficulty_level))
            
            conn.commit()
            conn.close()
            
            print(f"✅ Добавлена формула: {formula_name}")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка добавления формулы: {e}")
            return False
    
    def add_synonym(self, main_term: str, synonym: str, category: str = "general") -> bool:
        """Добавляет синоним термина"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Создаем таблицу синонимов если не существует
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS term_synonyms (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    main_term TEXT,
                    synonym TEXT,
                    category TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Вставляем синоним
            cursor.execute("""
                INSERT INTO term_synonyms (main_term, synonym, category)
                VALUES (?, ?, ?)
            """, (main_term, synonym, category))
            
            conn.commit()
            conn.close()
            
            print(f"✅ Добавлен синоним: {main_term} -> {synonym}")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка добавления синонима: {e}")
            return False
    
    def load_from_file(self, file_path: str, category: str = "document") -> bool:
        """Загружает знания из файла"""
        try:
            if not os.path.exists(file_path):
                print(f"❌ Файл не найден: {file_path}")
                return False
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            filename = os.path.basename(file_path)
            
            # Добавляем как документ
            success = self.add_document(filename, content, category)
            
            if success:
                print(f"✅ Файл загружен: {filename}")
            
            return success
            
        except Exception as e:
            print(f"❌ Ошибка загрузки файла: {e}")
            return False
    
    def load_mathematics_knowledge(self) -> bool:
        """Загружает математические знания"""
        math_knowledge = [
            {
                "title": "Производная функции",
                "content": """
# ПРОИЗВОДНАЯ ФУНКЦИИ

## Определение:
Производная функции f(x) в точке x₀ - это предел отношения приращения функции к приращению аргумента при стремлении приращения аргумента к нулю.

f'(x₀) = lim[Δx→0] (f(x₀ + Δx) - f(x₀)) / Δx

## Основные правила дифференцирования:

1. **Производная константы:** (C)' = 0
2. **Производная степени:** (xⁿ)' = nxⁿ⁻¹
3. **Производная суммы:** (f + g)' = f' + g'
4. **Производная произведения:** (fg)' = f'g + fg'
5. **Производная частного:** (f/g)' = (f'g - fg') / g²
6. **Производная сложной функции:** (f(g(x)))' = f'(g(x)) · g'(x)

## Примеры:
- (x³)' = 3x²
- (sin x)' = cos x
- (cos x)' = -sin x
- (eˣ)' = eˣ
- (ln x)' = 1/x

## Применение:
- Нахождение экстремумов функции
- Исследование функции на монотонность
- Построение графиков функций
- Решение задач оптимизации
                """,
                "category": "mathematics",
                "tags": "производная, дифференцирование, математика, анализ"
            },
            {
                "title": "Интеграл функции",
                "content": """
# ИНТЕГРАЛ ФУНКЦИИ

## Определение:
Интеграл функции f(x) - это функция F(x), производная которой равна f(x).

∫f(x)dx = F(x) + C, где F'(x) = f(x)

## Основные правила интегрирования:

1. **Интеграл константы:** ∫C dx = Cx + C₁
2. **Интеграл степени:** ∫xⁿ dx = xⁿ⁺¹/(n+1) + C (n ≠ -1)
3. **Интеграл суммы:** ∫(f + g) dx = ∫f dx + ∫g dx
4. **Интеграл произведения на константу:** ∫Cf dx = C∫f dx

## Таблица основных интегралов:
- ∫xⁿ dx = xⁿ⁺¹/(n+1) + C
- ∫1/x dx = ln|x| + C
- ∫eˣ dx = eˣ + C
- ∫sin x dx = -cos x + C
- ∫cos x dx = sin x + C
- ∫1/(1+x²) dx = arctan x + C

## Методы интегрирования:
1. **Замена переменной:** ∫f(g(x))g'(x)dx = ∫f(t)dt
2. **Интегрирование по частям:** ∫u dv = uv - ∫v du

## Применение:
- Вычисление площадей
- Вычисление объемов
- Решение дифференциальных уравнений
- Вычисление работы, массы, заряда
                """,
                "category": "mathematics",
                "tags": "интеграл, интегрирование, математика, анализ"
            }
        ]
        
        success_count = 0
        for knowledge in math_knowledge:
            if self.add_knowledge_entry(**knowledge):
                success_count += 1
        
        print(f"✅ Загружено математических знаний: {success_count}")
        return success_count > 0
    
    def load_physics_knowledge(self) -> bool:
        """Загружает физические знания"""
        physics_knowledge = [
            {
                "title": "Законы Ньютона",
                "content": """
# ЗАКОНЫ НЬЮТОНА

## Первый закон Ньютона (закон инерции):
Тело сохраняет состояние покоя или равномерного прямолинейного движения, если на него не действуют другие тела или действие других тел скомпенсировано.

## Второй закон Ньютона:
Ускорение тела прямо пропорционально равнодействующей всех сил, приложенных к телу, и обратно пропорционально его массе.

F = ma

где:
- F - равнодействующая сила (Н)
- m - масса тела (кг)
- a - ускорение (м/с²)

## Третий закон Ньютона:
Силы, с которыми два тела действуют друг на друга, равны по модулю и противоположны по направлению.

F₁₂ = -F₂₁

## Применение:
- Анализ движения тел
- Расчет сил в механических системах
- Решение задач динамики
- Проектирование механизмов
                """,
                "category": "physics",
                "tags": "ньютон, законы, механика, физика, динамика"
            },
            {
                "title": "Электромагнитная индукция",
                "content": """
# ЭЛЕКТРОМАГНИТНАЯ ИНДУКЦИЯ

## Закон Фарадея:
ЭДС индукции в замкнутом контуре равна скорости изменения магнитного потока через поверхность, ограниченную этим контуром.

ε = -dΦ/dt

где:
- ε - ЭДС индукции (В)
- Φ - магнитный поток (Вб)
- t - время (с)

## Правило Ленца:
Индукционный ток всегда направлен так, чтобы противодействовать причине, его вызвавшей.

## Самоиндукция:
ЭДС самоиндукции пропорциональна скорости изменения силы тока.

εₛ = -L(dI/dt)

где:
- L - индуктивность (Гн)
- I - сила тока (А)

## Применение:
- Генераторы электрического тока
- Трансформаторы
- Электродвигатели
- Индукционные печи
- Беспроводная зарядка
                """,
                "category": "physics",
                "tags": "индукция, фарадей, ленц, электромагнетизм, физика"
            }
        ]
        
        success_count = 0
        for knowledge in physics_knowledge:
            if self.add_knowledge_entry(**knowledge):
                success_count += 1
        
        print(f"✅ Загружено физических знаний: {success_count}")
        return success_count > 0
    
    def load_programming_knowledge(self) -> bool:
        """Загружает знания по программированию"""
        programming_knowledge = [
            {
                "title": "Основы Python",
                "content": """
# ОСНОВЫ PYTHON

## Переменные и типы данных:
```python
# Числа
x = 42          # int
y = 3.14        # float
z = 1 + 2j      # complex

# Строки
name = "Python"  # str
text = '''Многострочная
строка'''

# Списки
numbers = [1, 2, 3, 4, 5]
mixed = [1, "hello", 3.14, True]

# Словари
person = {"name": "John", "age": 30}
```

## Управляющие структуры:
```python
# Условные операторы
if x > 0:
    print("Положительное")
elif x < 0:
    print("Отрицательное")
else:
    print("Ноль")

# Циклы
for i in range(5):
    print(i)

while x > 0:
    x -= 1
```

## Функции:
```python
def greet(name):
    return f"Привет, {name}!"

# Лямбда-функции
square = lambda x: x ** 2
```

## Классы:
```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def greet(self):
        return f"Меня зовут {self.name}"
```
                """,
                "category": "programming",
                "tags": "python, программирование, основы, синтаксис"
            },
            {
                "title": "Основы C++",
                "content": """
# ОСНОВЫ C++

## Переменные и типы данных:
```cpp
// Основные типы
int age = 25;           // целое число
double price = 99.99;   // число с плавающей точкой
char grade = 'A';       // символ
bool isActive = true;   // логический тип
string name = "John";   // строка
```

## Управляющие структуры:
```cpp
// Условные операторы
if (age >= 18) {
    cout << "Совершеннолетний" << endl;
} else if (age >= 13) {
    cout << "Подросток" << endl;
} else {
    cout << "Ребенок" << endl;
}

// Циклы
for (int i = 0; i < 10; i++) {
    cout << i << endl;
}

while (x > 0) {
    x--;
}
```

## Функции:
```cpp
int add(int a, int b) {
    return a + b;
}

// Перегрузка функций
double add(double a, double b) {
    return a + b;
}
```

## Классы:
```cpp
class Person {
private:
    string name;
    int age;
    
public:
    Person(string n, int a) : name(n), age(a) {}
    
    void greet() {
        cout << "Привет, меня зовут " << name << endl;
    }
};
```
                """,
                "category": "programming",
                "tags": "c++, программирование, основы, синтаксис"
            }
        ]
        
        success_count = 0
        for knowledge in programming_knowledge:
            if self.add_knowledge_entry(**knowledge):
                success_count += 1
        
        print(f"✅ Загружено знаний по программированию: {success_count}")
        return success_count > 0
    
    def load_automation_knowledge(self) -> bool:
        """Загружает знания по автоматизации"""
        automation_knowledge = [
            {
                "title": "Программирование PLC",
                "content": """
# ПРОГРАММИРОВАНИЕ PLC

## Основные языки программирования PLC:

### 1. Ladder Logic (LD)
```
|--[ ]--[ ]--( )--|
|  I1   I2   Q1   |
```
- Графический язык
- Похож на электрические схемы
- Легок для понимания электриками

### 2. Function Block Diagram (FBD)
```
[AND]--[TIMER]--[OUTPUT]
 I1,I2    T1       Q1
```
- Блочное программирование
- Переиспользование функций
- Хорош для сложной логики

### 3. Structured Text (ST)
```st
IF I1 AND I2 THEN
    Q1 := TRUE;
    TON(T1, 1000);
END_IF;
```
- Текстовый язык
- Похож на Pascal/C
- Мощный для сложных алгоритмов

## Основные функции:
- **TON** - таймер включения
- **TOF** - таймер выключения
- **CTU** - счетчик вверх
- **CTD** - счетчик вниз
- **MOV** - перемещение данных

## Применение:
- Автоматизация производственных процессов
- Управление конвейерами
- Контроль температуры и давления
- Управление двигателями
                """,
                "category": "automation",
                "tags": "plc, автоматизация, программирование, ladder, fbd, st"
            },
            {
                "title": "PMAC контроллеры",
                "content": """
# PMAC КОНТРОЛЛЕРЫ

## Основные возможности:
- Высокоскоростное управление движением
- Многоосевое позиционирование
- Синхронизация осей
- Интерполяция траекторий

## Основные переменные:
- **P-переменные** - параметры позиции
- **I-переменные** - параметры интерфейса
- **M-переменные** - переменные пользователя
- **Q-переменные** - переменные состояния

## Команды движения:
```pmac
#1p1000    // Установить позицию оси 1 в 1000
#1v500     // Установить скорость оси 1 в 500
#1a1000    // Установить ускорение оси 1 в 1000
#1j100     // Установить рывок оси 1 в 100
```

## Программирование:
```pmac
WHILE (1)
    #1p=1000
    #1j=1
    DWELL 1000
    #1p=0
    #1j=1
    DWELL 1000
ENDWHILE
```

## Применение:
- Станки с ЧПУ
- Робототехника
- Измерительные системы
- Прецизионное позиционирование
                """,
                "category": "automation",
                "tags": "pmac, контроллер, движение, позиционирование, cnc"
            }
        ]
        
        success_count = 0
        for knowledge in automation_knowledge:
            if self.add_knowledge_entry(**knowledge):
                success_count += 1
        
        print(f"✅ Загружено знаний по автоматизации: {success_count}")
        return success_count > 0
    
    def load_all_knowledge(self) -> bool:
        """Загружает все виды знаний"""
        print("🧠 ЗАГРУЗКА ВСЕХ ЗНАНИЙ В SMART RUBIN AI")
        print("=" * 50)
        
        success_count = 0
        
        # Загружаем математику
        if self.load_mathematics_knowledge():
            success_count += 1
        
        # Загружаем физику
        if self.load_physics_knowledge():
            success_count += 1
        
        # Загружаем программирование
        if self.load_programming_knowledge():
            success_count += 1
        
        # Загружаем автоматизацию
        if self.load_automation_knowledge():
            success_count += 1
        
        print(f"\n🎯 ИТОГО ЗАГРУЖЕНО КАТЕГОРИЙ: {success_count}")
        return success_count > 0
    
    def get_knowledge_stats(self) -> Dict[str, int]:
        """Получает статистику базы знаний"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Подсчитываем записи по категориям
            cursor.execute("SELECT category, COUNT(*) FROM knowledge GROUP BY category")
            categories = dict(cursor.fetchall())
            
            # Общее количество записей
            cursor.execute("SELECT COUNT(*) FROM knowledge")
            total = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                "total": total,
                "categories": categories
            }
            
        except Exception as e:
            print(f"❌ Ошибка получения статистики: {e}")
            return {"total": 0, "categories": {}}

def main():
    """Основная функция"""
    print("🚀 SMART RUBIN AI - СИСТЕМА ПОПОЛНЕНИЯ БАЗЫ ЗНАНИЙ")
    print("=" * 60)
    
    # Создаем экземпляр расширителя знаний
    expander = RubinKnowledgeExpander()
    
    # Показываем текущую статистику
    print("\n📊 ТЕКУЩАЯ СТАТИСТИКА БАЗЫ ЗНАНИЙ:")
    stats = expander.get_knowledge_stats()
    print(f"   Всего записей: {stats['total']}")
    for category, count in stats['categories'].items():
        print(f"   {category}: {count}")
    
    # Загружаем все знания
    print("\n🔄 ЗАГРУЗКА НОВЫХ ЗНАНИЙ:")
    success = expander.load_all_knowledge()
    
    if success:
        print("\n✅ ЗАГРУЗКА ЗАВЕРШЕНА УСПЕШНО!")
        
        # Показываем обновленную статистику
        print("\n📊 ОБНОВЛЕННАЯ СТАТИСТИКА:")
        new_stats = expander.get_knowledge_stats()
        print(f"   Всего записей: {new_stats['total']}")
        for category, count in new_stats['categories'].items():
            print(f"   {category}: {count}")
        
        print(f"\n🎉 ДОБАВЛЕНО НОВЫХ ЗАПИСЕЙ: {new_stats['total'] - stats['total']}")
    else:
        print("\n❌ ОШИБКА ПРИ ЗАГРУЗКЕ ЗНАНИЙ")

if __name__ == "__main__":
    main()
