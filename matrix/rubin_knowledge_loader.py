#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Rubin AI - Улучшенная система загрузки знаний
Работает с существующей структурой базы данных
"""

import sqlite3
import json
import os
import datetime
from typing import Dict, List, Any, Optional

class RubinKnowledgeLoader:
    """Класс для загрузки знаний в Smart Rubin AI"""
    
    def __init__(self, db_path: str = "rubin_knowledge_base.db"):
        self.db_path = db_path
        
    def add_knowledge_entry(self, title: str, content: str, category: str = "general", 
                          tags: str = "", source: str = "manual") -> bool:
        """Добавляет новую запись в базу знаний"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Проверяем структуру таблицы knowledge
            cursor.execute("PRAGMA table_info(knowledge)")
            columns = cursor.fetchall()
            print(f"Структура таблицы knowledge: {[col[1] for col in columns]}")
            
            # Вставляем запись с правильными полями
            cursor.execute("""
                INSERT INTO knowledge (title, content, category, tags, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (title, content, category, tags, datetime.datetime.now().isoformat()))
            
            conn.commit()
            conn.close()
            
            print(f"✅ Добавлена запись: {title}")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка добавления записи: {e}")
            return False
    
    def add_to_knowledge_base_table(self, title: str, content: str, category: str = "general", 
                                  tags: str = "", keywords: str = "") -> bool:
        """Добавляет запись в таблицу knowledge_base"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Проверяем структуру таблицы knowledge_base
            cursor.execute("PRAGMA table_info(knowledge_base)")
            columns = cursor.fetchall()
            print(f"Структура таблицы knowledge_base: {[col[1] for col in columns]}")
            
            # Вставляем запись
            cursor.execute("""
                INSERT INTO knowledge_base (title, content, category, tags, keywords, created_at, usage_count, relevance_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (title, content, category, tags, keywords, datetime.datetime.now().isoformat(), 0, 1.0))
            
            conn.commit()
            conn.close()
            
            print(f"✅ Добавлена запись в knowledge_base: {title}")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка добавления в knowledge_base: {e}")
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
                "tags": "производная, дифференцирование, математика, анализ",
                "keywords": "производная, дифференцирование, математика, анализ, экстремум, монотонность"
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
                "tags": "интеграл, интегрирование, математика, анализ",
                "keywords": "интеграл, интегрирование, математика, анализ, площадь, объем, дифференциальные уравнения"
            },
            {
                "title": "Линейная алгебра - матрицы",
                "content": """
# ЛИНЕЙНАЯ АЛГЕБРА - МАТРИЦЫ

## Определение матрицы:
Матрица - это прямоугольная таблица чисел, расположенных в строках и столбцах.

A = [a₁₁  a₁₂  a₁₃]
    [a₂₁  a₂₂  a₂₃]
    [a₃₁  a₃₂  a₃₃]

## Основные операции с матрицами:

### 1. Сложение матриц:
C = A + B, где cᵢⱼ = aᵢⱼ + bᵢⱼ

### 2. Умножение на число:
B = kA, где bᵢⱼ = k·aᵢⱼ

### 3. Умножение матриц:
C = A·B, где cᵢⱼ = Σ(aᵢₖ·bₖⱼ)

## Специальные типы матриц:

### Единичная матрица:
I = [1  0  0]
    [0  1  0]
    [0  0  1]

### Транспонированная матрица:
Aᵀ, где aᵢⱼᵀ = aⱼᵢ

### Обратная матрица:
A⁻¹, где A·A⁻¹ = I

## Определитель матрицы:
det(A) = Σ(-1)ᵏ·a₁ₖ·M₁ₖ

## Применение:
- Решение систем линейных уравнений
- Преобразования координат
- Компьютерная графика
- Машинное обучение
- Квантовая механика
                """,
                "category": "mathematics",
                "tags": "матрицы, линейная алгебра, математика, определитель",
                "keywords": "матрицы, линейная алгебра, математика, определитель, обратная матрица, транспонирование"
            }
        ]
        
        success_count = 0
        for knowledge in math_knowledge:
            if self.add_to_knowledge_base_table(**knowledge):
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
                "tags": "ньютон, законы, механика, физика, динамика",
                "keywords": "ньютон, законы, механика, физика, динамика, сила, ускорение, масса"
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
                "tags": "индукция, фарадей, ленц, электромагнетизм, физика",
                "keywords": "индукция, фарадей, ленц, электромагнетизм, физика, ЭДС, магнитный поток, самоиндукция"
            },
            {
                "title": "Квантовая механика - основы",
                "content": """
# КВАНТОВАЯ МЕХАНИКА - ОСНОВЫ

## Принцип неопределенности Гейзенберга:
Невозможно одновременно точно измерить координату и импульс частицы.

Δx·Δp ≥ ℏ/2

где:
- Δx - неопределенность координаты
- Δp - неопределенность импульса
- ℏ - постоянная Планка

## Уравнение Шредингера:
iℏ(∂ψ/∂t) = Ĥψ

где:
- ψ - волновая функция
- Ĥ - гамильтониан
- ℏ - постоянная Планка

## Квантование энергии:
E = nℏω

где:
- n - квантовое число
- ω - частота
- ℏ - постоянная Планка

## Принцип суперпозиции:
Частица может находиться в нескольких состояниях одновременно.

## Применение:
- Лазеры
- Полупроводники
- Ядерная энергетика
- Квантовые компьютеры
- МРТ
                """,
                "category": "physics",
                "tags": "квантовая механика, гейзенберг, шредингер, физика",
                "keywords": "квантовая механика, гейзенберг, шредингер, физика, неопределенность, волновая функция, квантование"
            }
        ]
        
        success_count = 0
        for knowledge in physics_knowledge:
            if self.add_to_knowledge_base_table(**knowledge):
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
                "tags": "python, программирование, основы, синтаксис",
                "keywords": "python, программирование, основы, синтаксис, переменные, функции, классы, циклы"
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
                "tags": "c++, программирование, основы, синтаксис",
                "keywords": "c++, программирование, основы, синтаксис, переменные, функции, классы, циклы, перегрузка"
            }
        ]
        
        success_count = 0
        for knowledge in programming_knowledge:
            if self.add_to_knowledge_base_table(**knowledge):
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
                "tags": "plc, автоматизация, программирование, ladder, fbd, st",
                "keywords": "plc, автоматизация, программирование, ladder, fbd, st, таймер, счетчик, логика"
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
                "tags": "pmac, контроллер, движение, позиционирование, cnc",
                "keywords": "pmac, контроллер, движение, позиционирование, cnc, переменные, команды, программирование"
            }
        ]
        
        success_count = 0
        for knowledge in automation_knowledge:
            if self.add_to_knowledge_base_table(**knowledge):
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
            
            # Подсчитываем записи по категориям в таблице knowledge_base
            cursor.execute("SELECT category, COUNT(*) FROM knowledge_base GROUP BY category")
            categories = dict(cursor.fetchall())
            
            # Общее количество записей
            cursor.execute("SELECT COUNT(*) FROM knowledge_base")
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
    print("🚀 SMART RUBIN AI - СИСТЕМА ЗАГРУЗКИ ЗНАНИЙ")
    print("=" * 60)
    
    # Создаем экземпляр загрузчика знаний
    loader = RubinKnowledgeLoader()
    
    # Показываем текущую статистику
    print("\n📊 ТЕКУЩАЯ СТАТИСТИКА БАЗЫ ЗНАНИЙ:")
    stats = loader.get_knowledge_stats()
    print(f"   Всего записей: {stats['total']}")
    for category, count in stats['categories'].items():
        print(f"   {category}: {count}")
    
    # Загружаем все знания
    print("\n🔄 ЗАГРУЗКА НОВЫХ ЗНАНИЙ:")
    success = loader.load_all_knowledge()
    
    if success:
        print("\n✅ ЗАГРУЗКА ЗАВЕРШЕНА УСПЕШНО!")
        
        # Показываем обновленную статистику
        print("\n📊 ОБНОВЛЕННАЯ СТАТИСТИКА:")
        new_stats = loader.get_knowledge_stats()
        print(f"   Всего записей: {new_stats['total']}")
        for category, count in new_stats['categories'].items():
            print(f"   {category}: {count}")
        
        print(f"\n🎉 ДОБАВЛЕНО НОВЫХ ЗАПИСЕЙ: {new_stats['total'] - stats['total']}")
    else:
        print("\n❌ ОШИБКА ПРИ ЗАГРУЗКЕ ЗНАНИЙ")

if __name__ == "__main__":
    main()
