#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Быстрый запуск системы документов Rubin AI v2.0
"""

import os
import sys
import subprocess
import time
import threading
from pathlib import Path

def create_sample_documents():
    """Создание образцов документов"""
    docs_dir = Path("sample_documents")
    docs_dir.mkdir(exist_ok=True)
    
    # Создаем образцы документов
    documents = {
        "Электротехника_Закон_Ома.txt": """
# Закон Ома в Электротехнике

## Основная формула
U = I × R

где:
- U - напряжение (Вольт)
- I - ток (Ампер)
- R - сопротивление (Ом)

## Примеры расчета

### Пример 1: Расчет тока
Дано: U = 220В, R = 10Ом
Найти: I = ?

Решение: I = U/R = 220/10 = 22А

### Пример 2: Расчет напряжения
Дано: I = 5А, R = 44Ом
Найти: U = ?

Решение: U = I×R = 5×44 = 220В

### Пример 3: Расчет сопротивления
Дано: U = 12В, I = 0.5А
Найти: R = ?

Решение: R = U/I = 12/0.5 = 24Ом

## Мощность в цепи
P = U × I = I² × R = U² / R

## Применение
Закон Ома применяется для:
- Расчетов электрических цепей
- Выбора номиналов компонентов
- Анализа работы схем
""",
        
        "Программирование_Алгоритмы.txt": """
# Алгоритмы и Структуры Данных

## Что такое алгоритм
Алгоритм - это последовательность шагов для решения задачи.

## Сложность алгоритмов
- O(1) - константная
- O(log n) - логарифмическая
- O(n) - линейная
- O(n²) - квадратичная

## Сортировка пузырьком
```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
```

## Бинарный поиск
```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1
```

## Связанные списки
```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None
    
    def append(self, data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
```
""",
        
        "Автоматизация_PID_Регулятор.txt": """
# PID Регулятор в Автоматизации

## Что такое PID регулятор
PID (Proportional-Integral-Derivative) - это алгоритм управления с обратной связью.

## Компоненты PID

### Пропорциональная составляющая (P)
u(t) = Kp × e(t)

где:
- Kp - коэффициент пропорциональности
- e(t) - ошибка (разность между заданным и текущим значением)

### Интегральная составляющая (I)
u(t) = Ki × ∫e(t)dt

где:
- Ki - коэффициент интегральности
- ∫e(t)dt - интеграл ошибки по времени

### Дифференциальная составляющая (D)
u(t) = Kd × de(t)/dt

где:
- Kd - коэффициент дифференцирования
- de(t)/dt - производная ошибки по времени

## Полная формула PID
u(t) = Kp×e(t) + Ki×∫e(t)dt + Kd×de(t)/dt

## Настройка PID регулятора

### Метод Зиглера-Николса
1. Установить Ki = 0, Kd = 0
2. Увеличивать Kp до появления колебаний
3. Записать критический коэффициент Kc и период Tc
4. Рассчитать параметры:
   - Kp = 0.6 × Kc
   - Ki = 2 × Kp / Tc
   - Kd = Kp × Tc / 8

## Применение
- Управление температурой
- Контроль скорости двигателей
- Позиционирование
- Регулирование давления
""",
        
        "Радиотехника_Модуляция.txt": """
# Модуляция в Радиотехнике

## Что такое модуляция
Модуляция - это процесс изменения параметров несущего сигнала в соответствии с информационным сигналом.

## Типы модуляции

### Амплитудная модуляция (AM)
s(t) = [A + m(t)] × cos(2πfct)

где:
- A - амплитуда несущего сигнала
- m(t) - модулирующий сигнал
- fc - частота несущего сигнала

### Частотная модуляция (FM)
s(t) = A × cos[2πfct + 2πkf∫m(τ)dτ]

где:
- kf - коэффициент частотной модуляции
- ∫m(τ)dτ - интеграл модулирующего сигнала

### Фазовая модуляция (PM)
s(t) = A × cos[2πfct + kp×m(t)]

где:
- kp - коэффициент фазовой модуляции

## Цифровая модуляция

### ASK (Amplitude Shift Keying)
- Двоичная амплитудная модуляция
- Логический 0: A = 0
- Логический 1: A = A0

### FSK (Frequency Shift Keying)
- Двоичная частотная модуляция
- Логический 0: f = f0
- Логический 1: f = f1

### PSK (Phase Shift Keying)
- Двоичная фазовая модуляция
- Логический 0: φ = 0°
- Логический 1: φ = 180°

## Применение
- Радиовещание (AM, FM)
- Цифровая связь (PSK, QAM)
- Спутниковая связь
- Мобильная связь
"""
    }
    
    # Записываем файлы
    for filename, content in documents.items():
        file_path = docs_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Создан документ: {filename}")
    
    return docs_dir

def start_documents_api():
    """Запуск API документов"""
    try:
        print("🚀 Запуск API документов...")
        process = subprocess.Popen([
            sys.executable, 'api/documents_api.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("✅ API документов запущен (порт 8088)")
        return process
    except Exception as e:
        print(f"❌ Ошибка запуска API: {e}")
        return None

def main():
    """Главная функция"""
    print("📚 СИСТЕМА ДОКУМЕНТОВ RUBIN AI v2.0")
    print("=" * 50)
    
    # Создаем образцы документов
    print("📄 Создание образцов документов...")
    docs_dir = create_sample_documents()
    
    # Загружаем документы в базу
    print("\n📥 Загрузка документов в базу данных...")
    try:
        from document_loader import DocumentLoader
        loader = DocumentLoader()
        success = loader.load_directory(docs_dir)
        
        if success:
            print("✅ Документы успешно загружены!")
            
            # Показываем статистику
            stats = loader.get_document_stats()
            if stats:
                print(f"\n📊 Статистика:")
                print(f"   Всего документов: {stats['total_documents']}")
                print(f"   Категорий: {len(stats['categories'])}")
                
                print(f"\n📂 Категории:")
                for category, count in stats['categories']:
                    print(f"   {category}: {count} документов")
        else:
            print("❌ Ошибка загрузки документов")
            return
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return
    
    # Запускаем API
    print("\n🌐 Запуск веб-интерфейса...")
    api_process = start_documents_api()
    
    if api_process:
        print("\n" + "=" * 50)
        print("🎉 СИСТЕМА ДОКУМЕНТОВ ЗАПУЩЕНА!")
        print("=" * 50)
        print("🌐 Доступные интерфейсы:")
        print("   - API документов: http://localhost:8088")
        print("   - Веб-интерфейс: http://localhost:8088/DocumentsManager.html")
        print("   - Статус API: http://localhost:8088/health")
        print("\n📚 Загруженные документы:")
        print("   - Электротехника: Закон Ома")
        print("   - Программирование: Алгоритмы")
        print("   - Автоматизация: PID регулятор")
        print("   - Радиотехника: Модуляция")
        print("\n⏳ Нажмите Ctrl+C для остановки")
        
        try:
            api_process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Остановка системы...")
            api_process.terminate()
            print("✅ Система остановлена")
    else:
        print("❌ Не удалось запустить API документов")

if __name__ == "__main__":
    main()


















