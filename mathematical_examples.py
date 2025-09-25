#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📚 Примеры использования математического решателя
================================================

Демонстрация возможностей математического решателя Rubin AI
с различными типами задач и их решениями.

Автор: Rubin AI System
Версия: 2.0
"""

import sys
import os
import time
import json

# Добавляем путь к модулям
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mathematical_problem_solver import MathematicalProblemSolver, ProblemType

def print_solution(solution, problem_text):
    """Красивый вывод решения"""
    print(f"\n{'='*60}")
    print(f"📝 Задача: {problem_text}")
    print(f"🎯 Тип: {solution.problem_type.value}")
    print(f"✅ Ответ: {solution.final_answer}")
    print(f"🎲 Уверенность: {solution.confidence:.2f}")
    print(f"🔍 Проверка: {'✓' if solution.verification else '✗'}")
    print(f"⏱️  Время: {solution.processing_time:.3f}s")
    
    if solution.solution_steps:
        print(f"\n📋 Пошаговое решение:")
        for i, step in enumerate(solution.solution_steps, 1):
            print(f"   {i}. {step}")
    
    if solution.explanation:
        print(f"\n💡 Объяснение: {solution.explanation}")

def example_arithmetic():
    """Примеры арифметических задач"""
    print("\n🧮 АРИФМЕТИЧЕСКИЕ ЗАДАЧИ")
    print("="*60)
    
    solver = MathematicalProblemSolver()
    
    problems = [
        "Вычисли 2 + 3 * 4",
        "Посчитай (10 - 3) * 2 + 5",
        "Найди результат 15 / 3 + 2 * 4",
        "Вычисли 2^3 + 4 * 5",
        "Найди значение 100 - 25 * 2 + 10"
    ]
    
    for problem in problems:
        start_time = time.time()
        solution = solver.solve_problem(problem)
        solution.processing_time = time.time() - start_time
        print_solution(solution, problem)

def example_equations():
    """Примеры решения уравнений"""
    print("\n📐 РЕШЕНИЕ УРАВНЕНИЙ")
    print("="*60)
    
    solver = MathematicalProblemSolver()
    
    problems = [
        "Реши уравнение 2x + 5 = 13",
        "Найди x в уравнении 3x - 7 = 8",
        "Реши 4x + 2 = 18",
        "Найди корень уравнения 5x - 3 = 12",
        "Реши уравнение 2x + 3 = 7"
    ]
    
    for problem in problems:
        start_time = time.time()
        solution = solver.solve_problem(problem)
        solution.processing_time = time.time() - start_time
        print_solution(solution, problem)

def example_quadratic_equations():
    """Примеры квадратных уравнений"""
    print("\n📊 КВАДРАТНЫЕ УРАВНЕНИЯ")
    print("="*60)
    
    solver = MathematicalProblemSolver()
    
    problems = [
        "Реши квадратное уравнение x² - 5x + 6 = 0",
        "Найди корни уравнения x² - 4 = 0",
        "Реши уравнение x² - 3x + 2 = 0",
        "Найди решения x² + 2x - 3 = 0"
    ]
    
    for problem in problems:
        start_time = time.time()
        solution = solver.solve_problem(problem)
        solution.processing_time = time.time() - start_time
        print_solution(solution, problem)

def example_geometry():
    """Примеры геометрических задач"""
    print("\n📏 ГЕОМЕТРИЧЕСКИЕ ЗАДАЧИ")
    print("="*60)
    
    solver = MathematicalProblemSolver()
    
    problems = [
        "Найди площадь треугольника с основанием 5 и высотой 3",
        "Рассчитай площадь круга с радиусом 4",
        "Найди площадь прямоугольника длиной 6 и шириной 8",
        "Вычисли площадь квадрата со стороной 5",
        "Найди площадь треугольника с основанием 10 и высотой 6"
    ]
    
    for problem in problems:
        start_time = time.time()
        solution = solver.solve_problem(problem)
        solution.processing_time = time.time() - start_time
        print_solution(solution, problem)

def example_trigonometry():
    """Примеры тригонометрических задач"""
    print("\n📐 ТРИГОНОМЕТРИЯ")
    print("="*60)
    
    solver = MathematicalProblemSolver()
    
    problems = [
        "Вычисли sin(30°)",
        "Найди cos(60°)",
        "Вычисли tan(45°)",
        "Найди sin(90°)",
        "Вычисли cos(0°)"
    ]
    
    for problem in problems:
        start_time = time.time()
        solution = solver.solve_problem(problem)
        solution.processing_time = time.time() - start_time
        print_solution(solution, problem)

def example_statistics():
    """Примеры статистических задач"""
    print("\n📊 СТАТИСТИКА")
    print("="*60)
    
    solver = MathematicalProblemSolver()
    
    problems = [
        "Найди среднее значение чисел 1, 2, 3, 4, 5",
        "Рассчитай статистику для данных 10, 20, 30, 40, 50",
        "Найди среднее, медиану и стандартное отклонение для 2, 4, 6, 8, 10",
        "Вычисли статистические характеристики 15, 25, 35, 45, 55"
    ]
    
    for problem in problems:
        start_time = time.time()
        solution = solver.solve_problem(problem)
        solution.processing_time = time.time() - start_time
        print_solution(solution, problem)

def example_percentage():
    """Примеры задач на проценты"""
    print("\n📈 ЗАДАЧИ НА ПРОЦЕНТЫ")
    print("="*60)
    
    solver = MathematicalProblemSolver()
    
    problems = [
        "Найди 15% от 200",
        "Сколько составляет 25% от 80",
        "Найди 10% от 150",
        "Вычисли 30% от 300",
        "Найди 5% от 1000"
    ]
    
    for problem in problems:
        start_time = time.time()
        solution = solver.solve_problem(problem)
        solution.processing_time = time.time() - start_time
        print_solution(solution, problem)

def example_complex_problems():
    """Примеры сложных задач"""
    print("\n🎯 СЛОЖНЫЕ ЗАДАЧИ")
    print("="*60)
    
    solver = MathematicalProblemSolver()
    
    problems = [
        "Реши систему уравнений: 2x + y = 5, x - y = 1",
        "Найди объем цилиндра с радиусом 3 и высотой 5",
        "Вычисли площадь треугольника по формуле Герона со сторонами 3, 4, 5",
        "Найди производную функции sin(x)",
        "Реши уравнение x³ - 6x² + 11x - 6 = 0"
    ]
    
    for problem in problems:
        start_time = time.time()
        solution = solver.solve_problem(problem)
        solution.processing_time = time.time() - start_time
        print_solution(solution, problem)

def interactive_mode():
    """Интерактивный режим"""
    print("\n🎮 ИНТЕРАКТИВНЫЙ РЕЖИМ")
    print("="*60)
    print("Введите математическую задачу или 'выход' для завершения")
    
    solver = MathematicalProblemSolver()
    
    while True:
        try:
            problem = input("\n📝 Введите задачу: ").strip()
            
            if problem.lower() in ['выход', 'exit', 'quit']:
                print("👋 До свидания!")
                break
            
            if not problem:
                continue
            
            start_time = time.time()
            solution = solver.solve_problem(problem)
            solution.processing_time = time.time() - start_time
            
            print_solution(solution, problem)
            
        except KeyboardInterrupt:
            print("\n👋 До свидания!")
            break
        except Exception as e:
            print(f"❌ Ошибка: {e}")

def benchmark_test():
    """Тест производительности"""
    print("\n⚡ ТЕСТ ПРОИЗВОДИТЕЛЬНОСТИ")
    print("="*60)
    
    solver = MathematicalProblemSolver()
    
    test_problems = [
        "Вычисли 2 + 3 * 4",
        "Реши уравнение 2x + 5 = 13",
        "Найди площадь треугольника с основанием 5 и высотой 3",
        "Реши квадратное уравнение x² - 5x + 6 = 0",
        "Найди 15% от 200",
        "Вычисли sin(30°)",
        "Найди среднее значение чисел 1, 2, 3, 4, 5"
    ]
    
    total_time = 0
    results = []
    
    for i, problem in enumerate(test_problems, 1):
        start_time = time.time()
        solution = solver.solve_problem(problem)
        processing_time = time.time() - start_time
        total_time += processing_time
        
        results.append({
            "problem": problem,
            "type": solution.problem_type.value,
            "answer": solution.final_answer,
            "confidence": solution.confidence,
            "time": processing_time
        })
        
        print(f"{i:2d}. {problem[:40]:<40} | {processing_time:.3f}s | {solution.confidence:.2f}")
    
    average_time = total_time / len(test_problems)
    
    print(f"\n📊 Статистика:")
    print(f"   Всего задач: {len(test_problems)}")
    print(f"   Общее время: {total_time:.3f}s")
    print(f"   Среднее время: {average_time:.3f}s")
    print(f"   Задач в секунду: {1/average_time:.1f}")
    
    return results

def save_results_to_file(results, filename="mathematical_results.json"):
    """Сохранение результатов в файл"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"💾 Результаты сохранены в {filename}")
    except Exception as e:
        print(f"❌ Ошибка сохранения: {e}")

def main():
    """Главная функция"""
    print("🧮 МАТЕМАТИЧЕСКИЙ РЕШАТЕЛЬ RUBIN AI")
    print("="*60)
    print("Выберите режим:")
    print("1. Арифметические задачи")
    print("2. Решение уравнений")
    print("3. Квадратные уравнения")
    print("4. Геометрические задачи")
    print("5. Тригонометрия")
    print("6. Статистика")
    print("7. Задачи на проценты")
    print("8. Сложные задачи")
    print("9. Тест производительности")
    print("10. Интерактивный режим")
    print("11. Все примеры")
    print("0. Выход")
    
    while True:
        try:
            choice = input("\n🎯 Выберите режим (0-11): ").strip()
            
            if choice == '0':
                print("👋 До свидания!")
                break
            elif choice == '1':
                example_arithmetic()
            elif choice == '2':
                example_equations()
            elif choice == '3':
                example_quadratic_equations()
            elif choice == '4':
                example_geometry()
            elif choice == '5':
                example_trigonometry()
            elif choice == '6':
                example_statistics()
            elif choice == '7':
                example_percentage()
            elif choice == '8':
                example_complex_problems()
            elif choice == '9':
                results = benchmark_test()
                save_results_to_file(results)
            elif choice == '10':
                interactive_mode()
            elif choice == '11':
                example_arithmetic()
                example_equations()
                example_quadratic_equations()
                example_geometry()
                example_trigonometry()
                example_statistics()
                example_percentage()
                example_complex_problems()
            else:
                print("❌ Неверный выбор. Попробуйте снова.")
                
        except KeyboardInterrupt:
            print("\n👋 До свидания!")
            break
        except Exception as e:
            print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    main()

