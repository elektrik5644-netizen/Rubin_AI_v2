#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Отладка проблемы с физическими формулами
"""

from mathematical_problem_solver import MathematicalProblemSolver

def debug_physics_solver():
    """Отладка физического решателя"""
    print("🔍 ОТЛАДКА ФИЗИЧЕСКОГО РЕШАТЕЛЯ")
    print("=" * 50)
    
    solver = MathematicalProblemSolver()
    
    # Тест 1: Проверка метода _detect_physics_formula
    print("\n1. Тест _detect_physics_formula:")
    problem_text = "Найти напряжение при токе 2 А и сопротивлении 5 Ом"
    print(f"Входной текст: {problem_text}")
    print(f"Тип: {type(problem_text)}")
    
    try:
        formula = solver._detect_physics_formula(problem_text)
        print(f"Найденная формула: {formula}")
    except Exception as e:
        print(f"Ошибка: {e}")
    
    # Тест 2: Проверка метода _extract_physics_variables
    print("\n2. Тест _extract_physics_variables:")
    try:
        variables = solver._extract_physics_variables(problem_text)
        print(f"Извлеченные переменные: {variables}")
    except Exception as e:
        print(f"Ошибка: {e}")
    
    # Тест 3: Проверка полного решения
    print("\n3. Тест полного решения:")
    try:
        result = solver.solve_problem(problem_text)
        print(f"Тип задачи: {result.problem_type.value}")
        print(f"Ответ: {result.final_answer}")
        print(f"Уверенность: {result.confidence:.1%}")
    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_physics_solver()





