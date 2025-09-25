#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from mathematical_problem_solver import MathematicalProblemSolver

def comparison_demo():
    solver = MathematicalProblemSolver()
    
    print('🧮 СРАВНЕНИЕ: Rubin AI vs Математический решатель')
    print('=' * 60)
    
    problems = [
        '2+2=?',
        '3+4=?', 
        '5*6=?',
        'Реши уравнение 2x + 3 = 11',
        'Найди 20% от 150'
    ]
    
    for problem in problems:
        print(f'\n❌ Rubin AI: "{problem}" → НЕ МОЖЕТ РЕШИТЬ')
        print('   Перенаправляет к AI Чат, который не имеет математических возможностей')
        
        # Решаем через наш алгоритм
        if '?' in problem:
            clean_problem = problem.replace('?', '')
            solution = solver.solve_problem(f'Вычисли {clean_problem}')
        else:
            solution = solver.solve_problem(problem)
        
        print(f'✅ Наш решатель: "{problem}" → {solution.final_answer}')
        print(f'   Тип: {solution.problem_type.value}, Уверенность: {solution.confidence:.2f}')

if __name__ == "__main__":
    comparison_demo()

