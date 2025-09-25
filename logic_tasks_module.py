#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль логических задач для Rubin AI v2
Интеграция базы данных LogiEval
"""

import json
import os
import random
from typing import Dict, List, Any, Optional
from datetime import datetime

class LogicTaskDatabase:
    """База данных логических задач."""
    
    def __init__(self, data_path: str = r"C:\Users\elekt\OneDrive\Desktop\LogiEval-main\Data"):
        self.data_path = data_path
        self.datasets = {}
        self.load_all_datasets()
    
    def load_all_datasets(self):
        """Загружает все наборы данных."""
        dataset_files = {
            'logiqa': 'logiqa.jsonl',
            'logiqa_zh': 'logiqa_zh.jsonl', 
            'logiqa_ood': 'logiqa_ood.jsonl',
            'proofwriter': 'proofwriter.jsonl',
            'ruletaker': 'ruletaker.jsonl',
            'taxi': 'taxi.jsonl',
            'fracas': 'fracas.jsonl',
            'control': 'control.jsonl',
            'help': 'help.jsonl',
            'med': 'med.jsonl',
            'mnli': 'mnli.jsonl',
            'nannli': 'nannli.jsonl',
            'conjnli': 'conjnli.jsonl',
            'reclor': 'reclor.jsonl',
            'ar_lsat': 'ar_lsat.jsonl'
        }
        
        for dataset_name, filename in dataset_files.items():
            file_path = os.path.join(self.data_path, filename)
            if os.path.exists(file_path):
                try:
                    self.datasets[dataset_name] = self.load_jsonl(file_path)
                    print(f"✅ Загружен датасет {dataset_name}: {len(self.datasets[dataset_name])} задач")
                except Exception as e:
                    print(f"❌ Ошибка загрузки {dataset_name}: {e}")
            else:
                print(f"⚠️ Файл не найден: {file_path}")
    
    def load_jsonl(self, file_path: str) -> List[Dict]:
        """Загружает JSONL файл."""
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        except Exception as e:
            print(f"❌ Ошибка чтения {file_path}: {e}")
        return data
    
    def get_random_task(self, dataset_name: str = None) -> Optional[Dict]:
        """Получает случайную задачу."""
        if dataset_name and dataset_name in self.datasets:
            dataset = self.datasets[dataset_name]
        else:
            # Выбираем случайный датасет
            if not self.datasets:
                return None
            dataset_name = random.choice(list(self.datasets.keys()))
            dataset = self.datasets[dataset_name]
        
        if not dataset:
            return None
        
        task = random.choice(dataset)
        task['dataset_name'] = dataset_name
        return task
    
    def get_task_by_type(self, task_type: str) -> Optional[Dict]:
        """Получает задачу определенного типа."""
        type_mapping = {
            'логика': 'logiqa',
            'доказательства': 'proofwriter',
            'правила': 'ruletaker',
            'такси': 'taxi',
            'контроль': 'control',
            'медицина': 'med',
            'математика': 'mnli',
            'китайский': 'logiqa_zh',
            'аргументы': 'ar_lsat'
        }
        
        dataset_name = type_mapping.get(task_type.lower())
        if dataset_name and dataset_name in self.datasets:
            return self.get_random_task(dataset_name)
        
        return self.get_random_task()
    
    def format_task_for_chat(self, task: Dict) -> str:
        """Форматирует задачу для чата."""
        if not task:
            return "Задача не найдена"
        
        dataset_name = task.get('dataset_name', 'unknown')
        input_data = task.get('input', [])
        
        if not input_data:
            return "Некорректная задача"
        
        # Извлекаем содержимое задачи
        content = ""
        for item in input_data:
            if item.get('role') == 'user':
                content = item.get('content', '')
                break
        
        if not content:
            return "Содержимое задачи не найдено"
        
        # Форматируем для Rubin
        formatted_task = f"""
🧠 **Логическая задача** (тип: {dataset_name})

{content}

---
💡 **Подсказка**: Проанализируйте логические связи и выберите правильный ответ.
"""
        return formatted_task.strip()
    
    def get_task_answer(self, task: Dict) -> str:
        """Получает правильный ответ на задачу."""
        return task.get('ideal', 'Ответ не найден')
    
    def get_dataset_stats(self) -> Dict:
        """Получает статистику по датасетам."""
        stats = {}
        for name, data in self.datasets.items():
            stats[name] = {
                'count': len(data),
                'description': self.get_dataset_description(name)
            }
        return stats
    
    def get_dataset_description(self, dataset_name: str) -> str:
        """Получает описание датасета."""
        descriptions = {
            'logiqa': 'Логические вопросы и ответы',
            'logiqa_zh': 'Логические вопросы на китайском',
            'logiqa_ood': 'Логические вопросы вне домена',
            'proofwriter': 'Задачи на доказательства',
            'ruletaker': 'Задачи с правилами',
            'taxi': 'Задачи такси (логические рассуждения)',
            'fracas': 'Фрактальные задачи',
            'control': 'Контрольные задачи',
            'help': 'Задачи помощи',
            'med': 'Медицинские логические задачи',
            'mnli': 'Математические логические задачи',
            'nannli': 'Неаннотированные логические задачи',
            'conjnli': 'Соединенные логические задачи',
            'reclor': 'Задачи чтения и понимания',
            'ar_lsat': 'Аргументативные задачи LSAT'
        }
        return descriptions.get(dataset_name, 'Неизвестный тип задач')

class LogicTaskSolver:
    """Решатель логических задач."""
    
    def __init__(self, database: LogicTaskDatabase):
        self.database = database
        self.solved_tasks = []
        self.failed_tasks = []
    
    def solve_task(self, task: Dict, user_answer: str = None) -> Dict:
        """Решает задачу."""
        if not task:
            return {'success': False, 'error': 'Задача не найдена'}
        
        correct_answer = self.database.get_task_answer(task)
        dataset_name = task.get('dataset_name', 'unknown')
        
        result = {
            'task': self.database.format_task_for_chat(task),
            'correct_answer': correct_answer,
            'user_answer': user_answer,
            'dataset_name': dataset_name,
            'timestamp': datetime.now().isoformat()
        }
        
        if user_answer:
            is_correct = str(user_answer).strip().upper() == str(correct_answer).strip().upper()
            result['is_correct'] = is_correct
            
            if is_correct:
                self.solved_tasks.append(result)
                result['message'] = '🎉 Правильно! Отличная работа!'
            else:
                self.failed_tasks.append(result)
                result['message'] = f'❌ Неправильно. Правильный ответ: {correct_answer}'
        else:
            result['message'] = f'💡 Правильный ответ: {correct_answer}'
        
        return result
    
    def get_statistics(self) -> Dict:
        """Получает статистику решения задач."""
        total_attempts = len(self.solved_tasks) + len(self.failed_tasks)
        success_rate = (len(self.solved_tasks) / total_attempts * 100) if total_attempts > 0 else 0
        
        return {
            'total_attempts': total_attempts,
            'solved_tasks': len(self.solved_tasks),
            'failed_tasks': len(self.failed_tasks),
            'success_rate': f"{success_rate:.1f}%",
            'dataset_stats': self.database.get_dataset_stats()
        }

# Глобальные экземпляры
logic_database = LogicTaskDatabase()
logic_solver = LogicTaskSolver(logic_database)

def get_logic_task(task_type: str = None) -> str:
    """Получает логическую задачу для Rubin."""
    task = logic_database.get_task_by_type(task_type) if task_type else logic_database.get_random_task()
    return logic_database.format_task_for_chat(task)

def solve_logic_task(task_type: str = None, user_answer: str = None) -> Dict:
    """Решает логическую задачу."""
    task = logic_database.get_task_by_type(task_type) if task_type else logic_database.get_random_task()
    return logic_solver.solve_task(task, user_answer)

def get_logic_statistics() -> Dict:
    """Получает статистику логических задач."""
    return logic_solver.get_statistics()

if __name__ == '__main__':
    # Тестирование модуля
    print("🧠 Тестирование модуля логических задач")
    print("=" * 50)
    
    # Получаем случайную задачу
    task = get_logic_task()
    print("📝 Случайная задача:")
    print(task)
    
    print("\n📊 Статистика датасетов:")
    stats = get_logic_statistics()
    for name, info in stats['dataset_stats'].items():
        print(f"  🔹 {name}: {info['count']} задач - {info['description']}")



