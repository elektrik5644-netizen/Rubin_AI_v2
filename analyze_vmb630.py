#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Анализ проекта VMB630
"""

from project_folder_reader import ProjectFolderReader
from rubin_project_integration import RubinProjectIntegration
import os

def analyze_vmb630_project():
    """Анализ проекта VMB630"""
    
    # Проверяем существование пути
    path = r'C:\Users\elekt\OneDrive\Desktop\VMB630_v_005_019_000'
    print(f'🔍 Анализируем путь: {path}')
    
    if os.path.exists(path):
        print('✅ Путь существует!')
        
        # Анализируем проект
        reader = ProjectFolderReader()
        success = reader.analyze_project_folder(path, 'VMB630 Project')
        
        if success:
            print('✅ Проект успешно проанализирован!')
            
            # Получаем инсайты
            integration = RubinProjectIntegration()
            projects = integration.project_reader.get_all_projects()
            
            if projects:
                project_id = projects[0][0]  # Берем последний проанализированный
                insights = integration.generate_project_insights(project_id)
                
                print('\n📊 РЕЗУЛЬТАТЫ АНАЛИЗА:')
                overview = insights['project_overview']
                print(f'  - Название: {overview["name"]}')
                print(f'  - Тип проекта: {overview["type"]}')
                print(f'  - Файлов: {overview["total_files"]}')
                print(f'  - Размер: {overview["total_size_mb"]} MB')
                print(f'  - Средняя сложность: {overview["average_complexity"]}')
                
                # Архитектура
                arch = insights['architecture_analysis']
                if arch['component_types']:
                    print(f'  - Компонентов: {arch["total_components"]}')
                    print(f'  - Типы: {", ".join(arch["component_types"])}')
                
                # Паттерны
                patterns = insights['design_patterns']
                if patterns['total_patterns'] > 0:
                    print(f'  - Паттернов: {patterns["total_patterns"]}')
                    print(f'  - Категории: {", ".join(patterns["pattern_categories"])}')
                
                # Рекомендации
                if insights['recommendations']:
                    print('\n💡 РЕКОМЕНДАЦИИ:')
                    for rec in insights['recommendations']:
                        print(f'  - {rec}')
                
                # Поиск специфичной информации
                print('\n🔍 ПОИСК СПЕЦИФИЧНОЙ ИНФОРМАЦИИ:')
                search_queries = ['VMB630', 'контроллер', 'PLC', 'программа', 'функция']
                
                for query in search_queries:
                    results = integration.search_project_knowledge(query, project_id)
                    print(f'  "{query}": {len(results)} результатов')
                    
                    if results:
                        first_result = results[0]
                        print(f'    → {first_result["file_name"]} ({first_result["language"]})')
                
                # Ответы на вопросы
                print('\n💬 ОТВЕТЫ НА ВОПРОСЫ:')
                questions = [
                    "Что такое VMB630?",
                    "Какие функции выполняет этот проект?",
                    "Как работает контроллер?"
                ]
                
                for question in questions:
                    answer = integration.answer_with_project_context(question, project_id)
                    print(f'\n❓ {question}')
                    if answer['confidence'] > 0:
                        print(f'🎯 Ответ (уверенность: {answer["confidence"]:.1%}):')
                        print(f'   {answer["answer"][:200]}...')
                    else:
                        print('🎯 Информация не найдена в проекте')
                        
        else:
            print('❌ Ошибка анализа проекта')
    else:
        print('❌ Путь не найден!')
        print('\n🔍 Проверяем альтернативные пути...')
        
        # Проверяем похожие пути
        base_path = r'C:\Users\elekt\OneDrive\Desktop'
        if os.path.exists(base_path):
            print(f'📁 Содержимое папки {base_path}:')
            try:
                items = os.listdir(base_path)
                vmb_items = [item for item in items if 'VMB' in item.upper()]
                if vmb_items:
                    print(f'  Найдены похожие папки: {vmb_items}')
                else:
                    print(f'  Всего элементов: {len(items)}')
                    print(f'  Первые 10: {items[:10]}')
            except Exception as e:
                print(f'  Ошибка чтения папки: {e}')

if __name__ == "__main__":
    analyze_vmb630_project()










