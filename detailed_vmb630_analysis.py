#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Детальный анализ проекта VMB630
"""

from rubin_project_integration import RubinProjectIntegration
import sqlite3
import os

def detailed_vmb630_analysis():
    """Детальный анализ проекта VMB630"""
    
    integration = RubinProjectIntegration()
    projects = integration.project_reader.get_all_projects()
    
    if not projects:
        print("❌ Нет проанализированных проектов")
        return
        
    # Находим VMB630 проект
    vmb_project = None
    for project in projects:
        if 'VMB630' in project[1]:
            vmb_project = project
            break
    
    if not vmb_project:
        print("❌ VMB630 проект не найден")
        return
        
    project_id = vmb_project[0]
    print(f'🔍 ДЕТАЛЬНЫЙ АНАЛИЗ ПРОЕКТА VMB630 (ID: {project_id})')
    
    # Поиск VMB630
    vmb_results = integration.search_project_knowledge('VMB630', project_id)
    print(f'\n📄 Файлы с VMB630 ({len(vmb_results)}):')
    for result in vmb_results:
        print(f'  - {result["file_name"]} ({result["language"]})')
        print(f'    Превью: {result["content_preview"][:150]}...')
        print()
    
    # Поиск PLC
    plc_results = integration.search_project_knowledge('PLC', project_id)
    print(f'\n🔧 Файлы с PLC ({len(plc_results)}):')
    for result in plc_results[:5]:  # Показываем первые 5
        print(f'  - {result["file_name"]} ({result["language"]})')
        print(f'    Превью: {result["content_preview"][:100]}...')
        print()
    
    # Архитектура проекта
    architecture = integration.get_project_architecture(project_id)
    print(f'\n🏗️ АРХИТЕКТУРА ПРОЕКТА:')
    for comp_type, components in architecture.items():
        print(f'  {comp_type}: {len(components)} компонентов')
        for comp in components[:3]:  # Показываем первые 3
            print(f'    - {comp["name"]}: {comp["description"]}')
    
    # Получаем все файлы проекта
    print(f'\n📁 ВСЕ ФАЙЛЫ ПРОЕКТА:')
    conn_db = sqlite3.connect(integration.project_db_path)
    cursor = conn_db.cursor()
    
    cursor.execute('''
        SELECT file_name, file_extension, language, file_size, complexity_score
        FROM project_files 
        WHERE project_id = ?
        ORDER BY file_size DESC
        LIMIT 10
    ''', (project_id,))
    
    files = cursor.fetchall()
    print(f'  Топ-10 файлов по размеру:')
    for file_name, ext, lang, size, complexity in files:
        size_kb = size / 1024
        print(f'    - {file_name} ({ext}) - {size_kb:.1f} KB - {lang} - сложность: {complexity}')
    
    # Статистика по типам файлов
    cursor.execute('''
        SELECT file_extension, COUNT(*) as count, SUM(file_size) as total_size
        FROM project_files 
        WHERE project_id = ?
        GROUP BY file_extension
        ORDER BY count DESC
    ''', (project_id,))
    
    file_types = cursor.fetchall()
    print(f'\n📊 СТАТИСТИКА ПО ТИПАМ ФАЙЛОВ:')
    for ext, count, total_size in file_types:
        size_kb = total_size / 1024
        print(f'    - {ext}: {count} файлов, {size_kb:.1f} KB')
    
    # Поиск специфичных терминов
    search_terms = ['контроллер', 'motor', 'drive', 'sensor', 'input', 'output', 'function', 'program']
    print(f'\n🔍 ПОИСК СПЕЦИФИЧНЫХ ТЕРМИНОВ:')
    for term in search_terms:
        results = integration.search_project_knowledge(term, project_id)
        if results:
            print(f'  "{term}": {len(results)} результатов')
            # Показываем первый результат
            first_result = results[0]
            print(f'    → {first_result["file_name"]} ({first_result["language"]})')
    
    conn_db.close()
    
    # Попытка ответить на вопросы с контекстом
    print(f'\n💬 ПОПЫТКА ОТВЕТИТЬ НА ВОПРОСЫ:')
    questions = [
        "Что такое VMB630?",
        "Какие функции выполняет этот проект?",
        "Как работает контроллер?",
        "Какие файлы конфигурации есть в проекте?"
    ]
    
    for question in questions:
        answer = integration.answer_with_project_context(question, project_id)
        print(f'\n❓ {question}')
        if answer['confidence'] > 0:
            print(f'🎯 Ответ (уверенность: {answer["confidence"]:.1%}):')
            print(f'   {answer["answer"][:300]}...')
            if answer['sources']:
                print(f'   📚 Источники: {len(answer["sources"])} файлов')
        else:
            print('🎯 Информация не найдена в проекте')

if __name__ == "__main__":
    detailed_vmb630_analysis()










