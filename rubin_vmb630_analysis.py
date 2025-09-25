#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Анализ VMB630 через Rubin AI
"""

from rubin_project_integration import RubinProjectIntegration
import os

def analyze_vmb630_with_rubin():
    """Анализ VMB630 проекта через Rubin AI"""
    
    print("🔍 RUBIN AI АНАЛИЗ ПРОЕКТА VMB630")
    print("=" * 50)
    
    # Проверяем существование пути
    path = r'C:\Users\elekt\OneDrive\Desktop\VMB630_v_005_019_000'
    
    if not os.path.exists(path):
        print("❌ Путь не найден!")
        return
    
    # Инициализируем Rubin AI интеграцию
    integration = RubinProjectIntegration()
    
    # Получаем все проекты
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
        print("❌ VMB630 проект не найден в базе данных")
        return
    
    project_id = vmb_project[0]
    print(f"✅ Найден проект VMB630 (ID: {project_id})")
    
    # Генерируем инсайты
    print("\n🧠 ГЕНЕРАЦИЯ ИНСАЙТОВ RUBIN AI...")
    insights = integration.generate_project_insights(project_id)
    
    # Выводим результаты
    print("\n📊 РЕЗУЛЬТАТЫ АНАЛИЗА:")
    print("-" * 30)
    
    overview = insights['project_overview']
    print(f"📁 Название: {overview['name']}")
    print(f"🔧 Тип проекта: {overview['type']}")
    print(f"📄 Файлов: {overview['total_files']}")
    print(f"💾 Размер: {overview['total_size_mb']} MB")
    print(f"⚙️ Сложность: {overview['average_complexity']}")
    
    # Архитектура
    arch = insights['architecture_analysis']
    if arch['component_types']:
        print(f"\n🏗️ АРХИТЕКТУРА:")
        print(f"  Компонентов: {arch['total_components']}")
        print(f"  Типы: {', '.join(arch['component_types'])}")
    
    # Паттерны
    patterns = insights['design_patterns']
    if patterns['total_patterns'] > 0:
        print(f"\n🎨 ПАТТЕРНЫ ПРОЕКТИРОВАНИЯ:")
        print(f"  Всего: {patterns['total_patterns']}")
        print(f"  Категории: {', '.join(patterns['pattern_categories'])}")
    
    # Рекомендации
    if insights['recommendations']:
        print(f"\n💡 РЕКОМЕНДАЦИИ RUBIN AI:")
        for i, rec in enumerate(insights['recommendations'], 1):
            print(f"  {i}. {rec}")
    
    # Поиск специфичной информации
    print(f"\n🔍 ПОИСК СПЕЦИФИЧНОЙ ИНФОРМАЦИИ:")
    search_terms = ['VMB630', 'контроллер', 'PLC', 'motor', 'axis', 'spindle']
    
    for term in search_terms:
        results = integration.search_project_knowledge(term, project_id)
        if results:
            print(f"  '{term}': {len(results)} результатов")
            # Показываем первый результат
            first_result = results[0]
            print(f"    → {first_result['file_name']} ({first_result['language']})")
            print(f"    Превью: {first_result['content_preview'][:100]}...")
    
    # Ответы на вопросы через Rubin AI
    print(f"\n💬 ОТВЕТЫ RUBIN AI НА ВОПРОСЫ:")
    questions = [
        "Что такое VMB630?",
        "Какие функции выполняет этот проект?",
        "Как работает система управления осями?",
        "Какие файлы конфигурации есть в проекте?",
        "Что такое PLC в этом проекте?"
    ]
    
    for question in questions:
        print(f"\n❓ {question}")
        answer = integration.answer_with_project_context(question, project_id)
        
        if answer['confidence'] > 0:
            print(f"🎯 Ответ (уверенность: {answer['confidence']:.1%}):")
            print(f"   {answer['answer'][:300]}...")
            if answer['sources']:
                print(f"   📚 Источники: {len(answer['sources'])} файлов")
        else:
            print("🎯 Информация не найдена в проекте")
    
    # Получаем архитектуру проекта
    print(f"\n🏗️ ДЕТАЛЬНАЯ АРХИТЕКТУРА:")
    architecture = integration.get_project_architecture(project_id)
    
    for comp_type, components in architecture.items():
        if components:
            print(f"\n  {comp_type.upper()}:")
            for comp in components[:5]:  # Показываем первые 5
                print(f"    - {comp['name']}: {comp['description']}")
    
    print(f"\n🎉 АНАЛИЗ ЗАВЕРШЕН!")
    print("Rubin AI успешно проанализировал проект VMB630!")

if __name__ == "__main__":
    analyze_vmb630_with_rubin()





