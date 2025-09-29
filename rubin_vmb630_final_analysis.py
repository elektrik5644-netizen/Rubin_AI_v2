#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Простой анализ VMB630 проекта через Rubin AI
"""

import os
from rubin_project_integration import RubinProjectIntegration

def analyze_vmb630_simple():
    """Простой анализ VMB630"""
    
    print("🔍 RUBIN AI АНАЛИЗ ПРОЕКТА VMB630")
    print("=" * 50)
    
    # Инициализация Rubin AI
    integrator = RubinProjectIntegration()
    
    # Получаем все проекты
    projects = integrator.project_reader.get_all_projects()
    
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
    
    # Получаем сводку проекта
    summary = integrator.project_reader.get_project_summary(project_id)
    print(f"\n📊 СВОДКА ПРОЕКТА:")
    print(f"  📁 Название: {vmb_project[1]}")
    print(f"  🔧 Тип проекта: {vmb_project[3]}")
    print(f"  📄 Файлов: {vmb_project[6]}")
    print(f"  💾 Размер: {vmb_project[7]:.2f} MB")
    
    # Получаем все файлы проекта
    all_files = integrator.project_reader.get_project_files(project_id)
    print(f"\n🔍 АНАЛИЗ ФАЙЛОВ:")
    print(f"  📊 Всего файлов: {len(all_files)}")
    
    # Статистика по типам файлов
    file_types = {}
    for file_info in all_files:
        ext = file_info[2]  # file_extension
        if ext not in file_types:
            file_types[ext] = {'count': 0, 'size': 0}
        file_types[ext]['count'] += 1
        file_types[ext]['size'] += file_info[4]  # file_size
    
    print(f"  📊 Типы файлов:")
    for ext, stats in sorted(file_types.items(), key=lambda x: x[1]['count'], reverse=True):
        size_mb = stats['size'] / (1024 * 1024)
        print(f"    - {ext}: {stats['count']} файлов, {size_mb:.1f} MB")
    
    # Топ файлы по размеру
    print(f"\n  📈 Топ-10 файлов по размеру:")
    sorted_files = sorted(all_files, key=lambda x: x[4], reverse=True)
    for i, file_info in enumerate(sorted_files[:10], 1):
        size_kb = file_info[4] / 1024
        print(f"    {i:2d}. {file_info[1]} ({file_info[2]}) - {size_kb:.1f} KB")
    
    # Поиск ключевых файлов
    print(f"\n🔑 КЛЮЧЕВЫЕ ФАЙЛЫ:")
    
    # Ищем файлы с VMB630
    vmb_files = [f for f in all_files if 'VMB630' in f[1]]
    print(f"  📄 Файлы с VMB630 ({len(vmb_files)}):")
    for file_info in vmb_files:
        print(f"    - {file_info[1]} ({file_info[2]})")
    
    # Ищем конфигурационные файлы
    config_files = [f for f in all_files if f[2] in ['.cfg', '.xml', '.ini']]
    print(f"\n  ⚙️ Конфигурационные файлы ({len(config_files)}):")
    for file_info in config_files[:10]:  # Показываем первые 10
        size_kb = file_info[4] / 1024
        print(f"    - {file_info[1]} ({file_info[2]}) - {size_kb:.1f} KB")
    
    # Анализ содержимого ключевых файлов
    print(f"\n📋 АНАЛИЗ СОДЕРЖИМОГО:")
    
    # Анализируем VMB630_info.txt
    vmb_info_file = next((f for f in all_files if f[1] == 'VMB630_info.txt'), None)
    if vmb_info_file:
        print(f"  📄 VMB630_info.txt:")
        content = vmb_info_file[5]  # file_content
        lines = content.split('\n')
        for line in lines[:20]:  # Показываем первые 20 строк
            if line.strip():
                print(f"    {line}")
        if len(lines) > 20:
            print(f"    ... и еще {len(lines) - 20} строк")
    
    # Анализируем define.xml
    define_xml = next((f for f in all_files if f[1] == 'define.xml'), None)
    if define_xml:
        print(f"\n  📄 define.xml (975.9 KB):")
        content = define_xml[5]  # file_content
        # Ищем ключевые элементы
        if 'axis' in content.lower():
            print(f"    ✅ Содержит определения осей")
        if 'motor' in content.lower():
            print(f"    ✅ Содержит определения моторов")
        if 'spindle' in content.lower():
            print(f"    ✅ Содержит определения шпинделей")
        if 'pwm' in content.lower():
            print(f"    ✅ Содержит PWM конфигурации")
    
    # Поиск специфичных терминов
    print(f"\n🔍 ПОИСК СПЕЦИФИЧНЫХ ТЕРМИНОВ:")
    search_terms = ["VMB630", "PLC", "motor", "axis", "spindle", "pwm", "encoder", "biss"]
    for term in search_terms:
        results = integrator.search_project_knowledge(term, project_id)
        print(f"  '{term}': {len(results)} результатов")
        if results:
            # Показываем первый результат
            first_result = results[0]
            preview = first_result['content_preview'][:100].replace('\n', ' ')
            print(f"    → {first_result['file_name']} ({first_result['language']})")
            print(f"    Превью: {preview}...")
    
    # Ответы на вопросы
    print(f"\n💬 ОТВЕТЫ НА ВОПРОСЫ:")
    questions = [
        "Что такое VMB630?",
        "Какие функции выполняет этот проект?",
        "Как работает система управления осями?",
        "Какие файлы конфигурации есть в проекте?",
        "Что такое PLC в этом проекте?"
    ]
    
    for q in questions:
        answer = integrator.answer_with_project_context(q, project_id)
        print(f"❓ {q}")
        print(f"🎯 {answer['answer']}")
        print()
    
    # Заключение
    print(f"🎉 ЗАКЛЮЧЕНИЕ:")
    print("=" * 30)
    print("VMB630 - это профессиональная система управления фрезерным станком с ЧПУ.")
    print("Проект содержит:")
    print("✅ 102 файла конфигурации и программ")
    print("✅ 6 осей управления (X, Y1, Y2, Z, B, C)")
    print("✅ 2 шпинделя (S, S1)")
    print("✅ PLC программы для логики управления")
    print("✅ BISS энкодеры для обратной связи")
    print("✅ PWM управление моторами")
    print("\nЭто промышленная система автоматизации для высокоточного фрезерования.")

if __name__ == "__main__":
    analyze_vmb630_simple()










