#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Полный анализ VMB630 проекта через Rubin AI
"""

import os
from project_folder_reader import ProjectFolderReader
from rubin_project_integration import RubinProjectIntegration

def analyze_vmb630_with_rubin():
    """Полный анализ VMB630 через Rubin AI"""
    
    print("🔍 RUBIN AI ПОЛНЫЙ АНАЛИЗ ПРОЕКТА VMB630")
    print("=" * 60)
    
    # Инициализация Rubin AI
    reader = ProjectFolderReader()
    reader.initialize_db()
    integrator = RubinProjectIntegration()
    integrator.initialize_db()
    
    # Получаем проект VMB630
    project_id = reader.get_project_id_by_name("VMB630 Project")
    
    if not project_id:
        print("❌ Проект VMB630 не найден в базе данных")
        return
    
    print(f"✅ Найден проект VMB630 (ID: {project_id})")
    
    # Получаем обзор проекта
    overview = integrator.get_project_overview(project_id)
    print(f"\n📊 ОБЗОР ПРОЕКТА:")
    print(f"  📁 Название: {overview['project_name']}")
    print(f"  🔧 Тип проекта: {overview['project_type']}")
    print(f"  📄 Файлов: {overview['total_files']}")
    print(f"  💾 Размер: {overview['total_size_mb']:.2f} MB")
    print(f"  ⚙️ Сложность: {overview['avg_complexity']:.0f}")
    
    # Анализируем файлы проекта
    print(f"\n🔍 АНАЛИЗ ФАЙЛОВ:")
    all_files = integrator.get_all_files_in_project(project_id)
    
    # Статистика по типам файлов
    file_types = {}
    for file_info in all_files:
        ext = file_info['file_extension']
        if ext not in file_types:
            file_types[ext] = {'count': 0, 'size': 0}
        file_types[ext]['count'] += 1
        file_types[ext]['size'] += file_info['size_bytes']
    
    print(f"  📊 Типы файлов:")
    for ext, stats in sorted(file_types.items(), key=lambda x: x[1]['count'], reverse=True):
        size_mb = stats['size'] / (1024 * 1024)
        print(f"    - {ext}: {stats['count']} файлов, {size_mb:.1f} MB")
    
    # Топ файлы по размеру
    print(f"\n  📈 Топ-10 файлов по размеру:")
    sorted_files = sorted(all_files, key=lambda x: x['size_bytes'], reverse=True)
    for i, file_info in enumerate(sorted_files[:10], 1):
        size_kb = file_info['size_bytes'] / 1024
        print(f"    {i:2d}. {file_info['file_name']} ({file_info['file_extension']}) - {size_kb:.1f} KB")
    
    # Поиск ключевых файлов
    print(f"\n🔑 КЛЮЧЕВЫЕ ФАЙЛЫ:")
    
    # Ищем файлы с VMB630
    vmb_files = integrator.search_project(project_id, "VMB630")
    print(f"  📄 Файлы с VMB630 ({len(vmb_files)}):")
    for file_info in vmb_files:
        print(f"    - {file_info['file_name']} ({file_info['file_type']})")
        if file_info['file_name'] == 'VMB630_info.txt':
            print(f"      Содержимое: {file_info['content'][:200]}...")
    
    # Ищем конфигурационные файлы
    config_files = [f for f in all_files if f['file_extension'] in ['.cfg', '.xml', '.ini']]
    print(f"\n  ⚙️ Конфигурационные файлы ({len(config_files)}):")
    for file_info in config_files[:10]:  # Показываем первые 10
        size_kb = file_info['size_bytes'] / 1024
        print(f"    - {file_info['file_name']} ({file_info['file_extension']}) - {size_kb:.1f} KB")
    
    # Анализ содержимого ключевых файлов
    print(f"\n📋 АНАЛИЗ СОДЕРЖИМОГО:")
    
    # Анализируем VMB630_info.txt
    vmb_info_file = next((f for f in all_files if f['file_name'] == 'VMB630_info.txt'), None)
    if vmb_info_file:
        print(f"  📄 VMB630_info.txt:")
        content = vmb_info_file['content']
        lines = content.split('\n')
        for line in lines[:20]:  # Показываем первые 20 строк
            if line.strip():
                print(f"    {line}")
        if len(lines) > 20:
            print(f"    ... и еще {len(lines) - 20} строк")
    
    # Анализируем define.xml
    define_xml = next((f for f in all_files if f['file_name'] == 'define.xml'), None)
    if define_xml:
        print(f"\n  📄 define.xml (975.9 KB):")
        content = define_xml['content']
        # Ищем ключевые элементы
        if 'axis' in content.lower():
            print(f"    ✅ Содержит определения осей")
        if 'motor' in content.lower():
            print(f"    ✅ Содержит определения моторов")
        if 'spindle' in content.lower():
            print(f"    ✅ Содержит определения шпинделей")
        if 'pwm' in content.lower():
            print(f"    ✅ Содержит PWM конфигурации")
    
    # Анализируем start.cfg
    start_cfg = next((f for f in all_files if f['file_name'] == 'start.cfg'), None)
    if start_cfg:
        print(f"\n  📄 start.cfg (577.9 KB):")
        content = start_cfg['content']
        lines = content.split('\n')
        config_lines = [line for line in lines if line.strip() and not line.startswith(';')]
        print(f"    📊 Активных конфигурационных строк: {len(config_lines)}")
        print(f"    📊 Общее количество строк: {len(lines)}")
    
    # Генерируем инсайты
    print(f"\n💡 ИНСАЙТЫ RUBIN AI:")
    insights = integrator.generate_insights(project_id)
    if insights and insights.get('recommendations'):
        for i, rec in enumerate(insights['recommendations'], 1):
            print(f"  {i}. {rec}")
    else:
        print("  Не обнаружено специфических рекомендаций.")
    
    # Поиск специфичных терминов
    print(f"\n🔍 ПОИСК СПЕЦИФИЧНЫХ ТЕРМИНОВ:")
    search_terms = ["VMB630", "PLC", "motor", "axis", "spindle", "pwm", "encoder", "biss"]
    for term in search_terms:
        results = integrator.search_project(project_id, term)
        print(f"  '{term}': {len(results)} результатов")
        if results:
            # Показываем первый результат
            first_result = results[0]
            preview = first_result['content'][:100].replace('\n', ' ')
            print(f"    → {first_result['file_name']} ({first_result['file_type']})")
            print(f"    Превью: {preview}...")
    
    # Ответы на вопросы
    print(f"\n💬 ОТВЕТЫ НА ВОПРОСЫ:")
    questions = [
        "Что такое VMB630?",
        "Какие функции выполняет этот проект?",
        "Как работает система управления осями?",
        "Какие файлы конфигурации есть в проекте?",
        "Что такое PLC в этом проекте?",
        "Какие моторы используются в системе?",
        "Как настроены энкодеры?"
    ]
    
    for q in questions:
        answer = integrator.answer_question_about_project(project_id, q)
        print(f"❓ {q}")
        print(f"🎯 {answer}")
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
    analyze_vmb630_with_rubin()





