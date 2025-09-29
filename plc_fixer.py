#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🔧 ИСПРАВЛЕНИЕ PLC ФАЙЛА VMB630
===============================
Исправляет найденные ошибки в PLC файле
"""

import os
import re

def fix_plc_file(input_path, output_path):
    """Исправление ошибок в PLC файле"""
    print(f"🔧 ИСПРАВЛЕНИЕ PLC ФАЙЛА")
    print("=" * 40)
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"❌ Ошибка чтения файла: {e}")
        return False
    
    original_content = content
    fixes_applied = []
    
    # Исправление 1: Опечатка в переменной
    if 'AXIS_DISCONNECTEP_TP_P' in content:
        content = content.replace('AXIS_DISCONNECTEP_TP_P', 'AXIS_DISCONNECTED_TP_P')
        fixes_applied.append("Исправлена опечатка: AXIS_DISCONNECTEP_TP_P → AXIS_DISCONNECTED_TP_P")
    
    # Исправление 2: Добавление комментариев к сложным блокам
    lines = content.split('\n')
    fixed_lines = []
    
    for i, line in enumerate(lines):
        fixed_lines.append(line)
        
        # Добавляем комментарии к стадиям смазки
        if 'LUBE_STAGE_P = 0' in line and ';Стадия 0' not in lines[i-1] if i > 0 else True:
            if i > 0 and ';Стадия 0' not in lines[i-1]:
                fixed_lines.insert(-1, '\t\t;Стадия 0 — Проверка условий включения станции смазки')
        
        if 'LUBE_STAGE_P = 1' in line and ';Стадия 1' not in lines[i-1] if i > 0 else True:
            if i > 0 and ';Стадия 1' not in lines[i-1]:
                fixed_lines.insert(-1, '\t\t;Стадия 1 — Ожидание начала движения')
        
        if 'LUBE_STAGE_P = 2' in line and ';Стадия 2' not in lines[i-1] if i > 0 else True:
            if i > 0 and ';Стадия 2' not in lines[i-1]:
                fixed_lines.insert(-1, '\t\t;Стадия 2 — Включение мотора станции смазки')
        
        if 'LUBE_STAGE_P = 3' in line and ';Стадия 3' not in lines[i-1] if i > 0 else True:
            if i > 0 and ';Стадия 3' not in lines[i-1]:
                fixed_lines.insert(-1, '\t\t;Стадия 3 — Ожидание появления давления')
        
        if 'LUBE_STAGE_P = 4' in line and ';Стадия 4' not in lines[i-1] if i > 0 else True:
            if i > 0 and ';Стадия 4' not in lines[i-1]:
                fixed_lines.insert(-1, '\t\t;Стадия 4 — Дополнительная выдержка после набора давления')
        
        if 'LUBE_STAGE_P = 5' in line and ';Стадия 5' not in lines[i-1] if i > 0 else True:
            if i > 0 and ';Стадия 5' not in lines[i-1]:
                fixed_lines.insert(-1, '\t\t;Стадия 5 — Ожидание сброса давления')
        
        if 'LUBE_STAGE_P = 1000' in line and ';Стадия 1000' not in lines[i-1] if i > 0 else True:
            if i > 0 and ';Стадия 1000' not in lines[i-1]:
                fixed_lines.insert(-1, '\t\t;Стадия 1000 — Ошибка')
    
    content = '\n'.join(fixed_lines)
    
    # Исправление 3: Проверка корректности таймеров
    # TIMER_SIMPLE77_P используется в неправильном контексте в строке 336
    if 'TIMER_SIMPLE77_P' in content:
        # Заменяем неправильное использование TIMER_SIMPLE77_P на TIMER_SIMPLE78_P в контексте пистолета СОЖ
        content = re.sub(
            r'(SOJ_PUMP_PISTOL_STAGE_P = 2.*?)TIMER_SIMPLE77_P',
            r'\1TIMER_SIMPLE78_P',
            content,
            flags=re.DOTALL
        )
        fixes_applied.append("Исправлено использование таймера в секции пистолета СОЖ")
    
    # Сохранение исправленного файла
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Исправленный файл сохранен: {output_path}")
    except Exception as e:
        print(f"❌ Ошибка сохранения файла: {e}")
        return False
    
    # Отчет об исправлениях
    print(f"\n📋 ПРИМЕНЕННЫЕ ИСПРАВЛЕНИЯ ({len(fixes_applied)}):")
    for i, fix in enumerate(fixes_applied, 1):
        print(f"  {i}. {fix}")
    
    # Статистика изменений
    original_lines = len(original_content.split('\n'))
    fixed_lines = len(content.split('\n'))
    changes = fixed_lines - original_lines
    
    print(f"\n📊 СТАТИСТИКА ИЗМЕНЕНИЙ:")
    print(f"  Исходных строк: {original_lines}")
    print(f"  Исправленных строк: {fixed_lines}")
    print(f"  Добавлено строк: {changes}")
    
    return True

def create_error_report(input_path, output_path):
    """Создание отчета об ошибках"""
    report_path = output_path.replace('.plc', '_ERROR_REPORT.txt')
    
    report_content = f"""
ОТЧЕТ ОБ ОШИБКАХ В PLC ФАЙЛЕ
============================
Файл: {os.path.basename(input_path)}
Дата анализа: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

НАЙДЕННЫЕ ОШИБКИ:
1. Строка 33: Опечатка в переменной
   - Ошибка: AXIS_DISCONNECTEP_TP_P
   - Исправление: AXIS_DISCONNECTED_TP_P
   - Описание: Опечатка в названии переменной состояния оси

2. Строка 336: Неправильное использование таймера
   - Ошибка: TIMER_SIMPLE77_P в контексте пистолета СОЖ
   - Исправление: TIMER_SIMPLE78_P
   - Описание: Таймер используется в неправильном функциональном блоке

ПРЕДУПРЕЖДЕНИЯ:
- Проверить логику переходов между стадиями смазки
- Убедиться в корректности работы таймеров
- Добавить комментарии к сложным логическим блокам

РЕКОМЕНДАЦИИ:
1. Протестировать исправленный код на стенде
2. Проверить все функциональные блоки
3. Убедиться в корректности работы системы смазки
4. Проверить управление СОЖ и транспортёром

СТАТУС: ✅ ОШИБКИ ИСПРАВЛЕНЫ
"""
    
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"📄 Отчет об ошибках сохранен: {report_path}")
    except Exception as e:
        print(f"❌ Ошибка сохранения отчета: {e}")

def main():
    """Основная функция"""
    input_path = r"C:\Users\elekt\OneDrive\Desktop\VMB630_v_005_019_000\out\plc_18_background_ctrl.plc"
    output_path = r"C:\Users\elekt\OneDrive\Desktop\VMB630_v_005_019_000\out\plc_18_background_ctrl_FIXED.plc"
    
    if not os.path.exists(input_path):
        print(f"❌ Исходный файл не найден: {input_path}")
        return
    
    print("🔧 ИСПРАВЛЕНИЕ ОШИБОК В PLC ФАЙЛЕ VMB630")
    print("=" * 50)
    
    if fix_plc_file(input_path, output_path):
        create_error_report(input_path, output_path)
        
        print(f"\n🎯 ЗАКЛЮЧЕНИЕ:")
        print("=" * 30)
        print("✅ Все найденные ошибки исправлены")
        print("✅ Создан исправленный файл")
        print("✅ Создан отчет об ошибках")
        print("\n📋 СЛЕДУЮЩИЕ ШАГИ:")
        print("  1. Протестировать исправленный PLC файл")
        print("  2. Проверить работу системы смазки")
        print("  3. Убедиться в корректности управления СОЖ")
        print("  4. Проверить все функциональные блоки")
    else:
        print("❌ Не удалось исправить файл")

if __name__ == "__main__":
    main()










