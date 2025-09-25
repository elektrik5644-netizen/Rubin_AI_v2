#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🔍 АНАЛИЗАТОР PLC ФАЙЛОВ VMB630
===============================
Анализирует PLC файлы и находит синтаксические и логические ошибки
"""

import re
import os

def analyze_plc_file(file_path):
    """Анализ PLC файла на наличие ошибок"""
    print(f"🔍 АНАЛИЗ PLC ФАЙЛА: {os.path.basename(file_path)}")
    print("=" * 60)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"❌ Ошибка чтения файла: {e}")
        return
    
    lines = content.split('\n')
    errors = []
    warnings = []
    
    # Анализ синтаксических ошибок
    print("\n📋 СИНТАКСИЧЕСКИЙ АНАЛИЗ:")
    print("-" * 30)
    
    # Проверка баланса скобок и операторов
    if_count = 0
    endif_count = 0
    while_count = 0
    endwhile_count = 0
    
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith(';'):
            continue
            
        # Подсчет операторов
        if_count += line.count('IF(') + line.count('IF (')
        endif_count += line.count('ENDIF')
        while_count += line.count('WHILE(') + line.count('WHILE (')
        endwhile_count += line.count('ENDWHILE')
        
        # Проверка синтаксиса
        if 'IF(' in line and not line.endswith(')'):
            errors.append(f"Строка {i}: Незакрытая скобка в IF")
        
        # Проверка на опечатки в переменных
        if 'AXIS_DISCONNECTEP_TP_P' in line:
            errors.append(f"Строка {i}: Опечатка в переменной AXIS_DISCONNECTEP_TP_P (должно быть AXIS_DISCONNECTED_TP_P)")
        
        # Проверка на несоответствие таймеров
        if 'TIMER_SIMPLE77_P' in line and 'SOJ_PUMP_PISTOL_STAGE_P' in line:
            warnings.append(f"Строка {i}: Возможно несоответствие таймера TIMER_SIMPLE77_P в секции пистолета СОЖ")
    
    # Проверка баланса операторов
    if if_count != endif_count:
        errors.append(f"Несоответствие IF/ENDIF: IF={if_count}, ENDIF={endif_count}")
    
    if while_count != endwhile_count:
        errors.append(f"Несоответствие WHILE/ENDWHILE: WHILE={while_count}, ENDWHILE={endwhile_count}")
    
    # Анализ логических ошибок
    print("\n🧠 ЛОГИЧЕСКИЙ АНАЛИЗ:")
    print("-" * 30)
    
    # Проверка стадий смазки
    lube_stages = []
    for i, line in enumerate(lines, 1):
        if 'LUBE_STAGE_P =' in line:
            stage_match = re.search(r'LUBE_STAGE_P = (\d+)', line)
            if stage_match:
                stage = int(stage_match.group(1))
                lube_stages.append((i, stage))
    
    # Проверка переходов между стадиями
    valid_transitions = {0: [1], 1: [2], 2: [3], 3: [4, 1000], 4: [5], 5: [0], 1000: [0]}
    for i, (line_num, stage) in enumerate(lube_stages):
        if i < len(lube_stages) - 1:
            next_stage = lube_stages[i + 1][1]
            if stage in valid_transitions and next_stage not in valid_transitions[stage]:
                warnings.append(f"Строка {line_num}: Возможно некорректный переход стадии смазки {stage} → {next_stage}")
    
    # Проверка таймеров
    timer_usage = {}
    for i, line in enumerate(lines, 1):
        timer_matches = re.findall(r'TIMER_SIMPLE(\d+)_P', line)
        for timer in timer_matches:
            if timer not in timer_usage:
                timer_usage[timer] = []
            timer_usage[timer].append(i)
    
    # Проверка на конфликты таймеров
    for timer, lines_used in timer_usage.items():
        if len(lines_used) > 3:  # Если таймер используется в более чем 3 местах
            warnings.append(f"Таймер TIMER_SIMPLE{timer}_P используется в {len(lines_used)} местах: строки {lines_used}")
    
    # Вывод результатов
    print(f"📊 СТАТИСТИКА:")
    print(f"  Всего строк: {len(lines)}")
    print(f"  Операторов IF: {if_count}")
    print(f"  Операторов ENDIF: {endif_count}")
    print(f"  Операторов WHILE: {while_count}")
    print(f"  Операторов ENDWHILE: {endwhile_count}")
    print(f"  Стадий смазки: {len(lube_stages)}")
    print(f"  Используемых таймеров: {len(timer_usage)}")
    
    print(f"\n❌ НАЙДЕННЫЕ ОШИБКИ ({len(errors)}):")
    if errors:
        for error in errors:
            print(f"  • {error}")
    else:
        print("  ✅ Синтаксических ошибок не найдено")
    
    print(f"\n⚠️ ПРЕДУПРЕЖДЕНИЯ ({len(warnings)}):")
    if warnings:
        for warning in warnings:
            print(f"  • {warning}")
    else:
        print("  ✅ Предупреждений нет")
    
    # Анализ функциональности
    print(f"\n🔧 ФУНКЦИОНАЛЬНЫЙ АНАЛИЗ:")
    print("-" * 30)
    
    functions = {
        "Система смазки": 0,
        "Управление освещением": 0,
        "Управление СОЖ": 0,
        "Управление транспортёром": 0,
        "Управление поджимом подшипников": 0,
        "Аналоговые выходы": 0
    }
    
    for line in content:
        if "LUBE_STAGE_P" in line or "PULSE_LUBE" in line:
            functions["Система смазки"] += 1
        if "LED_WORKPLACE" in line or "LIGHT_WORKZONE" in line:
            functions["Управление освещением"] += 1
        if "SOJ_PUMP" in line or "SOJ_" in line:
            functions["Управление СОЖ"] += 1
        if "TRANSPORT" in line:
            functions["Управление транспортёром"] += 1
        if "BEAR_TIGHT" in line or "PRELOAD_BEAR" in line:
            functions["Управление поджимом подшипников"] += 1
        if "ANALOG" in line or "_CMD_LIM" in line:
            functions["Аналоговые выходы"] += 1
    
    print("  Функциональные блоки:")
    for func, count in functions.items():
        if count > 0:
            print(f"    ✅ {func}: {count} упоминаний")
    
    # Рекомендации
    print(f"\n💡 РЕКОМЕНДАЦИИ:")
    print("-" * 30)
    
    if errors:
        print("  🔧 Исправить синтаксические ошибки:")
        for error in errors:
            if "AXIS_DISCONNECTEP_TP_P" in error:
                print("    • Заменить AXIS_DISCONNECTEP_TP_P на AXIS_DISCONNECTED_TP_P")
    
    if warnings:
        print("  ⚠️ Проверить логику:")
        for warning in warnings:
            if "таймер" in warning.lower():
                print("    • Проверить корректность использования таймеров")
    
    print("  📚 Общие рекомендации:")
    print("    • Добавить комментарии к сложным логическим блокам")
    print("    • Проверить все переходы между стадиями")
    print("    • Убедиться в корректности работы таймеров")
    print("    • Протестировать все функциональные блоки")

def main():
    """Основная функция"""
    file_path = r"C:\Users\elekt\OneDrive\Desktop\VMB630_v_005_019_000\out\plc_18_background_ctrl.plc"
    
    if not os.path.exists(file_path):
        print(f"❌ Файл не найден: {file_path}")
        return
    
    analyze_plc_file(file_path)
    
    print(f"\n🎯 ЗАКЛЮЧЕНИЕ:")
    print("=" * 30)
    print("PLC файл содержит логику управления фоновыми процессами станка VMB630.")
    print("Основные функции: система смазки, управление СОЖ, освещение, транспортёр.")
    print("Рекомендуется исправить найденные ошибки перед использованием.")

if __name__ == "__main__":
    main()





