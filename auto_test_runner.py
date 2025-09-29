#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🤖 Auto Test Runner - Автоматический запуск тестов для Rubin AI v2
Запускается когда Rubin остается без подключения к сети
"""

import subprocess
import time
import json
import os
from datetime import datetime

def run_quick_test():
    """Запускает быстрое тестирование"""
    print("🤖 Автоматический запуск тестов Rubin AI v2")
    print("=" * 50)
    
    try:
        # Запускаем быстрое тестирование
        result = subprocess.run(['python', 'quick_api_test.py'], 
                              capture_output=True, text=True, timeout=60)
        
        print("📊 Результаты быстрого тестирования:")
        print(result.stdout)
        
        if result.stderr:
            print("⚠️ Предупреждения:")
            print(result.stderr)
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("⏰ Тестирование превысило лимит времени")
        return False
    except Exception as e:
        print(f"❌ Ошибка запуска тестов: {e}")
        return False

def run_comprehensive_test():
    """Запускает комплексное тестирование"""
    print("\n🧪 Запуск комплексного тестирования...")
    
    try:
        result = subprocess.run(['python', 'api_test_suite.py'], 
                              capture_output=True, text=True, timeout=300)
        
        print("📊 Результаты комплексного тестирования:")
        print(result.stdout)
        
        if result.stderr:
            print("⚠️ Предупреждения:")
            print(result.stderr)
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("⏰ Комплексное тестирование превысило лимит времени")
        return False
    except Exception as e:
        print(f"❌ Ошибка комплексного тестирования: {e}")
        return False

def save_test_report(quick_success, comprehensive_success):
    """Сохраняет отчет о тестировании"""
    report = {
        'timestamp': datetime.now().isoformat(),
        'quick_test_success': quick_success,
        'comprehensive_test_success': comprehensive_success,
        'overall_status': 'PASS' if (quick_success and comprehensive_success) else 'FAIL'
    }
    
    try:
        with open('auto_test_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print("📄 Отчет сохранен в auto_test_report.json")
    except Exception as e:
        print(f"❌ Ошибка сохранения отчета: {e}")

def main():
    """Главная функция автоматического тестирования"""
    print("🤖 Rubin AI v2 - Автоматическое тестирование")
    print("Запуск без подключения к сети")
    print("=" * 50)
    
    # Быстрое тестирование
    quick_success = run_quick_test()
    
    # Комплексное тестирование (только если быстрое прошло успешно)
    comprehensive_success = False
    if quick_success:
        comprehensive_success = run_comprehensive_test()
    else:
        print("⚠️ Пропускаем комплексное тестирование из-за проблем с быстрым тестом")
    
    # Сохраняем отчет
    save_test_report(quick_success, comprehensive_success)
    
    # Итоговая оценка
    print("\n" + "=" * 50)
    print("🏁 АВТОМАТИЧЕСКОЕ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 50)
    
    if quick_success and comprehensive_success:
        print("🎉 ВСЕ ТЕСТЫ ПРОШЛИ УСПЕШНО!")
        print("✅ Система Rubin AI v2 работает корректно")
    elif quick_success:
        print("👍 БЫСТРЫЕ ТЕСТЫ ПРОШЛИ")
        print("⚠️ Комплексное тестирование не выполнено")
    else:
        print("🚨 ОБНАРУЖЕНЫ ПРОБЛЕМЫ!")
        print("❌ Система требует внимания")
    
    print("=" * 50)
    
    return quick_success and comprehensive_success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)








