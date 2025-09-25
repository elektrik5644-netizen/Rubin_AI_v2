#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Запуск исправленного Rubin AI сервера
"""

import sys
import os
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Запуск исправленного сервера"""
    print("🚀 ЗАПУСК ИСПРАВЛЕННОГО RUBIN AI")
    print("=" * 50)
    
    # Проверяем компоненты
    print("🔍 Проверка компонентов...")
    
    try:
        from enhanced_request_categorizer import get_enhanced_categorizer
        print("✅ Улучшенный категоризатор")
    except ImportError as e:
        print(f"❌ Улучшенный категоризатор: {e}")
    
    try:
        from programming_knowledge_handler import get_programming_handler
        print("✅ Обработчик программирования")
    except ImportError as e:
        print(f"❌ Обработчик программирования: {e}")
    
    try:
        from electrical_knowledge_handler import get_electrical_handler
        print("✅ Обработчик электротехники")
    except ImportError as e:
        print(f"❌ Обработчик электротехники: {e}")
    
    try:
        from intelligent_dispatcher import get_intelligent_dispatcher
        print("✅ Интеллектуальный диспетчер")
    except ImportError as e:
        print(f"❌ Интеллектуальный диспетчер: {e}")
    
    try:
        from neural_rubin import get_neural_rubin
        print("✅ Нейронная сеть")
    except ImportError as e:
        print(f"❌ Нейронная сеть: {e}")
    
    print("\n🌐 Запуск Flask сервера...")
    print("📍 Сервер будет доступен на: http://localhost:5000")
    print("🔗 RubinDeveloper: file:///C:/Users/elekt/OneDrive/Desktop/Rubin_AI_v2/matrix/RubinDeveloper.html")
    print("\n🧪 ТЕСТОВЫЕ ЗАПРОСЫ:")
    print("• Сравни C++ и Python для задач промышленной автоматизации")
    print("• Как защитить электрические цепи от короткого замыкания?")
    print("• 2 + 3 = ?")
    print("• Привет, как дела?")
    print("\n" + "=" * 50)
    
    # Запускаем основной сервер
    try:
        from rubin_server import app
        app.run(host='0.0.0.0', port=5000, debug=False)
    except Exception as e:
        print(f"❌ Ошибка запуска сервера: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()