#!/usr/bin/env python3
"""
Проверка статуса нейронной сети в системе Rubin AI
"""

import sys
sys.path.append('.')

def check_neural_network_status():
    """Проверка статуса нейронной сети"""
    
    print("🧠 СТАТУС НЕЙРОННОЙ СЕТИ RUBIN AI")
    print("=" * 50)
    
    # Проверяем статус нейронной сети
    try:
        from neural_rubin import get_neural_rubin
        neural_ai = get_neural_rubin()
        stats = neural_ai.get_neural_stats()
        
        print("✅ НЕЙРОННАЯ СЕТЬ:")
        print(f"  • Устройство: {stats['device']}")
        print(f"  • Нейронная сеть активна: {stats['neural_network_active']}")
        print(f"  • Sentence модель активна: {stats['sentence_model_active']}")
        print(f"  • Параметров в модели: {stats['model_parameters']:,}")
        print(f"  • Диалогов: {stats['conversation_count']}")
        print(f"  • Категории: {len(stats['categories'])}")
        
        # Тестируем классификацию
        print("\n🧪 ТЕСТ КЛАССИФИКАЦИИ:")
        test_questions = [
            "2+3",
            "Что такое транзистор?",
            "Как работает Python?"
        ]
        
        for question in test_questions:
            try:
                category, confidence = neural_ai.classify_question(question)
                print(f"  • '{question}' -> {category} (уверенность: {confidence:.1%})")
            except Exception as e:
                print(f"  • '{question}' -> Ошибка: {e}")
        
    except Exception as e:
        print(f"❌ Ошибка нейронной сети: {e}")
    
    # Проверяем интеграцию с основным сервером
    print("\n🔗 ИНТЕГРАЦИЯ С СЕРВЕРОМ:")
    try:
        from api.rubin_ai_v2_server import ENHANCED_DISPATCHER_AVAILABLE, enhanced_dispatcher
        print(f"  • Улучшенный диспетчер доступен: {ENHANCED_DISPATCHER_AVAILABLE}")
        if enhanced_dispatcher:
            print("  • Диспетчер инициализирован")
        else:
            print("  • Диспетчер не инициализирован")
    except Exception as e:
        print(f"  • Ошибка интеграции: {e}")
    
    # Проверяем использование в Ultimate системе
    print("\n🚀 ИСПОЛЬЗОВАНИЕ В ULTIMATE СИСТЕМЕ:")
    try:
        from rubin_ultimate_system import RubinUltimateSystem
        ultimate_ai = RubinUltimateSystem()
        print(f"  • Ultimate система инициализирована")
        print(f"  • Математический решатель: {'✅' if ultimate_ai.math_solver_available else '❌'}")
        print(f"  • Тестовых документов: {len(ultimate_ai.test_documents)}")
        print(f"  • Элементов из БД: {len(ultimate_ai.database_content)}")
    except Exception as e:
        print(f"  • Ошибка Ultimate системы: {e}")

def show_neural_architecture():
    """Показать архитектуру нейронной сети"""
    
    print("\n🏗️ АРХИТЕКТУРА НЕЙРОННОЙ СЕТИ:")
    print("=" * 40)
    
    print("""
    📥 ВХОДНЫЕ ДАННЫЕ:
    ├── Текст вопроса пользователя
    ├── Sentence Transformer (all-MiniLM-L6-v2)
    └── Эмбеддинг 384 измерения
    
    🧠 НЕЙРОННАЯ СЕТЬ:
    ├── Входной слой: 384 нейрона
    ├── Скрытый слой 1: 512 нейронов + ReLU + Dropout(0.2)
    ├── Скрытый слой 2: 256 нейронов + ReLU + Dropout(0.2)
    ├── Скрытый слой 3: 128 нейронов + ReLU
    ├── Выходной слой: 10 нейронов (категории)
    └── Softmax классификатор
    
    📤 ВЫХОДНЫЕ ДАННЫЕ:
    ├── Категория вопроса (0-9)
    ├── Уверенность классификации (0-1)
    └── Метаданные для принятия решений
    
    🎯 КАТЕГОРИИ:
    ├── 0: математика
    ├── 1: физика
    ├── 2: электротехника
    ├── 3: программирование
    ├── 4: геометрия
    ├── 5: химия
    ├── 6: общие_вопросы
    ├── 7: техника
    ├── 8: наука
    └── 9: другое
    """)

def show_decision_flow():
    """Показать поток принятия решений"""
    
    print("\n🔄 ПОТОК ПРИНЯТИЯ РЕШЕНИЙ:")
    print("=" * 40)
    
    print("""
    1️⃣ ПОЛУЧЕНИЕ ВОПРОСА:
       └── Пользователь отправляет сообщение
    
    2️⃣ ПРЕДВАРИТЕЛЬНАЯ ОБРАБОТКА:
       ├── Очистка текста
       ├── Нормализация
       └── Создание эмбеддинга
    
    3️⃣ НЕЙРОННАЯ КЛАССИФИКАЦИЯ:
       ├── Прямой проход через сеть
       ├── Получение категории
       └── Расчет уверенности
    
    4️⃣ ПРИНЯТИЕ РЕШЕНИЯ:
       ├── Если уверенность > 0.7 → Использовать категорию
       ├── Если уверенность < 0.7 → Fallback к простой логике
       └── Если ошибка → Общий ответ
    
    5️⃣ ГЕНЕРАЦИЯ ОТВЕТА:
       ├── Математика → Математический решатель
       ├── Электротехника → База знаний по электронике
       ├── Программирование → База знаний по программированию
       └── Другое → Общий ответ
    
    6️⃣ ОБУЧЕНИЕ:
       ├── Сохранение в историю
       ├── Сбор обратной связи
       └── Обновление модели
    """)

if __name__ == "__main__":
    check_neural_network_status()
    show_neural_architecture()
    show_decision_flow()
    
    print("\n🎯 ВЫВОДЫ:")
    print("1. Нейронная сеть используется для классификации вопросов")
    print("2. На основе классификации выбирается стратегия ответа")
    print("3. Система имеет fallback механизмы при ошибках")
    print("4. Поддерживается обучение на основе обратной связи")
    print("5. Интеграция с различными компонентами системы")

















