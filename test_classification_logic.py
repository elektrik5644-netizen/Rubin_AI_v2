#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тест логики классификации
"""

def test_classification_logic():
    """Тестирует логику классификации напрямую"""
    
    print("🧪 Тест логики классификации")
    print("=" * 60)
    
    # Копируем SERVERS из smart_dispatcher.py
    SERVERS = {
        'electrical': {
            'port': 8087,
            'endpoint': '/api/electrical/solve',
            'keywords': ['закон', 'кирхгофа', 'резистор', 'резисторы', 'транзистор', 'транзисторы', 'диод', 'диоды', 'конденсатор', 'конденсаторы', 'контактор', 'реле', 'мощность', 'ток', 'напряжение', 'схема', 'схемы', 'электрические', 'электричество', 'цепи', 'тиристор', 'симистр', 'ом', 'закон ома', 'электрическая цепь', 'сопротивление', 'катушка', 'индуктивность', 'емкость', 'коэффициент мощности', 'power factor', 'cos φ', 'cosφ', 'реактивная мощность', 'как работает', 'как устроен', 'принцип работы', 'электротехника', 'электроника', 'электронные компоненты']
        },
        'physics': {
            'port': 8110,
            'endpoint': '/api/physics/explain',
            'keywords': ['фотон', 'электрон', 'протон', 'нейтрон', 'атом', 'молекула', 'квант', 'квантовая', 'физика', 'механика', 'термодинамика', 'оптика', 'электродинамика', 'ядерная физика', 'релятивистская', 'эйнштейн', 'ньютон', 'законы ньютона', 'гравитация', 'магнетизм', 'электромагнитное поле', 'волна', 'частица', 'энергия', 'масса', 'скорость света', 'планк', 'бозон', 'фермион', 'спин', 'орбиталь', 'изотоп', 'радиоактивность', 'ядерная реакция', 'синтез', 'деление', 'плазма', 'сверхпроводимость', 'криогеника', 'лазер', 'полупроводник', 'диэлектрик', 'проводник', 'изолятор', 'что такое', 'что такой', 'объясни', 'расскажи']
        },
        'general': {
            'port': 8085,
            'endpoint': '/api/chat',
            'keywords': ['привет', 'hello', 'hi', 'здравствуй', 'помощь', 'help', 'справка', 'статус', 'status', 'работает', 'онлайн', 'как', 'объясни', 'расскажи']
        }
    }
    
    def categorize_message(message):
        """Определяет категорию сообщения"""
        message_lower = message.lower()
        
        # Подсчитываем совпадения для каждой категории
        scores = {}
        for category, config in SERVERS.items():
            score = sum(1 for keyword in config['keywords'] if keyword in message_lower)
            scores[category] = score
        
        print(f"📊 Счетчики для '{message}':")
        for cat, score in scores.items():
            if score > 0:
                print(f"   {cat}: {score}")
        
        # Приоритет для технических терминов
        technical_categories = ['electrical', 'physics']
        technical_scores = {cat: scores.get(cat, 0) for cat in technical_categories if scores.get(cat, 0) > 0}
        
        print(f"🔧 Технические счетчики: {technical_scores}")
        
        # Специальная логика для физики - приоритет над electrical
        if 'фотон' in message_lower or 'электрон' in message_lower or 'атом' in message_lower or 'квант' in message_lower:
            if 'physics' in technical_scores and technical_scores['physics'] > 0:
                print(f"⚛️ Приоритет физики сработал!")
                return 'physics'
        
        if technical_scores:
            # Если есть технические совпадения, выбираем лучший технический
            best_technical = max(technical_scores, key=technical_scores.get)
            print(f"🎯 Выбрана техническая категория: {best_technical}")
            return best_technical
        
        # Возвращаем категорию с наибольшим количеством совпадений
        if scores and max(scores.values()) > 0:
            best_category = max(scores, key=scores.get)
            print(f"📈 Выбрана категория по счетчику: {best_category}")
            return best_category
        
        # Если нет совпадений, возвращаем general как fallback
        print(f"❓ Fallback к general")
        return 'general'
    
    # Тестовые вопросы
    test_questions = [
        "что такое фотон?",
        "расскажи про электрон",
        "что такое атом?",
        "объясни квантовую механику"
    ]
    
    for question in test_questions:
        print(f"\n📝 Вопрос: {question}")
        result = categorize_message(question)
        print(f"✅ Результат: {result}")
        print("-" * 40)

if __name__ == '__main__':
    test_classification_logic()



