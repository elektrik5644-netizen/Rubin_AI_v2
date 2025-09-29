#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Physics Server - сервер для физических вопросов
"""

import logging
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# База знаний по физике
PHYSICS_KNOWLEDGE = {
    'фотон': {
        'definition': 'Фотон — элементарная частица, квант электромагнитного излучения',
        'properties': [
            'Не имеет массы покоя',
            'Движется со скоростью света',
            'Имеет энергию E = hν',
            'Имеет импульс p = h/λ',
            'Может проявлять свойства как частицы, так и волны'
        ],
        'applications': [
            'Лазеры',
            'Солнечные батареи',
            'Фотоэлементы',
            'Оптическая связь',
            'Медицинская диагностика'
        ]
    },
    'электрон': {
        'definition': 'Электрон — элементарная частица с отрицательным электрическим зарядом',
        'properties': [
            'Масса: 9.109 × 10⁻³¹ кг',
            'Заряд: -1.602 × 10⁻¹⁹ Кл',
            'Спин: 1/2',
            'Фермион',
            'Участвует в химических связях'
        ],
        'applications': [
            'Электроника',
            'Химические реакции',
            'Электрический ток',
            'Катодные лучи',
            'Электронные микроскопы'
        ]
    },
    'атом': {
        'definition': 'Атом — наименьшая частица химического элемента',
        'structure': [
            'Ядро (протоны + нейтроны)',
            'Электронная оболочка',
            'Размер: ~10⁻¹⁰ м',
            'Масса сосредоточена в ядре'
        ],
        'models': [
            'Модель Резерфорда',
            'Модель Бора',
            'Квантовая модель',
            'Электронные орбитали'
        ]
    },
    'квантовая механика': {
        'definition': 'Раздел физики, изучающий поведение материи на атомном и субатомном уровне',
        'principles': [
            'Принцип неопределенности Гейзенберга',
            'Квантовая суперпозиция',
            'Квантовая запутанность',
            'Волновая функция',
            'Квантование энергии'
        ],
        'applications': [
            'Лазеры',
            'Сверхпроводимость',
            'Квантовые компьютеры',
            'Атомные часы',
            'МРТ'
        ]
    }
}

def find_physics_concept(query):
    """Находит физическое понятие в базе знаний"""
    query_lower = query.lower()
    
    # Прямой поиск
    for concept, data in PHYSICS_KNOWLEDGE.items():
        if concept in query_lower:
            return concept, data
    
    # Поиск по ключевым словам
    keywords_map = {
        'свет': 'фотон',
        'частица света': 'фотон',
        'квант света': 'фотон',
        'отрицательная частица': 'электрон',
        'химический элемент': 'атом',
        'квант': 'квантовая механика',
        'волна-частица': 'квантовая механика'
    }
    
    for keyword, concept in keywords_map.items():
        if keyword in query_lower:
            return concept, PHYSICS_KNOWLEDGE[concept]
    
    return None, None

def generate_physics_explanation(concept, data, query):
    """Генерирует объяснение физического понятия"""
    explanation_parts = []
    
    # Определение
    explanation_parts.append(f"**{concept.upper()}**")
    explanation_parts.append(f"📖 {data['definition']}")
    
    # Свойства
    if 'properties' in data:
        explanation_parts.append("\n🔬 **Свойства:**")
        for prop in data['properties']:
            explanation_parts.append(f"• {prop}")
    
    # Структура (для атома)
    if 'structure' in data:
        explanation_parts.append("\n🏗️ **Структура:**")
        for struct in data['structure']:
            explanation_parts.append(f"• {struct}")
    
    # Модели (для квантовой механики)
    if 'models' in data:
        explanation_parts.append("\n📐 **Модели:**")
        for model in data['models']:
            explanation_parts.append(f"• {model}")
    
    # Принципы (для квантовой механики)
    if 'principles' in data:
        explanation_parts.append("\n⚛️ **Принципы:**")
        for principle in data['principles']:
            explanation_parts.append(f"• {principle}")
    
    # Применения
    if 'applications' in data:
        explanation_parts.append("\n🚀 **Применения:**")
        for app in data['applications']:
            explanation_parts.append(f"• {app}")
    
    # Дополнительная информация
    if concept == 'фотон':
        explanation_parts.append("\n💡 **Интересные факты:**")
        explanation_parts.append("• Фотоны не стареют - они существуют вечно")
        explanation_parts.append("• Каждый фотон уникален по энергии")
        explanation_parts.append("• Фотоны могут создавать и уничтожаться")
    
    elif concept == 'электрон':
        explanation_parts.append("\n💡 **Интересные факты:**")
        explanation_parts.append("• Электроны движутся вокруг ядра со скоростью ~2,200 км/с")
        explanation_parts.append("• Электроны определяют химические свойства элементов")
        explanation_parts.append("• В металлах электроны могут свободно перемещаться")
    
    return "\n".join(explanation_parts)

@app.route('/api/physics/explain', methods=['GET', 'POST'])
def explain_physics():
    """Объяснение физических понятий"""
    try:
        if request.method == 'GET':
            concept = request.args.get('concept', '')
        else:
            data = request.get_json()
            concept = data.get('concept', '')
        
        logger.info(f"⚛️ Получен физический запрос: {concept[:50]}...")
        
        if not concept:
            return jsonify({
                'success': False,
                'error': 'Физическое понятие не указано'
            }), 400
        
        # Поиск в базе знаний
        found_concept, concept_data = find_physics_concept(concept)
        
        if found_concept:
            explanation = generate_physics_explanation(found_concept, concept_data, concept)
            
            return jsonify({
                'success': True,
                'concept': found_concept,
                'explanation': explanation,
                'module': 'physics',
                'timestamp': datetime.now().isoformat()
            })
        else:
            # Общий ответ для неизвестных понятий
            return jsonify({
                'success': True,
                'concept': concept,
                'explanation': f"**{concept.upper()}**\n\n🔬 Физическое понятие '{concept}' требует дополнительного изучения.\n\n📚 **Рекомендации:**\n• Обратитесь к учебникам по физике\n• Изучите квантовую механику\n• Исследуйте современную физику\n\n💡 **Популярные темы:**\n• Фотоны и свет\n• Электроны и атомы\n• Квантовая механика\n• Теория относительности",
                'module': 'physics',
                'timestamp': datetime.now().isoformat()
            })
        
    except Exception as e:
        logger.error(f"Ошибка обработки физического запроса: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/physics/health', methods=['GET'])
@app.route('/api/health', methods=['GET'])
def health_check():
    """Проверка здоровья сервера"""
    return jsonify({
        "service": "physics",
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "capabilities": [
            "Объяснение физических понятий",
            "Квантовая механика",
            "Элементарные частицы",
            "Атомная физика",
            "Оптика и фотоны"
        ],
        "knowledge_base": {
            "concepts": list(PHYSICS_KNOWLEDGE.keys()),
            "total_concepts": len(PHYSICS_KNOWLEDGE)
        }
    })

@app.route('/api/physics/concepts', methods=['GET'])
def list_concepts():
    """Список доступных физических понятий"""
    return jsonify({
        'success': True,
        'concepts': list(PHYSICS_KNOWLEDGE.keys()),
        'total': len(PHYSICS_KNOWLEDGE),
        'categories': [
            'Элементарные частицы',
            'Атомная физика', 
            'Квантовая механика',
            'Оптика',
            'Электродинамика'
        ]
    })

if __name__ == '__main__':
    print("⚛️ Physics Server запущен")
    print("URL: http://localhost:8110")
    print("Доступные эндпоинты:")
    print("  - GET/POST /api/physics/explain - объяснение физических понятий")
    print("  - GET /api/physics/health - проверка здоровья")
    print("  - GET /api/physics/concepts - список понятий")
    print("  - GET /api/health - общая проверка здоровья")
    print("=" * 60)
    app.run(host='0.0.0.0', port=8110, debug=False)



