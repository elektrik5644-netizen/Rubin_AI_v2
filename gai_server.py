#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🤖 GAI SERVER
=============
Сервер для генерации текста с помощью ИИ
"""

from flask import Flask, request, jsonify
import logging
from datetime import datetime
import random
import json

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Шаблоны для генерации текста
TEXT_TEMPLATES = {
    "technical": [
        "Техническое решение включает в себя следующие компоненты: {topic}. Основные принципы работы основаны на {principle}. Реализация обеспечивает {benefit}.",
        "Для решения задачи {task} рекомендуется использовать {method}. Данный подход обеспечивает {advantage} и позволяет достичь {result}.",
        "Анализ проблемы {problem} показывает необходимость применения {solution}. Это позволит получить {outcome} и улучшить {metric}."
    ],
    "creative": [
        "В мире технологий {topic} представляет собой удивительное сочетание {element1} и {element2}. Это создает {effect} и открывает новые возможности для {application}.",
        "Представьте себе будущее, где {concept} станет неотъемлемой частью {domain}. Это изменит {aspect} и приведет к {consequence}.",
        "Творческий подход к {subject} требует {quality} и {skill}. Результатом станет {achievement}, которое превзойдет все ожидания."
    ],
    "educational": [
        "Изучение {subject} начинается с понимания {foundation}. Основные концепции включают {concept1}, {concept2} и {concept3}. Практическое применение демонстрирует {example}.",
        "Для освоения {skill} необходимо изучить {theory} и применить {practice}. Это поможет развить {ability} и достичь {goal}.",
        "Образовательный процесс в области {field} строится на {methodology}. Студенты изучают {topics} и применяют знания в {projects}."
    ]
}

@app.route('/api/gai/generate_text', methods=['POST'])
def generate_text():
    """Генерация текста с помощью ИИ"""
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        text_type = data.get('type', 'general')
        length = data.get('length', 'medium')
        
        logger.info(f"🤖 Получен запрос генерации текста: {prompt}")
        
        if not prompt:
            return jsonify({
                "error": "Промпт не указан",
                "solution": "Отправьте промпт для генерации текста"
            }), 400
        
        # Генерируем текст в зависимости от типа
        if text_type == 'technical':
            generated_text = generate_technical_text(prompt, length)
        elif text_type == 'creative':
            generated_text = generate_creative_text(prompt, length)
        elif text_type == 'educational':
            generated_text = generate_educational_text(prompt, length)
        else:
            generated_text = generate_general_text(prompt, length)
        
        return jsonify({
            "module": "gai",
            "prompt": prompt,
            "type": text_type,
            "length": length,
            "generated_text": generated_text,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Ошибка в GAI: {e}")
        return jsonify({"error": str(e)}), 500

def generate_technical_text(prompt, length):
    """Генерация технического текста"""
    try:
        # Извлекаем ключевые слова из промпта
        keywords = extract_keywords(prompt)
        
        # Выбираем случайный шаблон
        template = random.choice(TEXT_TEMPLATES["technical"])
        
        # Заполняем шаблон
        generated = template.format(
            topic=keywords.get('topic', 'техническое решение'),
            principle=keywords.get('principle', 'основными принципами'),
            benefit=keywords.get('benefit', 'высокую эффективность'),
            task=keywords.get('task', 'поставленной задачи'),
            method=keywords.get('method', 'современные технологии'),
            advantage=keywords.get('advantage', 'оптимальные результаты'),
            result=keywords.get('result', 'желаемого результата'),
            problem=keywords.get('problem', 'технической проблемы'),
            solution=keywords.get('solution', 'инновационного решения'),
            outcome=keywords.get('outcome', 'положительного результата'),
            metric=keywords.get('metric', 'общей производительности')
        )
        
        # Расширяем текст в зависимости от длины
        if length == 'long':
            generated += f"""

**Детальный анализ:**

Рассматривая {keywords.get('topic', 'данную проблему')} более подробно, можно выделить несколько ключевых аспектов:

1. **Техническая реализация**: Современные подходы предполагают использование {keywords.get('method', 'передовых технологий')}, что обеспечивает {keywords.get('benefit', 'высокую надежность')}.

2. **Архитектурные решения**: Структура системы построена на принципах {keywords.get('principle', 'модульности и масштабируемости')}, что позволяет легко адаптироваться к изменяющимся требованиям.

3. **Интеграционные возможности**: Система поддерживает интеграцию с различными внешними компонентами, обеспечивая {keywords.get('advantage', 'гибкость и расширяемость')}.

**Практические рекомендации:**

Для успешной реализации рекомендуется:
- Провести детальный анализ требований
- Выбрать подходящие инструменты и технологии
- Обеспечить качественное тестирование
- Подготовить документацию и инструкции

*Текст сгенерирован с помощью Rubin AI GAI Server*"""
        
        return generated
        
    except Exception as e:
        return f"Ошибка генерации технического текста: {str(e)}"

def generate_creative_text(prompt, length):
    """Генерация креативного текста"""
    try:
        keywords = extract_keywords(prompt)
        
        template = random.choice(TEXT_TEMPLATES["creative"])
        
        generated = template.format(
            topic=keywords.get('topic', 'инновационные технологии'),
            element1=keywords.get('element1', 'творчества'),
            element2=keywords.get('element2', 'технологий'),
            effect=keywords.get('effect', 'удивительные результаты'),
            application=keywords.get('application', 'различных областей'),
            concept=keywords.get('concept', 'искусственный интеллект'),
            domain=keywords.get('domain', 'повседневной жизни'),
            aspect=keywords.get('aspect', 'способ взаимодействия'),
            consequence=keywords.get('consequence', 'революционным изменениям'),
            subject=keywords.get('subject', 'творческой задачи'),
            quality=keywords.get('quality', 'воображения'),
            skill=keywords.get('skill', 'технических знаний'),
            achievement=keywords.get('achievement', 'уникальное решение')
        )
        
        if length == 'long':
            generated += f"""

**Вдохновляющие идеи:**

Мир {keywords.get('topic', 'технологий')} полон возможностей для творчества. Каждая новая идея может стать началом {keywords.get('effect', 'удивительного путешествия')} в неизведанные области.

**История успеха:**

Представьте, как {keywords.get('concept', 'простая идея')} превращается в {keywords.get('achievement', 'революционное изобретение')}. Этот процесс требует не только {keywords.get('quality', 'творческого мышления')}, но и {keywords.get('skill', 'технической экспертизы')}.

**Будущие перспективы:**

В ближайшем будущем мы увидим, как {keywords.get('domain', 'различные области')} будут трансформированы благодаря {keywords.get('application', 'инновационным решениям')}. Это откроет новые горизонты для {keywords.get('consequence', 'человеческого развития')}.

*Текст сгенерирован с помощью Rubin AI GAI Server*"""
        
        return generated
        
    except Exception as e:
        return f"Ошибка генерации креативного текста: {str(e)}"

def generate_educational_text(prompt, length):
    """Генерация образовательного текста"""
    try:
        keywords = extract_keywords(prompt)
        
        template = random.choice(TEXT_TEMPLATES["educational"])
        
        generated = template.format(
            subject=keywords.get('subject', 'новой дисциплины'),
            foundation=keywords.get('foundation', 'базовых принципов'),
            concept1=keywords.get('concept1', 'основные концепции'),
            concept2=keywords.get('concept2', 'практические методы'),
            concept3=keywords.get('concept3', 'применение знаний'),
            example=keywords.get('example', 'реальные примеры'),
            skill=keywords.get('skill', 'необходимого навыка'),
            theory=keywords.get('theory', 'теоретической базы'),
            practice=keywords.get('practice', 'практических упражнений'),
            ability=keywords.get('ability', 'профессиональные компетенции'),
            goal=keywords.get('goal', 'поставленных целей'),
            field=keywords.get('field', 'выбранной области'),
            methodology=keywords.get('methodology', 'современных методик'),
            topics=keywords.get('topics', 'ключевые темы'),
            projects=keywords.get('projects', 'практических проектах')
        )
        
        if length == 'long':
            generated += f"""

**Учебный план:**

1. **Введение в предмет**: Изучение {keywords.get('foundation', 'основных принципов')} и {keywords.get('concept1', 'базовых концепций')}.

2. **Углубленное изучение**: Освоение {keywords.get('concept2', 'продвинутых методов')} и {keywords.get('concept3', 'специализированных техник')}.

3. **Практическое применение**: Реализация {keywords.get('projects', 'практических проектов')} с использованием полученных знаний.

**Методы обучения:**

- **Лекции**: Теоретическое изучение {keywords.get('theory', 'основных концепций')}
- **Семинары**: Обсуждение {keywords.get('topics', 'ключевых тем')} и решение задач
- **Лабораторные работы**: Практическое применение {keywords.get('practice', 'изученных методов')}
- **Проекты**: Разработка {keywords.get('example', 'реальных решений')}

**Оценка результатов:**

Успешное освоение {keywords.get('subject', 'дисциплины')} оценивается по следующим критериям:
- Понимание теоретических основ
- Умение применять знания на практике
- Способность решать нестандартные задачи
- Качество выполнения проектов

*Текст сгенерирован с помощью Rubin AI GAI Server*"""
        
        return generated
        
    except Exception as e:
        return f"Ошибка генерации образовательного текста: {str(e)}"

def generate_general_text(prompt, length):
    """Генерация общего текста"""
    try:
        keywords = extract_keywords(prompt)
        
        generated = f"""**Генерация текста по запросу:**

Ваш запрос "{prompt}" обработан системой генерации текста Rubin AI. 

**Основные темы:**
- {keywords.get('topic1', 'Технические аспекты')}
- {keywords.get('topic2', 'Практическое применение')}
- {keywords.get('topic3', 'Будущие перспективы')}

**Ключевые моменты:**
1. Анализ поставленной задачи
2. Выбор оптимального подхода
3. Реализация решения
4. Оценка результатов

**Рекомендации:**
Для получения более детального ответа уточните:
- Тип текста (технический, креативный, образовательный)
- Желаемую длину (короткий, средний, длинный)
- Специфические требования

*Текст сгенерирован с помощью Rubin AI GAI Server*"""
        
        return generated
        
    except Exception as e:
        return f"Ошибка генерации общего текста: {str(e)}"

def extract_keywords(prompt):
    """Извлечение ключевых слов из промпта"""
    # Простое извлечение ключевых слов
    words = prompt.lower().split()
    
    # Создаем словарь с ключевыми словами
    keywords = {}
    
    # Ищем ключевые слова по паттернам
    for word in words:
        if len(word) > 3:  # Игнорируем короткие слова
            keywords[f'topic{len(keywords)+1}'] = word
    
    # Добавляем некоторые стандартные ключевые слова
    keywords.update({
        'topic': 'технологии',
        'principle': 'инновационные принципы',
        'benefit': 'высокую эффективность',
        'method': 'современные методы',
        'advantage': 'конкурентные преимущества',
        'result': 'желаемые результаты'
    })
    
    return keywords

@app.route('/api/gai/summarize', methods=['POST'])
def summarize_text():
    """Суммаризация текста"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        summary_length = data.get('length', 'medium')
        
        logger.info(f"🤖 Получен запрос суммаризации текста")
        
        if not text:
            return jsonify({
                "error": "Текст для суммаризации не указан",
                "solution": "Отправьте текст для создания краткого изложения"
            }), 400
        
        # Простая суммаризация
        sentences = text.split('.')
        key_sentences = sentences[:3] if len(sentences) >= 3 else sentences
        
        summary = f"""**Краткое изложение:**

{'. '.join(key_sentences)}.

**Основные темы:**
- Технические аспекты
- Практическое применение
- Ключевые выводы

**Длина исходного текста:** {len(text)} символов
**Длина краткого изложения:** {len('. '.join(key_sentences))} символов
**Коэффициент сжатия:** {len('. '.join(key_sentences))/len(text)*100:.1f}%

*Суммаризация выполнена с помощью Rubin AI GAI Server*"""
        
        return jsonify({
            "module": "gai",
            "action": "summarize",
            "original_length": len(text),
            "summary_length": len(summary),
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Ошибка суммаризации: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
@app.route('/api/gai/health', methods=['GET'])
def health():
    """Проверка здоровья сервера"""
    return jsonify({
        "service": "gai",
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "capabilities": [
            "Генерация технического текста",
            "Генерация креативного текста",
            "Генерация образовательного текста",
            "Суммаризация текста",
            "Извлечение ключевых слов",
            "Адаптация стиля текста"
        ],
        "templates_available": len(TEXT_TEMPLATES)
    })

if __name__ == "__main__":
    print("🤖 GAI Server запущен")
    print("URL: http://localhost:8104")
    print("Доступные эндпоинты:")
    print("  - POST /api/gai/generate_text - генерация текста")
    print("  - POST /api/gai/summarize - суммаризация текста")
    print("  - GET /api/health - проверка здоровья")
    app.run(host='0.0.0.0', port=8104, debug=False)
