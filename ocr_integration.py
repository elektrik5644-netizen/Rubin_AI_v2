#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Интеграция OCR Module с Rubin AI
"""

import json
import requests
from datetime import datetime
from enhanced_ocr_module import EnhancedRubinOCRModule

def integrate_ocr_with_rubin():
    """Интеграция OCR модуля с Rubin AI"""
    print("🔍 ИНТЕГРАЦИЯ OCR MODULE С RUBIN AI")
    print("=" * 50)
    
    # Инициализация OCR модуля
    ocr = EnhancedRubinOCRModule()
    
    # Информация о модуле
    info = ocr.get_module_info()
    print(f"📊 OCR Module: {info['name']} v{info['version']}")
    print(f"🔧 Tesseract доступен: {'✅' if info['tesseract_available'] else '❌'}")
    print(f"⚡ Методы: {', '.join(info['methods'])}")
    
    # Тестирование интеграции
    print("\n🧪 ТЕСТИРОВАНИЕ ИНТЕГРАЦИИ:")
    print("-" * 30)
    
    test_cases = [
        {
            'name': 'Анализ графика функции',
            'image': 'test_graph.png',
            'description': 'Анализ графика квадратичной функции'
        },
        {
            'name': 'Анализ диаграммы',
            'image': 'test_chart.jpg',
            'description': 'Анализ столбчатой диаграммы продаж'
        },
        {
            'name': 'Анализ формул',
            'image': 'test_formula.png',
            'description': 'Извлечение физических формул'
        },
        {
            'name': 'Анализ схемы',
            'image': 'test_circuit.bmp',
            'description': 'Анализ электрической схемы'
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}️⃣ {test_case['name']}: {test_case['image']}")
        
        # Анализ изображения
        if 'график' in test_case['name'].lower():
            result = ocr.analyze_graph(test_case['image'])
        elif 'схема' in test_case['name'].lower():
            result = ocr.analyze_circuit_diagram(test_case['image'])
        else:
            result = ocr.extract_text_from_image(test_case['image'])
        
        if result['success']:
            print(f"✅ Анализ успешен (уверенность: {result['confidence']:.2f})")
            print(f"🔧 Метод: {result.get('method', 'unknown')}")
            
            # Отправка результата в Rubin AI
            try:
                response = send_to_rubin_ai(test_case, result)
                if response:
                    print(f"📡 Отправлено в Rubin AI: {response['status']}")
                else:
                    print("⚠️ Не удалось отправить в Rubin AI")
            except Exception as e:
                print(f"❌ Ошибка отправки в Rubin AI: {e}")
            
            results.append({
                'test': test_case['name'],
                'success': True,
                'confidence': result['confidence'],
                'method': result.get('method', 'unknown')
            })
        else:
            print(f"❌ Ошибка анализа: {result['error']}")
            results.append({
                'test': test_case['name'],
                'success': False,
                'error': result['error']
            })
    
    # Статистика интеграции
    print("\n📈 СТАТИСТИКА ИНТЕГРАЦИИ:")
    print("-" * 30)
    
    success_count = sum(1 for r in results if r['success'])
    print(f"✅ Успешных анализов: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
    
    if success_count > 0:
        avg_confidence = sum(r.get('confidence', 0) for r in results if r['success']) / success_count
        print(f"🎯 Средняя уверенность: {avg_confidence:.2f}")
    
    # Сохранение результатов
    integration_results = {
        'timestamp': datetime.now().isoformat(),
        'ocr_module_info': info,
        'test_results': results,
        'statistics': {
            'success_rate': success_count/len(results)*100,
            'avg_confidence': avg_confidence if success_count > 0 else 0,
            'total_tests': len(results)
        }
    }
    
    with open('ocr_integration_results.json', 'w', encoding='utf-8') as f:
        json.dump(integration_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Результаты интеграции сохранены в: ocr_integration_results.json")
    
    return integration_results

def send_to_rubin_ai(test_case: dict, ocr_result: dict) -> dict:
    """Отправка результата OCR анализа в Rubin AI"""
    try:
        # Формируем сообщение для Rubin AI
        message = f"""
Анализ изображения: {test_case['name']}
Описание: {test_case['description']}
Файл: {test_case['image']}

Результат OCR анализа:
- Уверенность: {ocr_result['confidence']:.2f}
- Метод: {ocr_result.get('method', 'unknown')}
- Извлеченный текст: {ocr_result['text'][:200]}...

Тип изображения: {ocr_result.get('image_type', 'unknown')}
"""
        
        # Отправляем в Smart Dispatcher
        response = requests.post(
            'http://localhost:8080/api/chat',
            json={'message': message},
            timeout=10
        )
        
        if response.status_code == 200:
            return {
                'status': 'success',
                'response': response.json()
            }
        else:
            return {
                'status': 'error',
                'code': response.status_code
            }
            
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }

def demonstrate_ocr_capabilities():
    """Демонстрация возможностей OCR модуля"""
    print("\n🎯 ДЕМОНСТРАЦИЯ ВОЗМОЖНОСТЕЙ OCR:")
    print("-" * 35)
    
    ocr = EnhancedRubinOCRModule()
    
    # Демонстрация анализа графика
    print("\n📊 Анализ графика функции:")
    graph_result = ocr.analyze_graph('test_graph.png')
    if graph_result['success']:
        analysis = graph_result['analysis']
        print(f"✅ Тип: {analysis['graph_type']}")
        print(f"📐 Функция: {analysis['function']}")
        print(f"📍 Точки: {len(analysis['data_points'])}")
        print(f"🏷️ Заголовок: {analysis['title']}")
    
    # Демонстрация анализа схемы
    print("\n⚡ Анализ электрической схемы:")
    circuit_result = ocr.analyze_circuit_diagram('test_circuit.bmp')
    if circuit_result['success']:
        analysis = circuit_result['analysis']
        print(f"✅ Тип схемы: {analysis['circuit_type']}")
        print(f"🔧 Компоненты: {len(analysis['components'])}")
        print(f"📊 Значения: {analysis['values']}")
    
    # Демонстрация извлечения формул
    print("\n📐 Извлечение математических формул:")
    formula_result = ocr.extract_text_from_image('test_formula.png')
    if formula_result['success']:
        math_content = ocr._extract_mathematical_content(formula_result['text'])
        print(f"✅ Формулы: {len(math_content['formulas'])}")
        print(f"📐 Уравнения: {len(math_content['equations'])}")
        print(f"🔢 Числа: {len(math_content['numbers'])}")
        print(f"📝 Переменные: {math_content['variables']}")

def main():
    """Основная функция"""
    print("🔍 ИНТЕГРАЦИЯ OCR MODULE С RUBIN AI")
    print("=" * 50)
    
    # Интеграция с Rubin AI
    integration_results = integrate_ocr_with_rubin()
    
    # Демонстрация возможностей
    demonstrate_ocr_capabilities()
    
    print("\n🎉 ИНТЕГРАЦИЯ OCR MODULE ЗАВЕРШЕНА!")
    print("=" * 40)
    
    # Итоговая оценка
    success_rate = integration_results['statistics']['success_rate']
    avg_confidence = integration_results['statistics']['avg_confidence']
    
    print(f"📊 Общий результат: {success_rate:.1f}%")
    print(f"🎯 Средняя уверенность: {avg_confidence:.2f}")
    
    if success_rate >= 80 and avg_confidence >= 0.6:
        print("🎉 OCR Module успешно интегрирован с Rubin AI!")
    elif success_rate >= 60:
        print("⚠️ OCR Module частично интегрирован")
    else:
        print("❌ OCR Module требует доработки")

if __name__ == "__main__":
    main()





