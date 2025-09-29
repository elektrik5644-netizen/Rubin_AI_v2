#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Анализ точности ответов Rubin AI
"""

import json
import sqlite3
from datetime import datetime

def analyze_test_results():
    """Анализ результатов тестирования"""
    print("🎯 Анализ точности ответов Rubin AI")
    print("=" * 50)
    
    # Загружаем результаты тестов
    try:
        with open('rubin_quick_test_results.json', 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        total_questions = test_data['total_questions']
        results = test_data['results']
        
        print(f"📊 Результаты тестирования:")
        print(f"Всего вопросов: {total_questions}")
        
        # Анализируем успешность
        successful_answers = 0
        failed_answers = 0
        response_times = []
        
        for result in results:
            if result['expected_result']['status'] == 'success':
                successful_answers += 1
                response_times.append(result['expected_result']['response_time'])
            else:
                failed_answers += 1
        
        success_rate = (successful_answers / total_questions) * 100
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        print(f"✅ Успешных ответов: {successful_answers}")
        print(f"❌ Неудачных ответов: {failed_answers}")
        print(f"📈 Процент успешности: {success_rate:.1f}%")
        print(f"⏱️ Среднее время ответа: {avg_response_time:.2f}с")
        
        return success_rate, avg_response_time
        
    except FileNotFoundError:
        print("❌ Файл результатов тестирования не найден")
        return 0, 0

def analyze_learning_database():
    """Анализ базы данных обучения"""
    print(f"\n📚 Анализ базы данных обучения:")
    print("-" * 30)
    
    try:
        conn = sqlite3.connect('rubin_learning.db')
        cursor = conn.cursor()
        
        # Общая статистика
        cursor.execute('SELECT COUNT(*) FROM interactions')
        total_interactions = cursor.fetchone()[0]
        
        # Анализ по оценкам успешности
        cursor.execute('SELECT AVG(success_score) FROM interactions WHERE success_score IS NOT NULL')
        avg_success = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM interactions WHERE success_score > 0.8')
        high_success = cursor.fetchone()[0]
        
        # Анализ по уверенности
        cursor.execute('SELECT AVG(confidence) FROM interactions WHERE confidence IS NOT NULL')
        avg_confidence = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM interactions WHERE confidence > 0.8')
        high_confidence = cursor.fetchone()[0]
        
        # Анализ по времени ответа
        cursor.execute('SELECT AVG(response_time) FROM interactions WHERE response_time IS NOT NULL')
        avg_response_time = cursor.fetchone()[0]
        
        print(f"Всего взаимодействий: {total_interactions}")
        print(f"Средняя оценка успешности: {avg_success:.3f}" if avg_success else "Нет данных")
        print(f"Высокие оценки (>0.8): {high_success} ({(high_success/total_interactions*100):.1f}%)" if total_interactions > 0 else "Нет данных")
        print(f"Средняя уверенность: {avg_confidence:.3f}" if avg_confidence else "Нет данных")
        print(f"Высокая уверенность (>0.8): {high_confidence} ({(high_confidence/total_interactions*100):.1f}%)" if total_interactions > 0 else "Нет данных")
        print(f"Среднее время ответа: {avg_response_time:.3f}с" if avg_response_time else "Нет данных")
        
        conn.close()
        
        return avg_success, high_success, total_interactions
        
    except sqlite3.Error as e:
        print(f"❌ Ошибка базы данных: {e}")
        return 0, 0, 0

def analyze_api_accuracy():
    """Анализ точности по API"""
    print(f"\n🔧 Анализ точности по API:")
    print("-" * 25)
    
    try:
        with open('rubin_quick_test_results.json', 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        results = test_data['results']
        api_stats = {}
        
        for result in results:
            api = result['expected_result']['api']
            status = result['expected_result']['status']
            
            if api not in api_stats:
                api_stats[api] = {'total': 0, 'success': 0, 'times': []}
            
            api_stats[api]['total'] += 1
            if status == 'success':
                api_stats[api]['success'] += 1
                api_stats[api]['times'].append(result['expected_result']['response_time'])
        
        for api, stats in api_stats.items():
            success_rate = (stats['success'] / stats['total']) * 100
            avg_time = sum(stats['times']) / len(stats['times']) if stats['times'] else 0
            print(f"{api}: {stats['success']}/{stats['total']} ({success_rate:.1f}%) - {avg_time:.2f}с")
        
    except FileNotFoundError:
        print("❌ Файл результатов тестирования не найден")

def analyze_response_quality():
    """Анализ качества ответов"""
    print(f"\n📝 Анализ качества ответов:")
    print("-" * 25)
    
    try:
        with open('rubin_quick_test_results.json', 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        results = test_data['results']
        
        # Анализируем соответствие ожидаемому API
        correct_routing = 0
        total_questions = len(results)
        
        for result in results:
            expected_api = result['expected_api']
            actual_api = result['expected_result']['api']
            
            if expected_api == actual_api:
                correct_routing += 1
        
        routing_accuracy = (correct_routing / total_questions) * 100
        
        print(f"Правильная маршрутизация: {correct_routing}/{total_questions} ({routing_accuracy:.1f}%)")
        
        # Анализируем специализацию ответов
        specialized_responses = 0
        for result in results:
            response = result['expected_result']['response']
            # Проверяем, содержит ли ответ специализированную информацию
            if any(keyword in response.lower() for keyword in ['pmac', 'plc', 'закон ома', 'антенна', 'передатчик']):
                specialized_responses += 1
        
        specialization_rate = (specialized_responses / total_questions) * 100
        print(f"Специализированные ответы: {specialized_responses}/{total_questions} ({specialization_rate:.1f}%)")
        
        return routing_accuracy, specialization_rate
        
    except FileNotFoundError:
        print("❌ Файл результатов тестирования не найден")
        return 0, 0

def main():
    """Основная функция анализа"""
    print(f"🧠 Анализ точности Rubin AI")
    print(f"📅 Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Анализируем результаты тестирования
    test_success_rate, test_avg_time = analyze_test_results()
    
    # Анализируем базу данных обучения
    learning_success, high_success, total_interactions = analyze_learning_database()
    
    # Анализируем точность по API
    analyze_api_accuracy()
    
    # Анализируем качество ответов
    routing_accuracy, specialization_rate = analyze_response_quality()
    
    # Итоговый отчет
    print(f"\n{'='*60}")
    print("📊 ИТОГОВЫЙ ОТЧЕТ О ТОЧНОСТИ")
    print("=" * 60)
    
    print(f"🎯 Тестирование (10 вопросов):")
    print(f"  • Успешность: {test_success_rate:.1f}%")
    print(f"  • Время ответа: {test_avg_time:.2f}с")
    
    print(f"\n📚 База обучения ({total_interactions} взаимодействий):")
    print(f"  • Средняя оценка: {learning_success:.3f}" if learning_success else "  • Нет данных")
    print(f"  • Высокие оценки: {high_success} ({(high_success/total_interactions*100):.1f}%)" if total_interactions > 0 else "  • Нет данных")
    
    print(f"\n🔧 Качество системы:")
    print(f"  • Маршрутизация: {routing_accuracy:.1f}%")
    print(f"  • Специализация: {specialization_rate:.1f}%")
    
    # Общая оценка точности
    if test_success_rate > 0:
        overall_accuracy = (test_success_rate + routing_accuracy + specialization_rate) / 3
        print(f"\n🏆 ОБЩАЯ ТОЧНОСТЬ: {overall_accuracy:.1f}%")
        
        if overall_accuracy >= 90:
            print("🟢 ОТЛИЧНО - система работает на высоком уровне")
        elif overall_accuracy >= 80:
            print("🟡 ХОРОШО - система работает стабильно")
        elif overall_accuracy >= 70:
            print("🟠 УДОВЛЕТВОРИТЕЛЬНО - есть место для улучшений")
        else:
            print("🔴 ТРЕБУЕТ УЛУЧШЕНИЙ - необходима доработка")

if __name__ == '__main__':
    main()























