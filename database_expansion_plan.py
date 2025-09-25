#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
План расширения базы данных Rubin AI до 200+ документов
"""

import sqlite3
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any
import os

class DatabaseExpansionPlanner:
    """Планировщик расширения базы данных"""
    
    def __init__(self, db_path: str = "rubin_ai_v2.db"):
        self.db_path = db_path
        self.connection = None
        self.connect_database()
    
    def connect_database(self):
        """Подключение к базе данных"""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.execute("PRAGMA foreign_keys = ON")
            print("✅ Подключение к базе данных установлено")
        except Exception as e:
            print(f"❌ Ошибка подключения к БД: {e}")
            raise
    
    def analyze_current_state(self) -> Dict[str, Any]:
        """Анализ текущего состояния базы данных"""
        try:
            cursor = self.connection.cursor()
            
            # Общая статистика
            cursor.execute("SELECT COUNT(*) FROM documents")
            total_docs = cursor.fetchone()[0]
            
            # Статистика по категориям
            cursor.execute("""
                SELECT category, COUNT(*) as count
                FROM documents 
                GROUP BY category
                ORDER BY count DESC
            """)
            categories = cursor.fetchall()
            
            # Статистика по уровням сложности
            cursor.execute("""
                SELECT difficulty_level, COUNT(*) as count
                FROM documents 
                WHERE difficulty_level IS NOT NULL
                GROUP BY difficulty_level
                ORDER BY count DESC
            """)
            difficulties = cursor.fetchall()
            
            # Статистика синонимов
            cursor.execute("SELECT COUNT(*) FROM technical_synonyms")
            total_synonyms = cursor.fetchone()[0]
            
            # Размер базы данных
            db_size = os.path.getsize(self.db_path)
            
            analysis = {
                'total_documents': total_docs,
                'total_synonyms': total_synonyms,
                'database_size_mb': round(db_size / (1024 * 1024), 2),
                'categories': [{'name': cat[0], 'count': cat[1]} for cat in categories],
                'difficulty_levels': [{'level': diff[0], 'count': diff[1]} for diff in difficulties],
                'analysis_date': datetime.now().isoformat()
            }
            
            return analysis
            
        except Exception as e:
            print(f"❌ Ошибка анализа: {e}")
            return {}
    
    def create_expansion_plan(self, target_documents: int = 200) -> Dict[str, Any]:
        """Создание плана расширения базы данных"""
        try:
            current_state = self.analyze_current_state()
            current_docs = current_state.get('total_documents', 0)
            
            documents_needed = target_documents - current_docs
            
            if documents_needed <= 0:
                print(f"✅ Цель уже достигнута! В базе {current_docs} документов")
                return {'status': 'goal_achieved', 'current_documents': current_docs}
            
            # План расширения по категориям
            expansion_plan = {
                'target_documents': target_documents,
                'current_documents': current_docs,
                'documents_needed': documents_needed,
                'expansion_categories': [
                    {
                        'category': 'автоматизация',
                        'current_count': self._get_category_count('автоматизация'),
                        'target_count': 50,
                        'documents_to_add': 0,
                        'priority': 'высокий',
                        'topics': [
                            'ПИД-регуляторы (дополнительно)',
                            'SCADA системы',
                            'ПЛК программирование',
                            'Промышленные сети',
                            'Системы безопасности',
                            'Аварийные системы',
                            'Резервирование',
                            'Диагностика оборудования'
                        ]
                    },
                    {
                        'category': 'электротехника',
                        'current_count': self._get_category_count('электротехника'),
                        'target_count': 40,
                        'documents_to_add': 0,
                        'priority': 'высокий',
                        'topics': [
                            'Электрические машины',
                            'Силовые преобразователи',
                            'Электроприводы',
                            'Релейная защита',
                            'Электроизмерения',
                            'Электробезопасность',
                            'Кабельные системы',
                            'Заземление и молниезащита'
                        ]
                    },
                    {
                        'category': 'программирование',
                        'current_count': self._get_category_count('программирование'),
                        'target_count': 35,
                        'documents_to_add': 0,
                        'priority': 'средний',
                        'topics': [
                            'Python для автоматизации',
                            'C++ для встраиваемых систем',
                            'JavaScript для веб-интерфейсов',
                            'SQL для баз данных',
                            'Алгоритмы и структуры данных',
                            'Тестирование кода',
                            'Версионирование',
                            'Документирование кода'
                        ]
                    },
                    {
                        'category': 'радиотехника',
                        'current_count': self._get_category_count('радиотехника'),
                        'target_count': 25,
                        'documents_to_add': 0,
                        'priority': 'средний',
                        'topics': [
                            'Антенные системы',
                            'Радиопередатчики',
                            'Радиоприемники',
                            'Модуляция сигналов',
                            'Цифровая обработка сигналов',
                            'Радиолокация',
                            'Спутниковая связь',
                            'Беспроводные сети'
                        ]
                    },
                    {
                        'category': 'механика',
                        'current_count': self._get_category_count('механика'),
                        'target_count': 20,
                        'documents_to_add': 0,
                        'priority': 'средний',
                        'topics': [
                            'Детали машин',
                            'Механические передачи',
                            'Подшипники',
                            'Смазка и смазочные материалы',
                            'Вибрация и балансировка',
                            'Механические испытания',
                            'Материаловедение',
                            'Прочность и надежность'
                        ]
                    },
                    {
                        'category': 'информационные_технологии',
                        'current_count': self._get_category_count('информационные_технологии'),
                        'target_count': 20,
                        'documents_to_add': 0,
                        'priority': 'средний',
                        'topics': [
                            'Базы данных',
                            'Сетевые технологии',
                            'Кибербезопасность',
                            'Облачные вычисления',
                            'Большие данные',
                            'Искусственный интеллект',
                            'Машинное обучение',
                            'Интернет вещей'
                        ]
                    },
                    {
                        'category': 'безопасность',
                        'current_count': self._get_category_count('безопасность'),
                        'target_count': 10,
                        'documents_to_add': 0,
                        'priority': 'высокий',
                        'topics': [
                            'Промышленная безопасность',
                            'Электробезопасность',
                            'Пожарная безопасность',
                            'Охрана труда',
                            'Экологическая безопасность',
                            'Информационная безопасность',
                            'Аварийные ситуации',
                            'Эвакуация и спасение'
                        ]
                    }
                ],
                'timeline': {
                    'phase_1': {
                        'duration': '1 месяц',
                        'documents': 50,
                        'focus': 'автоматизация, электротехника'
                    },
                    'phase_2': {
                        'duration': '1 месяц', 
                        'documents': 50,
                        'focus': 'программирование, радиотехника'
                    },
                    'phase_3': {
                        'duration': '1 месяц',
                        'documents': 50,
                        'focus': 'механика, ИТ, безопасность'
                    }
                },
                'resources_needed': {
                    'expertise': [
                        'Инженеры по автоматизации',
                        'Электротехники',
                        'Программисты',
                        'Радиоинженеры',
                        'Механики',
                        'IT-специалисты',
                        'Специалисты по безопасности'
                    ],
                    'tools': [
                        'Системы документооборота',
                        'Редакторы технической документации',
                        'Системы контроля версий',
                        'Инструменты для создания диаграмм',
                        'Платформы для совместной работы'
                    ],
                    'sources': [
                        'Техническая литература',
                        'Стандарты и нормативы',
                        'Опыт эксплуатации',
                        'Результаты испытаний',
                        'Методические материалы'
                    ]
                },
                'quality_metrics': {
                    'accuracy': '>95%',
                    'completeness': '>90%',
                    'relevance': '>90%',
                    'readability': '>85%',
                    'technical_depth': 'соответствует уровню сложности'
                },
                'created_date': datetime.now().isoformat()
            }
            
            # Рассчитываем количество документов для каждой категории
            total_target = sum(cat['target_count'] for cat in expansion_plan['expansion_categories'])
            for category in expansion_plan['expansion_categories']:
                category['documents_to_add'] = max(0, category['target_count'] - category['current_count'])
            
            return expansion_plan
            
        except Exception as e:
            print(f"❌ Ошибка создания плана: {e}")
            return {}
    
    def _get_category_count(self, category: str) -> int:
        """Получение количества документов в категории"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT COUNT(*) FROM documents WHERE category = ?", (category,))
            return cursor.fetchone()[0]
        except:
            return 0
    
    def create_document_templates(self) -> List[Dict[str, Any]]:
        """Создание шаблонов документов для расширения"""
        templates = [
            {
                'template_name': 'Техническое руководство',
                'structure': {
                    'title': 'Название документа',
                    'introduction': 'Введение и область применения',
                    'theory': 'Теоретические основы',
                    'practical': 'Практические аспекты',
                    'examples': 'Примеры и кейсы',
                    'troubleshooting': 'Диагностика и устранение неисправностей',
                    'references': 'Ссылки и дополнительная литература'
                },
                'target_categories': ['автоматизация', 'электротехника', 'механика']
            },
            {
                'template_name': 'Программное руководство',
                'structure': {
                    'title': 'Название библиотеки/модуля',
                    'installation': 'Установка и настройка',
                    'api_reference': 'Справочник API',
                    'examples': 'Примеры использования',
                    'best_practices': 'Рекомендации по использованию',
                    'troubleshooting': 'Решение проблем',
                    'changelog': 'История изменений'
                },
                'target_categories': ['программирование', 'информационные_технологии']
            },
            {
                'template_name': 'Безопасность и стандарты',
                'structure': {
                    'title': 'Название стандарта/правила',
                    'scope': 'Область применения',
                    'requirements': 'Требования и нормы',
                    'implementation': 'Реализация требований',
                    'compliance': 'Соответствие стандартам',
                    'audit': 'Проверка и аудит',
                    'penalties': 'Ответственность и штрафы'
                },
                'target_categories': ['безопасность', 'автоматизация', 'электротехника']
            },
            {
                'template_name': 'Научно-техническая статья',
                'structure': {
                    'title': 'Название исследования',
                    'abstract': 'Аннотация',
                    'introduction': 'Введение и постановка задачи',
                    'methodology': 'Методология исследования',
                    'results': 'Результаты и анализ',
                    'discussion': 'Обсуждение результатов',
                    'conclusions': 'Выводы и рекомендации',
                    'references': 'Список литературы'
                },
                'target_categories': ['радиотехника', 'механика', 'информационные_технологии']
            }
        ]
        
        return templates
    
    def generate_implementation_schedule(self) -> Dict[str, Any]:
        """Генерация графика реализации"""
        schedule = {
            'week_1_2': {
                'tasks': [
                    'Анализ текущего состояния базы данных',
                    'Создание детального плана расширения',
                    'Подготовка шаблонов документов',
                    'Настройка системы контроля качества'
                ],
                'deliverables': [
                    'План расширения базы данных',
                    'Шаблоны документов',
                    'Критерии качества',
                    'Система мониторинга прогресса'
                ]
            },
            'week_3_6': {
                'tasks': [
                    'Создание документов по автоматизации (25 документов)',
                    'Создание документов по электротехнике (20 документов)',
                    'Обновление синонимов и категорий',
                    'Тестирование качества поиска'
                ],
                'deliverables': [
                    '45 новых документов',
                    'Обновленная база синонимов',
                    'Отчет о качестве поиска',
                    'Метрики производительности'
                ]
            },
            'week_7_10': {
                'tasks': [
                    'Создание документов по программированию (20 документов)',
                    'Создание документов по радиотехнике (15 документов)',
                    'Интеграция с существующими системами',
                    'Оптимизация производительности'
                ],
                'deliverables': [
                    '35 новых документов',
                    'Интегрированная система поиска',
                    'Оптимизированная база данных',
                    'Обновленная документация'
                ]
            },
            'week_11_14': {
                'tasks': [
                    'Создание документов по механике (15 документов)',
                    'Создание документов по ИТ (15 документов)',
                    'Создание документов по безопасности (10 документов)',
                    'Финальное тестирование и валидация'
                ],
                'deliverables': [
                    '40 новых документов',
                    'Полностью протестированная система',
                    'Финальный отчет о качестве',
                    'Рекомендации по дальнейшему развитию'
                ]
            },
            'milestones': [
                {'date': 'Неделя 2', 'milestone': 'План и шаблоны готовы'},
                {'date': 'Неделя 6', 'milestone': '50% документов добавлено'},
                {'date': 'Неделя 10', 'milestone': '75% документов добавлено'},
                {'date': 'Неделя 14', 'milestone': 'Цель 200+ документов достигнута'}
            ]
        }
        
        return schedule
    
    def save_plan_to_file(self, plan: Dict[str, Any], filename: str = "database_expansion_plan.json"):
        """Сохранение плана в файл"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(plan, f, ensure_ascii=False, indent=2)
            print(f"✅ План сохранен в файл: {filename}")
        except Exception as e:
            print(f"❌ Ошибка сохранения плана: {e}")
    
    def close_connection(self):
        """Закрытие соединения с базой данных"""
        if self.connection:
            self.connection.close()
            print("✅ Соединение с БД закрыто")

def main():
    """Основная функция"""
    print("📈 ПЛАН РАСШИРЕНИЯ БАЗЫ ДАННЫХ ДО 200+ ДОКУМЕНТОВ")
    print("=" * 60)
    
    planner = DatabaseExpansionPlanner()
    
    try:
        # Анализируем текущее состояние
        print("\n📊 АНАЛИЗ ТЕКУЩЕГО СОСТОЯНИЯ:")
        current_state = planner.analyze_current_state()
        print(f"  - Всего документов: {current_state.get('total_documents', 0)}")
        print(f"  - Всего синонимов: {current_state.get('total_synonyms', 0)}")
        print(f"  - Размер БД: {current_state.get('database_size_mb', 0)} МБ")
        
        print("\n📋 По категориям:")
        for category in current_state.get('categories', []):
            print(f"    • {category['name']}: {category['count']} документов")
        
        # Создаем план расширения
        print("\n🎯 СОЗДАНИЕ ПЛАНА РАСШИРЕНИЯ:")
        expansion_plan = planner.create_expansion_plan(target_documents=200)
        
        if expansion_plan.get('status') == 'goal_achieved':
            print("✅ Цель уже достигнута!")
            return
        
        print(f"  - Целевое количество: {expansion_plan['target_documents']} документов")
        print(f"  - Текущее количество: {expansion_plan['current_documents']} документов")
        print(f"  - Необходимо добавить: {expansion_plan['documents_needed']} документов")
        
        print("\n📚 ПЛАН ПО КАТЕГОРИЯМ:")
        for category in expansion_plan['expansion_categories']:
            if category['documents_to_add'] > 0:
                print(f"  • {category['category']}: +{category['documents_to_add']} документов (приоритет: {category['priority']})")
        
        # Создаем шаблоны документов
        print("\n📝 ШАБЛОНЫ ДОКУМЕНТОВ:")
        templates = planner.create_document_templates()
        for template in templates:
            print(f"  • {template['template_name']} - для категорий: {', '.join(template['target_categories'])}")
        
        # Создаем график реализации
        print("\n📅 ГРАФИК РЕАЛИЗАЦИИ:")
        schedule = planner.generate_implementation_schedule()
        for period, details in schedule.items():
            if period != 'milestones':
                print(f"  {period.replace('_', ' ').title()}:")
                for task in details['tasks']:
                    print(f"    - {task}")
        
        print("\n🎯 ВЕХИ ПРОЕКТА:")
        for milestone in schedule['milestones']:
            print(f"  • {milestone['date']}: {milestone['milestone']}")
        
        # Сохраняем план
        planner.save_plan_to_file(expansion_plan)
        
        print(f"\n💡 РЕКОМЕНДАЦИИ:")
        print("  1. Начните с категорий высокого приоритета")
        print("  2. Используйте созданные шаблоны документов")
        print("  3. Регулярно тестируйте качество поиска")
        print("  4. Обновляйте синонимы при добавлении документов")
        print("  5. Мониторьте производительность системы")
        
        print(f"\n📅 План создан: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
    finally:
        planner.close_connection()

if __name__ == "__main__":
    main()

















