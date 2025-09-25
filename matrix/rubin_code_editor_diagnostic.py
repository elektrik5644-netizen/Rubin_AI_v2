#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Rubin AI - Система диагностики редактора кода
Анализирует код в реальном времени и предоставляет интеллектуальные рекомендации
"""

import re
import json
import sqlite3
import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import ast
import subprocess
import os

class DiagnosticSeverity(Enum):
    """Уровни серьезности диагностики"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class DiagnosticType(Enum):
    """Типы диагностики"""
    SYNTAX = "syntax"
    SEMANTIC = "semantic"
    STYLE = "style"
    PERFORMANCE = "performance"
    SECURITY = "security"
    PMAC_SPECIFIC = "pmac_specific"
    PLC_SPECIFIC = "plc_specific"
    BEST_PRACTICES = "best_practices"

@dataclass
class DiagnosticResult:
    """Результат диагностики"""
    id: str
    type: DiagnosticType
    severity: DiagnosticSeverity
    message: str
    line: int
    column: int
    length: int
    code_snippet: str
    suggestion: str
    quick_fix: Optional[str] = None
    documentation_url: Optional[str] = None
    confidence: float = 1.0

class RubinCodeEditorDiagnostic:
    """Основной класс диагностики редактора кода"""
    
    def __init__(self, db_path: str = "rubin_knowledge_base.db"):
        self.db_path = db_path
        self.diagnostic_rules = self.load_diagnostic_rules()
        self.code_patterns = self.load_code_patterns()
        self.best_practices = self.load_best_practices()
        
    def load_diagnostic_rules(self) -> Dict[str, List[Dict]]:
        """Загружает правила диагностики"""
        return {
            'python': [
                {
                    'pattern': r'import \*',
                    'type': DiagnosticType.STYLE,
                    'severity': DiagnosticSeverity.WARNING,
                    'message': 'Избегайте импорта всех модулей (import *)',
                    'suggestion': 'Импортируйте только необходимые функции'
                },
                {
                    'pattern': r'except:',
                    'type': DiagnosticType.SECURITY,
                    'severity': DiagnosticSeverity.ERROR,
                    'message': 'Не используйте голый except',
                    'suggestion': 'Указывайте конкретные типы исключений'
                },
                {
                    'pattern': r'eval\(',
                    'type': DiagnosticType.SECURITY,
                    'severity': DiagnosticSeverity.CRITICAL,
                    'message': 'Функция eval() опасна для безопасности',
                    'suggestion': 'Используйте ast.literal_eval() или другие безопасные альтернативы'
                }
            ],
            'cpp': [
                {
                    'pattern': r'using namespace std;',
                    'type': DiagnosticType.STYLE,
                    'severity': DiagnosticSeverity.WARNING,
                    'message': 'Избегайте using namespace std в заголовочных файлах',
                    'suggestion': 'Используйте std:: или конкретные using-декларации'
                },
                {
                    'pattern': r'new\s+\w+\[',
                    'type': DiagnosticType.PERFORMANCE,
                    'severity': DiagnosticSeverity.WARNING,
                    'message': 'Использование new[] может привести к утечкам памяти',
                    'suggestion': 'Используйте std::vector или умные указатели'
                }
            ],
            'pmac': [
                {
                    'pattern': r'I\d+\s*=\s*\d+',
                    'type': DiagnosticType.PMAC_SPECIFIC,
                    'severity': DiagnosticSeverity.INFO,
                    'message': 'PMAC переменная I обнаружена',
                    'suggestion': 'Убедитесь, что значение в допустимом диапазоне'
                },
                {
                    'pattern': r'#\d+\s*=\s*\d+',
                    'type': DiagnosticType.PMAC_SPECIFIC,
                    'severity': DiagnosticSeverity.INFO,
                    'message': 'PMAC переменная # обнаружена',
                    'suggestion': 'Проверьте соответствие номера переменной'
                }
            ],
            'plc': [
                {
                    'pattern': r'LD\s+[A-Z]\d+',
                    'type': DiagnosticType.PLC_SPECIFIC,
                    'severity': DiagnosticSeverity.INFO,
                    'message': 'Ladder Logic команда LD обнаружена',
                    'suggestion': 'Убедитесь в правильности адресации'
                },
                {
                    'pattern': r'TON\s+[A-Z]\d+',
                    'type': DiagnosticType.PLC_SPECIFIC,
                    'severity': DiagnosticSeverity.INFO,
                    'message': 'Таймер TON обнаружен',
                    'suggestion': 'Проверьте настройки времени таймера'
                }
            ]
        }
    
    def load_code_patterns(self) -> Dict[str, Dict]:
        """Загружает паттерны кода для анализа"""
        return {
            'performance_issues': {
                'nested_loops': r'for\s+.*:\s*\n.*for\s+.*:',
                'recursive_calls': r'def\s+\w+.*:\s*\n.*\w+\(',
                'string_concatenation': r'["\'].*["\']\s*\+\s*["\'].*["\']'
            },
            'security_issues': {
                'sql_injection': r'SELECT.*\+.*%',
                'path_traversal': r'open\s*\(\s*["\'].*\.\./',
                'command_injection': r'os\.system\s*\('
            },
            'pmac_specific': {
                'motion_commands': r'(ABS|INC|LINE|CIRCLE|SPLINE)\s+',
                'coordinate_systems': r'&[A-Z]\d+',
                'program_structures': r'OPEN\s+PROG\s+\d+'
            }
        }
    
    def load_best_practices(self) -> Dict[str, List[str]]:
        """Загружает лучшие практики программирования"""
        return {
            'python': [
                'Используйте type hints для лучшей читаемости',
                'Следуйте PEP 8 для стиля кода',
                'Используйте docstrings для документирования функций',
                'Предпочитайте list comprehensions простым циклам',
                'Используйте context managers (with) для работы с файлами'
            ],
            'cpp': [
                'Используйте const везде, где возможно',
                'Предпочитайте ссылки указателям',
                'Используйте RAII для управления ресурсами',
                'Следуйте правилу трех/пяти для классов',
                'Используйте std::unique_ptr вместо сырых указателей'
            ],
            'pmac': [
                'Используйте осмысленные имена для переменных',
                'Документируйте сложные алгоритмы движения',
                'Проверяйте границы движения перед выполнением',
                'Используйте программные структуры для организации кода',
                'Тестируйте программы на безопасных скоростях'
            ],
            'plc': [
                'Используйте комментарии для объяснения логики',
                'Организуйте код в функциональные блоки',
                'Проверяйте граничные условия',
                'Используйте таймеры для задержек вместо циклов',
                'Документируйте адресацию I/O'
            ]
        }
    
    def detect_language(self, code: str, filename: str = "") -> str:
        """Определяет язык программирования"""
        if filename:
            ext = filename.lower().split('.')[-1]
            if ext in ['py']:
                return 'python'
            elif ext in ['cpp', 'c', 'h', 'hpp']:
                return 'cpp'
            elif ext in ['pmc', 'pms']:
                return 'pmac'
            elif ext in ['awl', 'stl', 'lad']:
                return 'plc'
        
        # Определение по содержимому
        if 'import ' in code or 'def ' in code or 'class ' in code:
            return 'python'
        elif '#include' in code or 'int main' in code:
            return 'cpp'
        elif 'I' in code and '=' in code and any(c.isdigit() for c in code):
            return 'pmac'
        elif 'LD' in code or 'TON' in code or 'CTU' in code:
            return 'plc'
        
        return 'unknown'
    
    def analyze_syntax(self, code: str, language: str) -> List[DiagnosticResult]:
        """Анализирует синтаксис кода"""
        diagnostics = []
        
        if language == 'python':
            try:
                ast.parse(code)
            except SyntaxError as e:
                diagnostics.append(DiagnosticResult(
                    id=f"syntax_error_{e.lineno}",
                    type=DiagnosticType.SYNTAX,
                    severity=DiagnosticSeverity.ERROR,
                    message=f"Синтаксическая ошибка: {e.msg}",
                    line=e.lineno or 1,
                    column=e.offset or 1,
                    length=1,
                    code_snippet=code.split('\n')[e.lineno - 1] if e.lineno else "",
                    suggestion="Исправьте синтаксическую ошибку"
                ))
        
        return diagnostics
    
    def analyze_patterns(self, code: str, language: str) -> List[DiagnosticResult]:
        """Анализирует паттерны кода"""
        diagnostics = []
        lines = code.split('\n')
        
        # Применяем правила диагностики
        rules = self.diagnostic_rules.get(language, [])
        
        for rule in rules:
            pattern = re.compile(rule['pattern'], re.MULTILINE | re.IGNORECASE)
            matches = pattern.finditer(code)
            
            for match in matches:
                line_num = code[:match.start()].count('\n') + 1
                line_content = lines[line_num - 1] if line_num <= len(lines) else ""
                
                diagnostics.append(DiagnosticResult(
                    id=f"{rule['type'].value}_{line_num}_{match.start()}",
                    type=rule['type'],
                    severity=rule['severity'],
                    message=rule['message'],
                    line=line_num,
                    column=match.start() - code.rfind('\n', 0, match.start()) - 1,
                    length=match.end() - match.start(),
                    code_snippet=line_content,
                    suggestion=rule['suggestion']
                ))
        
        return diagnostics
    
    def analyze_performance(self, code: str, language: str) -> List[DiagnosticResult]:
        """Анализирует производительность кода"""
        diagnostics = []
        lines = code.split('\n')
        
        # Поиск проблем производительности
        performance_patterns = self.code_patterns.get('performance_issues', {})
        
        for pattern_name, pattern in performance_patterns.items():
            matches = re.finditer(pattern, code, re.MULTILINE | re.DOTALL)
            
            for match in matches:
                line_num = code[:match.start()].count('\n') + 1
                
                if pattern_name == 'nested_loops':
                    message = "Обнаружены вложенные циклы - возможны проблемы производительности"
                    suggestion = "Рассмотрите возможность оптимизации или использования векторизации"
                elif pattern_name == 'recursive_calls':
                    message = "Обнаружен рекурсивный вызов - возможен переполнение стека"
                    suggestion = "Рассмотрите итеративное решение или увеличьте лимит рекурсии"
                elif pattern_name == 'string_concatenation':
                    message = "Конкатенация строк в цикле неэффективна"
                    suggestion = "Используйте join() или f-строки"
                else:
                    message = f"Потенциальная проблема производительности: {pattern_name}"
                    suggestion = "Рассмотрите оптимизацию кода"
                
                diagnostics.append(DiagnosticResult(
                    id=f"performance_{pattern_name}_{line_num}",
                    type=DiagnosticType.PERFORMANCE,
                    severity=DiagnosticSeverity.WARNING,
                    message=message,
                    line=line_num,
                    column=1,
                    length=len(match.group()),
                    code_snippet=lines[line_num - 1] if line_num <= len(lines) else "",
                    suggestion=suggestion
                ))
        
        return diagnostics
    
    def analyze_security(self, code: str, language: str) -> List[DiagnosticResult]:
        """Анализирует безопасность кода"""
        diagnostics = []
        lines = code.split('\n')
        
        # Поиск проблем безопасности
        security_patterns = self.code_patterns.get('security_issues', {})
        
        for pattern_name, pattern in security_patterns.items():
            matches = re.finditer(pattern, code, re.MULTILINE | re.IGNORECASE)
            
            for match in matches:
                line_num = code[:match.start()].count('\n') + 1
                
                if pattern_name == 'sql_injection':
                    message = "Возможная SQL-инъекция"
                    suggestion = "Используйте параметризованные запросы"
                elif pattern_name == 'path_traversal':
                    message = "Возможная атака path traversal"
                    suggestion = "Валидируйте пути к файлам"
                elif pattern_name == 'command_injection':
                    message = "Возможная инъекция команд"
                    suggestion = "Используйте subprocess с параметрами вместо os.system"
                else:
                    message = f"Потенциальная уязвимость: {pattern_name}"
                    suggestion = "Проверьте безопасность кода"
                
                diagnostics.append(DiagnosticResult(
                    id=f"security_{pattern_name}_{line_num}",
                    type=DiagnosticType.SECURITY,
                    severity=DiagnosticSeverity.CRITICAL,
                    message=message,
                    line=line_num,
                    column=1,
                    length=len(match.group()),
                    code_snippet=lines[line_num - 1] if line_num <= len(lines) else "",
                    suggestion=suggestion
                ))
        
        return diagnostics
    
    def analyze_pmac_specific(self, code: str) -> List[DiagnosticResult]:
        """Анализирует PMAC-специфичный код"""
        diagnostics = []
        lines = code.split('\n')
        
        # PMAC-специфичные паттерны
        pmac_patterns = self.code_patterns.get('pmac_specific', {})
        
        for pattern_name, pattern in pmac_patterns.items():
            matches = re.finditer(pattern, code, re.MULTILINE | re.IGNORECASE)
            
            for match in matches:
                line_num = code[:match.start()].count('\n') + 1
                
                if pattern_name == 'motion_commands':
                    message = "PMAC команда движения обнаружена"
                    suggestion = "Убедитесь в правильности параметров движения"
                elif pattern_name == 'coordinate_systems':
                    message = "Система координат PMAC обнаружена"
                    suggestion = "Проверьте настройки системы координат"
                elif pattern_name == 'program_structures':
                    message = "Структура программы PMAC обнаружена"
                    suggestion = "Убедитесь в правильности организации программы"
                else:
                    message = f"PMAC-специфичный элемент: {pattern_name}"
                    suggestion = "Проверьте соответствие стандартам PMAC"
                
                diagnostics.append(DiagnosticResult(
                    id=f"pmac_{pattern_name}_{line_num}",
                    type=DiagnosticType.PMAC_SPECIFIC,
                    severity=DiagnosticSeverity.INFO,
                    message=message,
                    line=line_num,
                    column=1,
                    length=len(match.group()),
                    code_snippet=lines[line_num - 1] if line_num <= len(lines) else "",
                    suggestion=suggestion
                ))
        
        return diagnostics
    
    def generate_recommendations(self, code: str, language: str) -> List[str]:
        """Генерирует рекомендации по улучшению кода"""
        recommendations = []
        
        # Базовые рекомендации
        if language in self.best_practices:
            recommendations.extend(self.best_practices[language])
        
        # Специфичные рекомендации на основе анализа
        if language == 'python':
            if 'print(' in code and 'logging' not in code:
                recommendations.append("Рассмотрите использование logging вместо print для отладки")
            
            if 'global ' in code:
                recommendations.append("Избегайте использования global переменных")
            
            if len(code.split('\n')) > 100:
                recommendations.append("Файл слишком большой - рассмотрите разбиение на модули")
        
        elif language == 'cpp':
            if 'new ' in code and 'delete ' not in code:
                recommendations.append("Используйте умные указатели для автоматического управления памятью")
            
            if 'malloc(' in code:
                recommendations.append("В C++ предпочитайте new/delete или умные указатели malloc/free")
        
        elif language == 'pmac':
            if 'I' in code and 'P' in code:
                recommendations.append("Убедитесь в правильности соотношения I и P переменных")
            
            if 'ABS' in code or 'INC' in code:
                recommendations.append("Проверьте границы движения перед выполнением команд")
        
        return recommendations
    
    def save_diagnostic_results(self, diagnostics: List[DiagnosticResult], code_hash: str):
        """Сохраняет результаты диагностики в базу данных"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Создаем таблицу для диагностики, если её нет
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS code_diagnostics (
                    id TEXT PRIMARY KEY,
                    code_hash TEXT,
                    diagnostic_type TEXT,
                    severity TEXT,
                    message TEXT,
                    line INTEGER,
                    column INTEGER,
                    length INTEGER,
                    code_snippet TEXT,
                    suggestion TEXT,
                    quick_fix TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Сохраняем результаты
            for diagnostic in diagnostics:
                cursor.execute('''
                    INSERT OR REPLACE INTO code_diagnostics 
                    (id, code_hash, diagnostic_type, severity, message, line, column, 
                     length, code_snippet, suggestion, quick_fix)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    diagnostic.id,
                    code_hash,
                    diagnostic.type.value,
                    diagnostic.severity.value,
                    diagnostic.message,
                    diagnostic.line,
                    diagnostic.column,
                    diagnostic.length,
                    diagnostic.code_snippet,
                    diagnostic.suggestion,
                    diagnostic.quick_fix
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Ошибка сохранения диагностики: {e}")
    
    def diagnose_code(self, code: str, filename: str = "") -> Dict[str, Any]:
        """Основная функция диагностики кода"""
        if not code.strip():
            return {
                'language': 'unknown',
                'diagnostics': [],
                'recommendations': [],
                'summary': 'Код пуст'
            }
        
        # Определяем язык
        language = self.detect_language(code, filename)
        
        # Выполняем различные виды анализа
        all_diagnostics = []
        
        # Синтаксический анализ
        syntax_diagnostics = self.analyze_syntax(code, language)
        all_diagnostics.extend(syntax_diagnostics)
        
        # Анализ паттернов
        pattern_diagnostics = self.analyze_patterns(code, language)
        all_diagnostics.extend(pattern_diagnostics)
        
        # Анализ производительности
        performance_diagnostics = self.analyze_performance(code, language)
        all_diagnostics.extend(performance_diagnostics)
        
        # Анализ безопасности
        security_diagnostics = self.analyze_security(code, language)
        all_diagnostics.extend(security_diagnostics)
        
        # PMAC-специфичный анализ
        if language == 'pmac':
            pmac_diagnostics = self.analyze_pmac_specific(code)
            all_diagnostics.extend(pmac_diagnostics)
        
        # Генерируем рекомендации
        recommendations = self.generate_recommendations(code, language)
        
        # Создаем хэш кода для кэширования
        code_hash = str(hash(code))
        
        # Сохраняем результаты
        self.save_diagnostic_results(all_diagnostics, code_hash)
        
        # Создаем сводку
        summary = self.create_summary(all_diagnostics, language)
        
        return {
            'language': language,
            'diagnostics': [
                {
                    'id': d.id,
                    'type': d.type.value,
                    'severity': d.severity.value,
                    'message': d.message,
                    'line': d.line,
                    'column': d.column,
                    'length': d.length,
                    'code_snippet': d.code_snippet,
                    'suggestion': d.suggestion,
                    'quick_fix': d.quick_fix,
                    'confidence': d.confidence
                }
                for d in all_diagnostics
            ],
            'recommendations': recommendations,
            'summary': summary,
            'code_hash': code_hash,
            'timestamp': datetime.datetime.now().isoformat()
        }
    
    def create_summary(self, diagnostics: List[DiagnosticResult], language: str) -> str:
        """Создает сводку диагностики"""
        if not diagnostics:
            return f"✅ Код на языке {language} не содержит критических проблем"
        
        error_count = sum(1 for d in diagnostics if d.severity == DiagnosticSeverity.ERROR)
        warning_count = sum(1 for d in diagnostics if d.severity == DiagnosticSeverity.WARNING)
        critical_count = sum(1 for d in diagnostics if d.severity == DiagnosticSeverity.CRITICAL)
        
        summary_parts = []
        
        if critical_count > 0:
            summary_parts.append(f"🚨 {critical_count} критических проблем")
        if error_count > 0:
            summary_parts.append(f"❌ {error_count} ошибок")
        if warning_count > 0:
            summary_parts.append(f"⚠️ {warning_count} предупреждений")
        
        if summary_parts:
            return f"Обнаружено: {', '.join(summary_parts)}"
        else:
            return f"✅ Код на языке {language} в хорошем состоянии"

def main():
    """Основная функция для тестирования"""
    print("🔍 SMART RUBIN AI - ДИАГНОСТИКА РЕДАКТОРА КОДА")
    print("=" * 60)
    
    # Создаем диагностическую систему
    diagnostic = RubinCodeEditorDiagnostic()
    
    # Тестовые примеры кода
    test_cases = [
        {
            'name': 'Python с проблемами',
            'code': '''
import *
def bad_function():
    try:
        result = eval(input("Enter code: "))
        return result
    except:
        pass
''',
            'filename': 'test.py'
        },
        {
            'name': 'PMAC код',
            'code': '''
I100 = 1000
#100 = 2000
ABS X100 Y200
LINE X200 Y300
''',
            'filename': 'motion.pmc'
        },
        {
            'name': 'C++ с проблемами',
            'code': '''
#include <iostream>
using namespace std;

int main() {
    int* arr = new int[100];
    for(int i = 0; i < 100; i++) {
        for(int j = 0; j < 100; j++) {
            arr[i] = i * j;
        }
    }
    return 0;
}
''',
            'filename': 'test.cpp'
        }
    ]
    
    for test_case in test_cases:
        print(f"\n📝 Тест: {test_case['name']}")
        print("-" * 40)
        
        result = diagnostic.diagnose_code(test_case['code'], test_case['filename'])
        
        print(f"🔤 Язык: {result['language']}")
        print(f"📊 Сводка: {result['summary']}")
        
        if result['diagnostics']:
            print(f"🔍 Найдено проблем: {len(result['diagnostics'])}")
            for diag in result['diagnostics'][:3]:  # Показываем первые 3
                print(f"   {diag['severity'].upper()}: {diag['message']} (строка {diag['line']})")
        
        if result['recommendations']:
            print(f"💡 Рекомендации: {len(result['recommendations'])}")
            for rec in result['recommendations'][:2]:  # Показываем первые 2
                print(f"   • {rec}")
    
    print("\n🎉 ТЕСТИРОВАНИЕ ЗАВЕРШЕНО!")

if __name__ == "__main__":
    main()
