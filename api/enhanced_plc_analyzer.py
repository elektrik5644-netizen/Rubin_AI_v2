#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Улучшенный анализатор PLC кода по методологии:
1. Лексический анализ (токенизация)
2. Синтаксический анализ (парсинг)
3. Семантический анализ (проверка смысла)
4. AI-анализ (экспертная оценка)
"""

import re
import json
from typing import List, Dict, Any, Tuple

class PLCTokenizer:
    """Шаг 1: Лексический анализ - разбиение на токены"""
    
    def __init__(self):
        # Регулярные выражения для токенов PLC
        self.token_patterns = [
            ('PROGRAM', r'\bPROGRAM\b'),
            ('FUNCTION', r'\bFUNCTION\b'),
            ('VAR', r'\bVAR\b'),
            ('END_VAR', r'\bEND_VAR\b'),
            ('END_PROGRAM', r'\bEND_PROGRAM\b'),
            ('END_FUNCTION', r'\bEND_FUNCTION\b'),
            ('IF', r'\bIF\b'),
            ('THEN', r'\bTHEN\b'),
            ('ELSE', r'\bELSE\b'),
            ('END_IF', r'\bEND_IF\b'),
            ('WHILE', r'\bWHILE\b'),
            ('DO', r'\bDO\b'),
            ('END_WHILE', r'\bEND_WHILE\b'),
            ('FOR', r'\bFOR\b'),
            ('TO', r'\bTO\b'),
            ('BY', r'\bBY\b'),
            ('END_FOR', r'\bEND_FOR\b'),
            ('AND', r'\bAND\b'),
            ('OR', r'\bOR\b'),
            ('NOT', r'\bNOT\b'),
            ('XOR', r'\bXOR\b'),
            ('BOOL', r'\bBOOL\b'),
            ('INT', r'\bINT\b'),
            ('DINT', r'\bDINT\b'),
            ('REAL', r'\bREAL\b'),
            ('STRING', r'\bSTRING\b'),
            ('TRUE', r'\bTRUE\b'),
            ('FALSE', r'\bFALSE\b'),
            ('ASSIGN', r':='),
            ('EQUALS', r'='),
            ('PLUS', r'\+'),
            ('MINUS', r'-'),
            ('MULTIPLY', r'\*'),
            ('DIVIDE', r'/'),
            ('MODULO', r'MOD'),
            ('GREATER', r'>'),
            ('LESS', r'<'),
            ('GREATER_EQUAL', r'>='),
            ('LESS_EQUAL', r'<='),
            ('NOT_EQUAL', r'<>'),
            ('LPAREN', r'\('),
            ('RPAREN', r'\)'),
            ('SEMICOLON', r';'),
            ('COLON', r':'),
            ('COMMA', r','),
            ('DOT', r'\.'),
            ('COMMENT', r'//.*'),
            ('COMMENT_BLOCK', r'/\*.*?\*/'),
            ('STRING_LITERAL', r'"[^"]*"'),
            ('NUMBER', r'\b\d+(\.\d+)?\b'),
            ('IO_ADDRESS', r'\b[IQM]\d+\.\d+\b'),
            ('DB_ADDRESS', r'\bDB\d+\.DB[XY]\d+\.\d+\b'),
            ('IDENTIFIER', r'\b[A-Za-z_][A-Za-z0-9_]*\b'),
            ('WHITESPACE', r'\s+'),
        ]
    
    def tokenize(self, code: str) -> List[Tuple[str, str]]:
        """Разбивает код на токены"""
        tokens = []
        position = 0
        
        while position < len(code):
            matched = False
            
            for token_type, pattern in self.token_patterns:
                regex = re.compile(pattern, re.IGNORECASE)
                match = regex.match(code, position)
                
                if match:
                    token_value = match.group(0)
                    if token_type != 'WHITESPACE':  # Пропускаем пробелы
                        tokens.append((token_type, token_value))
                    position = match.end()
                    matched = True
                    break
            
            if not matched:
                # Неизвестный символ
                tokens.append(('UNKNOWN', code[position]))
                position += 1
        
        return tokens

class PLCParser:
    """Шаг 2: Синтаксический анализ - проверка структуры"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
    
    def parse(self, tokens: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Парсит токены и проверяет синтаксис"""
        self.errors = []
        self.warnings = []
        
        # Проверяем основные конструкции
        self._check_program_structure(tokens)
        self._check_control_structures(tokens)
        self._check_brackets_balance(tokens)
        self._check_variable_declarations(tokens)
        
        return {
            'errors': self.errors,
            'warnings': self.warnings,
            'syntax_valid': len(self.errors) == 0
        }
    
    def _check_program_structure(self, tokens: List[Tuple[str, str]]):
        """Проверяет структуру программы"""
        program_count = sum(1 for token_type, _ in tokens if token_type == 'PROGRAM')
        end_program_count = sum(1 for token_type, _ in tokens if token_type == 'END_PROGRAM')
        
        if program_count != end_program_count:
            self.errors.append(f"Несоответствие PROGRAM/END_PROGRAM: {program_count} vs {end_program_count}")
    
    def _check_control_structures(self, tokens: List[Tuple[str, str]]):
        """Проверяет управляющие структуры"""
        if_count = sum(1 for token_type, _ in tokens if token_type == 'IF')
        endif_count = sum(1 for token_type, _ in tokens if token_type == 'END_IF')
        
        if if_count != endif_count:
            self.errors.append(f"Несоответствие IF/END_IF: {if_count} vs {endif_count}")
        
        while_count = sum(1 for token_type, _ in tokens if token_type == 'WHILE')
        endwhile_count = sum(1 for token_type, _ in tokens if token_type == 'END_WHILE')
        
        if while_count != endwhile_count:
            self.errors.append(f"Несоответствие WHILE/END_WHILE: {while_count} vs {endwhile_count}")
    
    def _check_brackets_balance(self, tokens: List[Tuple[str, str]]):
        """Проверяет баланс скобок"""
        lparen_count = sum(1 for token_type, _ in tokens if token_type == 'LPAREN')
        rparen_count = sum(1 for token_type, _ in tokens if token_type == 'RPAREN')
        
        if lparen_count != rparen_count:
            self.errors.append(f"Несбалансированные скобки: {lparen_count} vs {rparen_count}")
    
    def _check_variable_declarations(self, tokens: List[Tuple[str, str]]):
        """Проверяет объявления переменных"""
        var_count = sum(1 for token_type, _ in tokens if token_type == 'VAR')
        end_var_count = sum(1 for token_type, _ in tokens if token_type == 'END_VAR')
        
        if var_count != end_var_count:
            self.errors.append(f"Несоответствие VAR/END_VAR: {var_count} vs {end_var_count}")

class PLCSemanticAnalyzer:
    """Шаг 3: Семантический анализ - проверка смысла"""
    
    def __init__(self):
        self.variables = {}
        self.functions = {}
        self.errors = []
        self.warnings = []
    
    def analyze(self, tokens: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Выполняет семантический анализ"""
        self.variables = {}
        self.functions = {}
        self.errors = []
        self.warnings = []
        
        self._extract_variables(tokens)
        self._extract_functions(tokens)
        self._check_variable_usage(tokens)
        self._check_function_calls(tokens)
        
        return {
            'variables': self.variables,
            'functions': self.functions,
            'errors': self.errors,
            'warnings': self.warnings,
            'semantic_valid': len(self.errors) == 0
        }
    
    def _extract_variables(self, tokens: List[Tuple[str, str]]):
        """Извлекает объявленные переменные"""
        in_var_block = False
        
        for i, (token_type, token_value) in enumerate(tokens):
            if token_type == 'VAR':
                in_var_block = True
            elif token_type == 'END_VAR':
                in_var_block = False
            elif in_var_block and token_type == 'IDENTIFIER':
                # Ищем тип переменной
                if i + 2 < len(tokens) and tokens[i + 1][0] == 'COLON':
                    var_type = tokens[i + 2][1]
                    self.variables[token_value] = {
                        'type': var_type,
                        'declared': True,
                        'used': False
                    }
    
    def _extract_functions(self, tokens: List[Tuple[str, str]]):
        """Извлекает объявленные функции"""
        for i, (token_type, token_value) in enumerate(tokens):
            if token_type == 'FUNCTION' and i + 1 < len(tokens):
                func_name = tokens[i + 1][1]
                self.functions[func_name] = {
                    'declared': True,
                    'called': False
                }
    
    def _check_variable_usage(self, tokens: List[Tuple[str, str]]):
        """Проверяет использование переменных"""
        for token_type, token_value in tokens:
            if token_type == 'IDENTIFIER' and token_value in self.variables:
                self.variables[token_value]['used'] = True
        
        # Проверяем неиспользуемые переменные
        for var_name, var_info in self.variables.items():
            if not var_info['used']:
                self.warnings.append(f"Неиспользуемая переменная: {var_name}")
    
    def _check_function_calls(self, tokens: List[Tuple[str, str]]):
        """Проверяет вызовы функций"""
        for token_type, token_value in tokens:
            if token_type == 'IDENTIFIER' and token_value in self.functions:
                self.functions[token_value]['called'] = True

class PLCAIAnalyzer:
    """Шаг 4: AI-анализ - экспертная оценка"""
    
    def __init__(self):
        self.recommendations = []
        self.quality_score = 0
    
    def analyze(self, code: str, tokens: List[Tuple[str, str]], 
                syntax_result: Dict, semantic_result: Dict) -> Dict[str, Any]:
        """Выполняет AI-анализ кода"""
        self.recommendations = []
        self.quality_score = 0
        
        self._analyze_code_quality(code)
        self._analyze_complexity(tokens)
        self._analyze_best_practices(code)
        self._analyze_safety_issues(code)
        
        return {
            'recommendations': self.recommendations,
            'quality_score': self.quality_score,
            'complexity_level': self._get_complexity_level(),
            'safety_issues': self._get_safety_issues(code)
        }
    
    def _analyze_code_quality(self, code: str):
        """Анализирует качество кода"""
        lines = code.split('\n')
        comment_lines = sum(1 for line in lines if line.strip().startswith('//'))
        total_lines = len([line for line in lines if line.strip()])
        
        comment_ratio = comment_lines / total_lines if total_lines > 0 else 0
        
        if comment_ratio < 0.1:
            self.recommendations.append("Добавьте больше комментариев (рекомендуется >10%)")
            self.quality_score -= 2
        elif comment_ratio > 0.3:
            self.recommendations.append("Слишком много комментариев, возможно избыточность")
            self.quality_score -= 1
        else:
            self.quality_score += 2
    
    def _analyze_complexity(self, tokens: List[Tuple[str, str]]):
        """Анализирует сложность кода"""
        if_count = sum(1 for token_type, _ in tokens if token_type == 'IF')
        while_count = sum(1 for token_type, _ in tokens if token_type == 'WHILE')
        for_count = sum(1 for token_type, _ in tokens if token_type == 'FOR')
        
        total_control_structures = if_count + while_count + for_count
        
        if total_control_structures > 20:
            self.recommendations.append("Высокая сложность - рассмотрите разбиение на функции")
            self.quality_score -= 3
        elif total_control_structures > 10:
            self.recommendations.append("Средняя сложность - проверьте возможность упрощения")
            self.quality_score -= 1
        else:
            self.quality_score += 1
    
    def _analyze_best_practices(self, code: str):
        """Анализирует соблюдение лучших практик"""
        # Проверяем использование магических чисел
        numbers = re.findall(r'\b\d+\b', code)
        magic_numbers = [n for n in numbers if int(n) > 10 and n not in ['100', '1000']]
        
        if len(magic_numbers) > 5:
            self.recommendations.append("Используйте именованные константы вместо магических чисел")
            self.quality_score -= 2
        
        # Проверяем длинные строки
        lines = code.split('\n')
        long_lines = [i for i, line in enumerate(lines) if len(line) > 80]
        
        if len(long_lines) > len(lines) * 0.2:
            self.recommendations.append("Слишком длинные строки - улучшите читаемость")
            self.quality_score -= 1
    
    def _analyze_safety_issues(self, code: str):
        """Анализирует проблемы безопасности"""
        safety_issues = []
        
        # Проверяем на критические ошибки
        if 'ERROR' in code.upper() and 'HANDLING' not in code.upper():
            safety_issues.append("Обнаружены ERROR переменные без обработки ошибок")
        
        if 'STOP' in code.upper() and 'SAFETY' not in code.upper():
            safety_issues.append("Обнаружены STOP команды без проверки безопасности")
        
        if safety_issues:
            self.recommendations.extend(safety_issues)
            self.quality_score -= 3
        
        return safety_issues
    
    def _get_complexity_level(self) -> str:
        """Определяет уровень сложности"""
        if self.quality_score >= 8:
            return "Низкая"
        elif self.quality_score >= 5:
            return "Средняя"
        else:
            return "Высокая"
    
    def _get_safety_issues(self, code: str) -> List[str]:
        """Возвращает список проблем безопасности"""
        issues = []
        
        if 'ERROR' in code.upper():
            issues.append("Обнаружены ERROR переменные")
        
        if 'STOP' in code.upper():
            issues.append("Обнаружены STOP команды")
        
        if 'FAULT' in code.upper():
            issues.append("Обнаружены FAULT переменные")
        
        return issues

def analyze_plc_code(code: str) -> Dict[str, Any]:
    """Главная функция анализа PLC кода"""
    print("🔍 Начинаем анализ PLC кода...")
    
    # Шаг 1: Лексический анализ
    print("📝 Шаг 1: Лексический анализ (токенизация)...")
    tokenizer = PLCTokenizer()
    tokens = tokenizer.tokenize(code)
    print(f"   Найдено токенов: {len(tokens)}")
    
    # Шаг 2: Синтаксический анализ
    print("🔧 Шаг 2: Синтаксический анализ (парсинг)...")
    parser = PLCParser()
    syntax_result = parser.parse(tokens)
    print(f"   Синтаксических ошибок: {len(syntax_result['errors'])}")
    print(f"   Предупреждений: {len(syntax_result['warnings'])}")
    
    # Шаг 3: Семантический анализ
    print("🧠 Шаг 3: Семантический анализ (проверка смысла)...")
    semantic_analyzer = PLCSemanticAnalyzer()
    semantic_result = semantic_analyzer.analyze(tokens)
    print(f"   Семантических ошибок: {len(semantic_result['errors'])}")
    print(f"   Переменных: {len(semantic_result['variables'])}")
    print(f"   Функций: {len(semantic_result['functions'])}")
    
    # Шаг 4: AI-анализ
    print("🤖 Шаг 4: AI-анализ (экспертная оценка)...")
    ai_analyzer = PLCAIAnalyzer()
    ai_result = ai_analyzer.analyze(code, tokens, syntax_result, semantic_result)
    print(f"   Оценка качества: {ai_result['quality_score']}/10")
    print(f"   Рекомендаций: {len(ai_result['recommendations'])}")
    
    return {
        'tokens': tokens,
        'syntax': syntax_result,
        'semantic': semantic_result,
        'ai_analysis': ai_result,
        'summary': {
            'total_tokens': len(tokens),
            'syntax_errors': len(syntax_result['errors']),
            'semantic_errors': len(semantic_result['errors']),
            'variables_count': len(semantic_result['variables']),
            'functions_count': len(semantic_result['functions']),
            'quality_score': ai_result['quality_score'],
            'complexity_level': ai_result['complexity_level']
        }
    }

def main():
    """Демонстрация анализатора"""
    print("🚀 Улучшенный анализатор PLC кода")
    print("=" * 50)
    
    # Читаем тестовый файл
    try:
        with open('test_simple.plc', 'r', encoding='utf-8') as f:
            code = f.read()
    except FileNotFoundError:
        print("❌ Файл test_simple.plc не найден")
        return
    
    # Анализируем код
    result = analyze_plc_code(code)
    
    print("\n📊 РЕЗУЛЬТАТЫ АНАЛИЗА:")
    print("=" * 50)
    
    # Выводим сводку
    summary = result['summary']
    print(f"📈 Общая статистика:")
    print(f"   • Токенов: {summary['total_tokens']}")
    print(f"   • Синтаксических ошибок: {summary['syntax_errors']}")
    print(f"   • Семантических ошибок: {summary['semantic_errors']}")
    print(f"   • Переменных: {summary['variables_count']}")
    print(f"   • Функций: {summary['functions_count']}")
    print(f"   • Оценка качества: {summary['quality_score']}/10")
    print(f"   • Уровень сложности: {summary['complexity_level']}")
    
    # Выводим ошибки
    if result['syntax']['errors']:
        print(f"\n❌ Синтаксические ошибки:")
        for error in result['syntax']['errors']:
            print(f"   • {error}")
    
    if result['semantic']['errors']:
        print(f"\n❌ Семантические ошибки:")
        for error in result['semantic']['errors']:
            print(f"   • {error}")
    
    # Выводим рекомендации
    if result['ai_analysis']['recommendations']:
        print(f"\n💡 Рекомендации:")
        for rec in result['ai_analysis']['recommendations']:
            print(f"   • {rec}")
    
    # Выводим переменные
    if result['semantic']['variables']:
        print(f"\n📋 Переменные:")
        for var_name, var_info in result['semantic']['variables'].items():
            print(f"   • {var_name}: {var_info['type']}")
    
    print(f"\n✅ Анализ завершен!")

if __name__ == "__main__":
    main()
