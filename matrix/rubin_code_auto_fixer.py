#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Rubin AI - Автоматическое исправление ошибок и генерация кода
Система, которая может автоматически исправлять ошибки и писать код в редакторе
"""

import re
import ast
import json
import sqlite3
import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import os
import shutil

class FixType(Enum):
    """Типы исправлений"""
    SYNTAX_FIX = "syntax_fix"
    SECURITY_FIX = "security_fix"
    PERFORMANCE_FIX = "performance_fix"
    STYLE_FIX = "style_fix"
    LOGIC_FIX = "logic_fix"
    CODE_GENERATION = "code_generation"

class FixConfidence(Enum):
    """Уровень уверенности в исправлении"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class CodeFix:
    """Информация об исправлении кода"""
    id: str
    type: FixType
    confidence: FixConfidence
    original_code: str
    fixed_code: str
    line_start: int
    line_end: int
    column_start: int
    column_end: int
    description: str
    explanation: str
    before_snippet: str
    after_snippet: str

@dataclass
class CodeGenerationRequest:
    """Запрос на генерацию кода"""
    id: str
    description: str
    language: str
    context: str
    requirements: List[str]
    examples: List[str]
    generated_code: Optional[str] = None
    confidence: FixConfidence = FixConfidence.MEDIUM

class RubinCodeAutoFixer:
    """Основной класс автоматического исправления кода"""
    
    def __init__(self, db_path: str = "rubin_code_fixes.db"):
        self.db_path = db_path
        self.fix_patterns = self.load_fix_patterns()
        self.code_templates = self.load_code_templates()
        self.init_database()
    
    def init_database(self):
        """Инициализация базы данных"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS code_fixes (
                    id TEXT PRIMARY KEY,
                    fix_type TEXT,
                    confidence TEXT,
                    original_code TEXT,
                    fixed_code TEXT,
                    line_start INTEGER,
                    line_end INTEGER,
                    description TEXT,
                    explanation TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    success BOOLEAN DEFAULT FALSE
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS code_generations (
                    id TEXT PRIMARY KEY,
                    description TEXT,
                    language TEXT,
                    context TEXT,
                    requirements TEXT,
                    generated_code TEXT,
                    confidence TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    success BOOLEAN DEFAULT FALSE
                )
            ''')
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Ошибка инициализации БД: {e}")
    
    def load_fix_patterns(self) -> Dict[str, List[Dict]]:
        """Загружает паттерны для исправления ошибок"""
        return {
            'python': [
                {
                    'pattern': r'import \*',
                    'fix_type': FixType.STYLE_FIX,
                    'confidence': FixConfidence.HIGH,
                    'description': 'Замена import * на конкретные импорты',
                    'fix_function': self.fix_import_star
                },
                {
                    'pattern': r'except:',
                    'fix_type': FixType.SECURITY_FIX,
                    'confidence': FixConfidence.VERY_HIGH,
                    'description': 'Замена голого except на конкретные исключения',
                    'fix_function': self.fix_bare_except
                },
                {
                    'pattern': r'eval\(',
                    'fix_type': FixType.SECURITY_FIX,
                    'confidence': FixConfidence.VERY_HIGH,
                    'description': 'Замена eval() на безопасную альтернативу',
                    'fix_function': self.fix_eval_usage
                },
                {
                    'pattern': r'(\w+)\s*\+\s*str\(',
                    'fix_type': FixType.PERFORMANCE_FIX,
                    'confidence': FixConfidence.HIGH,
                    'description': 'Оптимизация конкатенации строк',
                    'fix_function': self.fix_string_concatenation
                },
                {
                    'pattern': r'for\s+(\w+)\s+in\s+range\((\d+)\):\s*\n\s*for\s+(\w+)\s+in\s+range\((\d+)\):',
                    'fix_type': FixType.PERFORMANCE_FIX,
                    'confidence': FixConfidence.MEDIUM,
                    'description': 'Оптимизация вложенных циклов',
                    'fix_function': self.fix_nested_loops
                }
            ],
            'cpp': [
                {
                    'pattern': r'using namespace std;',
                    'fix_type': FixType.STYLE_FIX,
                    'confidence': FixConfidence.HIGH,
                    'description': 'Замена using namespace std на конкретные using',
                    'fix_function': self.fix_using_namespace
                },
                {
                    'pattern': r'new\s+(\w+)\[(\d+)\]',
                    'fix_type': FixType.PERFORMANCE_FIX,
                    'confidence': FixConfidence.HIGH,
                    'description': 'Замена new[] на std::vector',
                    'fix_function': self.fix_new_array
                },
                {
                    'pattern': r'malloc\(',
                    'fix_type': FixType.PERFORMANCE_FIX,
                    'confidence': FixConfidence.HIGH,
                    'description': 'Замена malloc на new или умные указатели',
                    'fix_function': self.fix_malloc_usage
                }
            ],
            'pmac': [
                {
                    'pattern': r'I(\d+)\s*=\s*(\d+)',
                    'fix_type': FixType.STYLE_FIX,
                    'confidence': FixConfidence.MEDIUM,
                    'description': 'Добавление комментариев к PMAC переменным',
                    'fix_function': self.fix_pmac_variables
                },
                {
                    'pattern': r'(ABS|INC|LINE|CIRCLE)\s+([XYZ]\d+)',
                    'fix_type': FixType.STYLE_FIX,
                    'confidence': FixConfidence.MEDIUM,
                    'description': 'Добавление проверок безопасности для команд движения',
                    'fix_function': self.fix_pmac_motion_commands
                }
            ]
        }
    
    def load_code_templates(self) -> Dict[str, Dict]:
        """Загружает шаблоны для генерации кода"""
        return {
            'python': {
                'function_template': '''def {function_name}({parameters}):
    """
    {description}
    
    Args:
        {args_doc}
    
    Returns:
        {return_doc}
    """
    {body}''',
                'class_template': '''class {class_name}:
    """
    {description}
    """
    
    def __init__(self{init_params}):
        {init_body}
    
    def {method_name}(self{method_params}):
        """
        {method_description}
        """
        {method_body}''',
                'error_handling_template': '''try:
    {code}
except {exception_type} as e:
    {error_handling}
except Exception as e:
    logger.error(f"Неожиданная ошибка: {{e}}")
    {fallback_handling}'''
            },
            'cpp': {
                'class_template': '''class {class_name} {{
public:
    {class_name}({constructor_params});
    ~{class_name}();
    
    {methods}
    
private:
    {private_members}
}};''',
                'function_template': '''{return_type} {function_name}({parameters}) {{
    {body}
}}'''
            },
            'pmac': {
                'program_template': '''OPEN PROG {program_number}
CLEAR
// {description}

// Инициализация переменных
{variables}

// Основная логика
{logic}

// Завершение программы
CLOSE''',
                'motion_template': '''// {description}
IF ({safety_condition}) THEN
    {motion_command}
ELSE
    // Ошибка безопасности
    M{error_flag} = 1
ENDIF'''
            }
        }
    
    def fix_import_star(self, match, code, line_num) -> CodeFix:
        """Исправляет import *"""
        original = match.group(0)
        # Простое исправление - замена на конкретный импорт
        fixed = original.replace('import *', 'import specific_function')
        
        return CodeFix(
            id=f"fix_import_star_{line_num}",
            type=FixType.STYLE_FIX,
            confidence=FixConfidence.HIGH,
            original_code=original,
            fixed_code=fixed,
            line_start=line_num,
            line_end=line_num,
            column_start=match.start(),
            column_end=match.end(),
            description="Замена import * на конкретные импорты",
            explanation="import * импортирует все функции модуля, что может привести к конфликтам имен",
            before_snippet=original,
            after_snippet=fixed
        )
    
    def fix_bare_except(self, match, code, line_num) -> CodeFix:
        """Исправляет голый except"""
        original = match.group(0)
        fixed = "except Exception as e:"
        
        return CodeFix(
            id=f"fix_bare_except_{line_num}",
            type=FixType.SECURITY_FIX,
            confidence=FixConfidence.VERY_HIGH,
            original_code=original,
            fixed_code=fixed,
            line_start=line_num,
            line_end=line_num,
            column_start=match.start(),
            column_end=match.end(),
            description="Замена голого except на конкретные исключения",
            explanation="Голый except может скрыть критические ошибки",
            before_snippet=original,
            after_snippet=fixed
        )
    
    def fix_eval_usage(self, match, code, line_num) -> CodeFix:
        """Исправляет использование eval()"""
        original = match.group(0)
        fixed = "ast.literal_eval("
        
        return CodeFix(
            id=f"fix_eval_{line_num}",
            type=FixType.SECURITY_FIX,
            confidence=FixConfidence.VERY_HIGH,
            original_code=original,
            fixed_code=fixed,
            line_start=line_num,
            line_end=line_num,
            column_start=match.start(),
            column_end=match.end(),
            description="Замена eval() на безопасную альтернативу",
            explanation="eval() может выполнить произвольный код и создать уязвимость",
            before_snippet=original,
            after_snippet=fixed
        )
    
    def fix_string_concatenation(self, match, code, line_num) -> CodeFix:
        """Исправляет неэффективную конкатенацию строк"""
        original = match.group(0)
        var_name = match.group(1)
        fixed = f"{var_name} = ''.join([{var_name}, str("
        
        return CodeFix(
            id=f"fix_string_concat_{line_num}",
            type=FixType.PERFORMANCE_FIX,
            confidence=FixConfidence.HIGH,
            original_code=original,
            fixed_code=fixed,
            line_start=line_num,
            line_end=line_num,
            column_start=match.start(),
            column_end=match.end(),
            description="Оптимизация конкатенации строк",
            explanation="Конкатенация строк в цикле неэффективна, лучше использовать join()",
            before_snippet=original,
            after_snippet=fixed
        )
    
    def fix_nested_loops(self, match, code, line_num) -> CodeFix:
        """Исправляет вложенные циклы"""
        original = match.group(0)
        # Простое исправление - добавление комментария о возможной оптимизации
        fixed = original + "\n    # TODO: Рассмотреть оптимизацию вложенных циклов"
        
        return CodeFix(
            id=f"fix_nested_loops_{line_num}",
            type=FixType.PERFORMANCE_FIX,
            confidence=FixConfidence.MEDIUM,
            original_code=original,
            fixed_code=fixed,
            line_start=line_num,
            line_end=line_num + 1,
            column_start=match.start(),
            column_end=match.end(),
            description="Добавление комментария об оптимизации вложенных циклов",
            explanation="Вложенные циклы могут быть неэффективными, рассмотрите векторизацию",
            before_snippet=original,
            after_snippet=fixed
        )
    
    def fix_using_namespace(self, match, code, line_num) -> CodeFix:
        """Исправляет using namespace std"""
        original = match.group(0)
        fixed = "using std::cout;\nusing std::endl;"
        
        return CodeFix(
            id=f"fix_using_namespace_{line_num}",
            type=FixType.STYLE_FIX,
            confidence=FixConfidence.HIGH,
            original_code=original,
            fixed_code=fixed,
            line_start=line_num,
            line_end=line_num,
            column_start=match.start(),
            column_end=match.end(),
            description="Замена using namespace std на конкретные using",
            explanation="using namespace std может привести к конфликтам имен",
            before_snippet=original,
            after_snippet=fixed
        )
    
    def fix_new_array(self, match, code, line_num) -> CodeFix:
        """Исправляет new[] на std::vector"""
        original = match.group(0)
        type_name = match.group(1)
        size = match.group(2)
        fixed = f"std::vector<{type_name}> arr({size});"
        
        return CodeFix(
            id=f"fix_new_array_{line_num}",
            type=FixType.PERFORMANCE_FIX,
            confidence=FixConfidence.HIGH,
            original_code=original,
            fixed_code=fixed,
            line_start=line_num,
            line_end=line_num,
            column_start=match.start(),
            column_end=match.end(),
            description="Замена new[] на std::vector",
            explanation="std::vector автоматически управляет памятью",
            before_snippet=original,
            after_snippet=fixed
        )
    
    def fix_malloc_usage(self, match, code, line_num) -> CodeFix:
        """Исправляет malloc на new"""
        original = match.group(0)
        fixed = "new"
        
        return CodeFix(
            id=f"fix_malloc_{line_num}",
            type=FixType.PERFORMANCE_FIX,
            confidence=FixConfidence.HIGH,
            original_code=original,
            fixed_code=fixed,
            line_start=line_num,
            line_end=line_num,
            column_start=match.start(),
            column_end=match.end(),
            description="Замена malloc на new",
            explanation="В C++ предпочитайте new/delete или умные указатели",
            before_snippet=original,
            after_snippet=fixed
        )
    
    def fix_pmac_variables(self, match, code, line_num) -> CodeFix:
        """Исправляет PMAC переменные"""
        original = match.group(0)
        var_num = match.group(1)
        value = match.group(2)
        fixed = f"{original}  // PMAC переменная I{var_num} = {value}"
        
        return CodeFix(
            id=f"fix_pmac_var_{line_num}",
            type=FixType.STYLE_FIX,
            confidence=FixConfidence.MEDIUM,
            original_code=original,
            fixed_code=fixed,
            line_start=line_num,
            line_end=line_num,
            column_start=match.start(),
            column_end=match.end(),
            description="Добавление комментариев к PMAC переменным",
            explanation="Комментарии помогают понять назначение переменных",
            before_snippet=original,
            after_snippet=fixed
        )
    
    def fix_pmac_motion_commands(self, match, code, line_num) -> CodeFix:
        """Исправляет PMAC команды движения"""
        original = match.group(0)
        command = match.group(1)
        coordinates = match.group(2)
        fixed = f"// Проверка безопасности перед {command}\nIF (M{command.lower()}_ENABLE = 1) THEN\n    {original}\nELSE\n    // Ошибка: команда {command} отключена\n    M{command.lower()}_ERROR = 1\nENDIF"
        
        return CodeFix(
            id=f"fix_pmac_motion_{line_num}",
            type=FixType.STYLE_FIX,
            confidence=FixConfidence.MEDIUM,
            original_code=original,
            fixed_code=fixed,
            line_start=line_num,
            line_end=line_num + 5,
            column_start=match.start(),
            column_end=match.end(),
            description="Добавление проверок безопасности для команд движения",
            explanation="Команды движения должны проверяться на безопасность",
            before_snippet=original,
            after_snippet=fixed
        )
    
    def analyze_and_fix_code(self, code: str, language: str, filename: str = "") -> List[CodeFix]:
        """Анализирует код и предлагает исправления"""
        fixes = []
        lines = code.split('\n')
        
        # Получаем паттерны для языка
        patterns = self.fix_patterns.get(language, [])
        
        for pattern_info in patterns:
            pattern = re.compile(pattern_info['pattern'], re.MULTILINE | re.DOTALL)
            matches = pattern.finditer(code)
            
            for match in matches:
                line_num = code[:match.start()].count('\n') + 1
                
                # Вызываем функцию исправления
                fix_function = pattern_info['fix_function']
                fix = fix_function(match, code, line_num)
                fixes.append(fix)
        
        return fixes
    
    def apply_fixes(self, code: str, fixes: List[CodeFix]) -> Tuple[str, List[CodeFix]]:
        """Применяет исправления к коду"""
        if not fixes:
            return code, []
        
        # Сортируем исправления по позиции (с конца к началу)
        fixes.sort(key=lambda f: f.line_start, reverse=True)
        
        lines = code.split('\n')
        applied_fixes = []
        
        for fix in fixes:
            try:
                # Применяем исправление
                if fix.line_start == fix.line_end:
                    # Исправление в одной строке
                    line_idx = fix.line_start - 1
                    if 0 <= line_idx < len(lines):
                        original_line = lines[line_idx]
                        # Простая замена для демонстрации
                        fixed_line = original_line.replace(fix.original_code, fix.fixed_code)
                        lines[line_idx] = fixed_line
                        applied_fixes.append(fix)
                else:
                    # Исправление в нескольких строках
                    # Для сложных исправлений просто добавляем комментарий
                    line_idx = fix.line_start - 1
                    if 0 <= line_idx < len(lines):
                        lines[line_idx] += f"  // {fix.description}"
                        applied_fixes.append(fix)
                
            except Exception as e:
                print(f"Ошибка применения исправления {fix.id}: {e}")
                continue
        
        fixed_code = '\n'.join(lines)
        return fixed_code, applied_fixes
    
    def generate_code(self, request: CodeGenerationRequest) -> str:
        """Генерирует код на основе запроса"""
        templates = self.code_templates.get(request.language, {})
        
        if request.language == 'python':
            if 'function' in request.description.lower():
                return self._generate_python_function(request, templates)
            elif 'class' in request.description.lower():
                return self._generate_python_class(request, templates)
            else:
                return self._generate_python_general(request, templates)
        
        elif request.language == 'cpp':
            if 'class' in request.description.lower():
                return self._generate_cpp_class(request, templates)
            else:
                return self._generate_cpp_function(request, templates)
        
        elif request.language == 'pmac':
            if 'program' in request.description.lower():
                return self._generate_pmac_program(request, templates)
            elif 'motion' in request.description.lower():
                return self._generate_pmac_motion(request, templates)
            else:
                return self._generate_pmac_general(request, templates)
        
        return f"// Сгенерированный код для: {request.description}\n// TODO: Реализовать функциональность"
    
    def _generate_python_function(self, request: CodeGenerationRequest, templates: Dict) -> str:
        """Генерирует Python функцию"""
        template = templates.get('function_template', 'def {function_name}():\n    pass')
        
        # Простая генерация на основе описания
        function_name = "generated_function"
        if "функция" in request.description.lower():
            function_name = "my_function"
        
        return template.format(
            function_name=function_name,
            parameters="self" if "метод" in request.description.lower() else "",
            description=request.description,
            args_doc="args: аргументы функции",
            return_doc="return: результат выполнения",
            body="    # TODO: Реализовать логику\n    pass"
        )
    
    def _generate_python_class(self, request: CodeGenerationRequest, templates: Dict) -> str:
        """Генерирует Python класс"""
        template = templates.get('class_template', 'class {class_name}:\n    pass')
        
        class_name = "GeneratedClass"
        if "класс" in request.description.lower():
            class_name = "MyClass"
        
        return template.format(
            class_name=class_name,
            description=request.description,
            init_params="",
            init_body="        # TODO: Инициализация",
            method_name="my_method",
            method_params="",
            method_description="Описание метода",
            method_body="        # TODO: Реализация метода\n        pass"
        )
    
    def _generate_cpp_class(self, request: CodeGenerationRequest, templates: Dict) -> str:
        """Генерирует C++ класс"""
        template = templates.get('class_template', 'class {class_name} {};')
        
        class_name = "GeneratedClass"
        if "класс" in request.description.lower():
            class_name = "MyClass"
        
        return template.format(
            class_name=class_name,
            constructor_params="",
            methods="    void method();",
            private_members="    int value_;"
        )
    
    def _generate_cpp_function(self, request: CodeGenerationRequest, templates: Dict) -> str:
        """Генерирует C++ функцию"""
        template = templates.get('function_template', '{return_type} {function_name}() {}')
        
        return template.format(
            return_type="void",
            function_name="generated_function",
            parameters="",
            body="    // TODO: Реализовать логику"
        )
    
    def _generate_pmac_program(self, request: CodeGenerationRequest, templates: Dict) -> str:
        """Генерирует PMAC программу"""
        template = templates.get('program_template', 'OPEN PROG 1\nCLEAR\n// {description}\nCLOSE')
        
        return template.format(
            program_number="1",
            description=request.description,
            variables="I100 = 1000  // Переменная 1\nI101 = 2000  // Переменная 2",
            logic="// Основная логика программы\n// TODO: Реализовать алгоритм"
        )
    
    def _generate_pmac_motion(self, request: CodeGenerationRequest, templates: Dict) -> str:
        """Генерирует PMAC команду движения"""
        template = templates.get('motion_template', '// {description}\n{motion_command}')
        
        return template.format(
            description=request.description,
            safety_condition="M1 = 1",
            motion_command="ABS X100 Y200",
            error_flag="100"
        )
    
    def _generate_pmac_general(self, request: CodeGenerationRequest, templates: Dict) -> str:
        """Генерирует общий PMAC код"""
        return f"""// Сгенерированный PMAC код
// Описание: {request.description}

// Инициализация
I100 = 1000

// Основная логика
// TODO: Реализовать функциональность

// Завершение"""
    
    def _generate_python_general(self, request: CodeGenerationRequest, templates: Dict) -> str:
        """Генерирует общий Python код"""
        return f'''"""
Сгенерированный Python код
Описание: {request.description}
"""

def main():
    """Основная функция"""
    # TODO: Реализовать логику
    pass

if __name__ == "__main__":
    main()'''
    
    def save_fix_to_database(self, fix: CodeFix):
        """Сохраняет исправление в базу данных"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO code_fixes 
                (id, fix_type, confidence, original_code, fixed_code, line_start, line_end, 
                 description, explanation, success)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                fix.id,
                fix.type.value,
                fix.confidence.value,
                fix.original_code,
                fix.fixed_code,
                fix.line_start,
                fix.line_end,
                fix.description,
                fix.explanation,
                True
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Ошибка сохранения исправления: {e}")
    
    def save_generation_to_database(self, request: CodeGenerationRequest):
        """Сохраняет генерацию кода в базу данных"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO code_generations 
                (id, description, language, context, requirements, generated_code, confidence, success)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                request.id,
                request.description,
                request.language,
                request.context,
                json.dumps(request.requirements),
                request.generated_code,
                request.confidence.value,
                True
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Ошибка сохранения генерации: {e}")

def main():
    """Основная функция для тестирования"""
    print("🔧 SMART RUBIN AI - АВТОМАТИЧЕСКОЕ ИСПРАВЛЕНИЕ КОДА")
    print("=" * 60)
    
    # Создаем систему исправления
    fixer = RubinCodeAutoFixer()
    
    # Тестовый код с ошибками
    test_code = '''import *
import os

def bad_function():
    try:
        result = eval(input("Enter code: "))
        return result
    except:
        pass

def inefficient_loop():
    text = ""
    for i in range(1000):
        text += str(i)
    return text

# SQL инъекция
def unsafe_query(user_input):
    query = "SELECT * FROM users WHERE name = '" + user_input + "'"
    return query'''
    
    print("📝 Анализируем код с ошибками...")
    print("-" * 40)
    
    # Анализируем и исправляем
    fixes = fixer.analyze_and_fix_code(test_code, 'python', 'test.py')
    
    print(f"🔍 Найдено {len(fixes)} исправлений:")
    for fix in fixes:
        print(f"   {fix.type.value.upper()}: {fix.description}")
        print(f"      Уверенность: {fix.confidence.value}")
        print(f"      Строка {fix.line_start}: {fix.before_snippet} → {fix.after_snippet}")
        print()
    
    # Применяем исправления
    if fixes:
        fixed_code, applied_fixes = fixer.apply_fixes(test_code, fixes)
        print(f"✅ Применено {len(applied_fixes)} исправлений")
        print("\n📄 Исправленный код:")
        print("-" * 40)
        print(fixed_code)
        
        # Сохраняем исправления
        for fix in applied_fixes:
            fixer.save_fix_to_database(fix)
    
    # Тестируем генерацию кода
    print("\n🎯 Тестируем генерацию кода...")
    print("-" * 40)
    
    generation_requests = [
        CodeGenerationRequest(
            id="gen_1",
            description="Создать функцию для работы с файлами",
            language="python",
            context="Обработка текстовых файлов",
            requirements=["Чтение файла", "Обработка данных", "Сохранение результата"],
            examples=[]
        ),
        CodeGenerationRequest(
            id="gen_2",
            description="Создать класс для управления PMAC контроллером",
            language="pmac",
            context="Программирование движения",
            requirements=["Инициализация", "Команды движения", "Проверка безопасности"],
            examples=[]
        )
    ]
    
    for request in generation_requests:
        print(f"\n📝 Генерация: {request.description}")
        generated_code = fixer.generate_code(request)
        request.generated_code = generated_code
        
        print("Сгенерированный код:")
        print("-" * 20)
        print(generated_code)
        
        # Сохраняем генерацию
        fixer.save_generation_to_database(request)
    
    print("\n🎉 ТЕСТИРОВАНИЕ ЗАВЕРШЕНО!")

if __name__ == "__main__":
    main()
