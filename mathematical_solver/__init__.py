#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mathematical Solver Module
==========================

Модуль для решения математических задач в системе Rubin AI.
"""

from .category_detector import MathematicalCategoryDetector
from .request_handler import MathematicalRequestHandler
from .response_formatter import MathematicalResponseFormatter
from .error_handler import MathematicalErrorHandler

__all__ = [
    'MathematicalCategoryDetector',
    'MathematicalRequestHandler', 
    'MathematicalResponseFormatter',
    'MathematicalErrorHandler'
]

__version__ = '1.0.0'

















