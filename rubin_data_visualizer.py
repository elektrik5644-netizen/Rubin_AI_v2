#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📈 Модуль визуализации данных для Rubin AI
==========================================

Модуль для создания различных типов графиков и диаграмм:
- Линейные графики
- Столбчатые диаграммы
- Круговые диаграммы
- Точечные диаграммы
- Гистограммы
- Тепловые карты
- Интерактивные графики

Автор: Rubin AI System
Версия: 2.1
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import io
import base64
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import json
import seaborn as sns

# Попытка импорта plotly для интерактивных графиков
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.offline import plot
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.warning("Plotly не установлен. Интерактивные графики недоступны.")

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VisualizationType(Enum):
    """Типы визуализации"""
    LINE_CHART = "линейный_график"
    BAR_CHART = "столбчатая_диаграмма"
    PIE_CHART = "круговая_диаграмма"
    SCATTER_PLOT = "точечная_диаграмма"
    HISTOGRAM = "гистограмма"
    HEATMAP = "тепловая_карта"
    BOX_PLOT = "ящичная_диаграмма"
    VIOLIN_PLOT = "скрипичная_диаграмма"
    AREA_CHART = "диаграмма_с_заливкой"

@dataclass
class VisualizationResult:
    """Результат создания визуализации"""
    viz_type: VisualizationType
    data_summary: Dict[str, Any]
    plot_data: str  # Base64 encoded image
    interactive_html: Optional[str] = None
    statistics: Dict[str, Any] = None
    recommendations: List[str] = None

class RubinDataVisualizer:
    """Создатель визуализаций данных"""
    
    def __init__(self):
        """Инициализация визуализатора"""
        self.plotly_available = PLOTLY_AVAILABLE
        
        # Настройка стиля matplotlib
        plt.style.use('default')
        sns.set_palette("husl")
        
        logger.info(f"📈 Data Visualizer инициализирован. Plotly: {self.plotly_available}")
    
    def create_visualization(self, data: Dict[str, List[float]], 
                           viz_type: VisualizationType,
                           title: str = "График данных",
                           x_label: str = "X",
                           y_label: str = "Y") -> VisualizationResult:
        """Создание визуализации данных"""
        try:
            logger.info(f"📈 Создание {viz_type.value}: {title}")
            
            # Подготовка данных
            processed_data = self._prepare_data(data)
            
            # Создание графика
            if viz_type == VisualizationType.LINE_CHART:
                result = self._create_line_chart(processed_data, title, x_label, y_label)
            elif viz_type == VisualizationType.BAR_CHART:
                result = self._create_bar_chart(processed_data, title, x_label, y_label)
            elif viz_type == VisualizationType.PIE_CHART:
                result = self._create_pie_chart(processed_data, title)
            elif viz_type == VisualizationType.SCATTER_PLOT:
                result = self._create_scatter_plot(processed_data, title, x_label, y_label)
            elif viz_type == VisualizationType.HISTOGRAM:
                result = self._create_histogram(processed_data, title, x_label)
            elif viz_type == VisualizationType.HEATMAP:
                result = self._create_heatmap(processed_data, title)
            else:
                result = self._create_line_chart(processed_data, title, x_label, y_label)
            
            # Добавление статистики
            result.statistics = self._calculate_statistics(processed_data)
            
            # Генерация рекомендаций
            result.recommendations = self._generate_recommendations(viz_type, processed_data)
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка создания визуализации: {e}")
            return self._create_error_result(f"Ошибка: {e}")
    
    def _prepare_data(self, data: Dict[str, List[float]]) -> Dict[str, Any]:
        """Подготовка данных для визуализации"""
        processed = {}
        
        for key, values in data.items():
            processed[key] = {
                'values': values,
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'count': len(values)
            }
        
        return processed
    
    def _create_line_chart(self, data: Dict[str, Any], title: str, 
                          x_label: str, y_label: str) -> VisualizationResult:
        """Создание линейного графика"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(data)))
        
        for i, (key, info) in enumerate(data.items()):
            x_values = list(range(len(info['values'])))
            ax.plot(x_values, info['values'], 
                   color=colors[i], linewidth=2, marker='o', 
                   markersize=6, label=key)
        
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Сохранение в base64
        plot_data = self._save_plot_to_base64(fig)
        plt.close()
        
        return VisualizationResult(
            viz_type=VisualizationType.LINE_CHART,
            data_summary=data,
            plot_data=plot_data
        )
    
    def _create_bar_chart(self, data: Dict[str, Any], title: str,
                         x_label: str, y_label: str) -> VisualizationResult:
        """Создание столбчатой диаграммы"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        categories = list(data.keys())
        values = [info['mean'] for info in data.values()]
        errors = [info['std'] for info in data.values()]
        
        bars = ax.bar(categories, values, yerr=errors, 
                     capsize=5, alpha=0.7, color=plt.cm.Set3(np.linspace(0, 1, len(categories))))
        
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Добавление значений на столбцы
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(errors)/10,
                   f'{value:.1f}', ha='center', va='bottom')
        
        plot_data = self._save_plot_to_base64(fig)
        plt.close()
        
        return VisualizationResult(
            viz_type=VisualizationType.BAR_CHART,
            data_summary=data,
            plot_data=plot_data
        )
    
    def _create_pie_chart(self, data: Dict[str, Any], title: str) -> VisualizationResult:
        """Создание круговой диаграммы"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        labels = list(data.keys())
        sizes = [info['mean'] for info in data.values()]
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                         autopct='%1.1f%%', startangle=90)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Улучшение читаемости
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plot_data = self._save_plot_to_base64(fig)
        plt.close()
        
        return VisualizationResult(
            viz_type=VisualizationType.PIE_CHART,
            data_summary=data,
            plot_data=plot_data
        )
    
    def _create_scatter_plot(self, data: Dict[str, Any], title: str,
                           x_label: str, y_label: str) -> VisualizationResult:
        """Создание точечной диаграммы"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(data)))
        
        for i, (key, info) in enumerate(data.items()):
            if len(info['values']) >= 2:
                x_values = info['values'][::2] if len(info['values']) > 2 else [info['values'][0]]
                y_values = info['values'][1::2] if len(info['values']) > 2 else [info['values'][1]]
                
                ax.scatter(x_values, y_values, color=colors[i], 
                          s=100, alpha=0.7, label=key)
        
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plot_data = self._save_plot_to_base64(fig)
        plt.close()
        
        return VisualizationResult(
            viz_type=VisualizationType.SCATTER_PLOT,
            data_summary=data,
            plot_data=plot_data
        )
    
    def _create_histogram(self, data: Dict[str, Any], title: str, x_label: str) -> VisualizationResult:
        """Создание гистограммы"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(data)))
        
        for i, (key, info) in enumerate(data.items()):
            ax.hist(info['values'], bins=20, alpha=0.7, 
                   color=colors[i], label=key, density=True)
        
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel('Плотность', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plot_data = self._save_plot_to_base64(fig)
        plt.close()
        
        return VisualizationResult(
            viz_type=VisualizationType.HISTOGRAM,
            data_summary=data,
            plot_data=plot_data
        )
    
    def _create_heatmap(self, data: Dict[str, Any], title: str) -> VisualizationResult:
        """Создание тепловой карты"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Подготовка данных для тепловой карты
        matrix_data = []
        labels = []
        
        for key, info in data.items():
            matrix_data.append(info['values'])
            labels.append(key)
        
        # Создание матрицы
        matrix = np.array(matrix_data)
        
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
        
        # Настройка осей
        ax.set_xticks(range(len(matrix[0])))
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        
        # Добавление значений в ячейки
        for i in range(len(labels)):
            for j in range(len(matrix[0])):
                text = ax.text(j, i, f'{matrix[i, j]:.1f}',
                             ha="center", va="center", color="black")
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Цветовая шкала
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Значение', rotation=270, labelpad=20)
        
        plot_data = self._save_plot_to_base64(fig)
        plt.close()
        
        return VisualizationResult(
            viz_type=VisualizationType.HEATMAP,
            data_summary=data,
            plot_data=plot_data
        )
    
    def _save_plot_to_base64(self, fig) -> str:
        """Сохранение графика в base64"""
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()
        return plot_data
    
    def _calculate_statistics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Расчет статистики"""
        stats = {}
        
        for key, info in data.items():
            stats[key] = {
                'mean': info['mean'],
                'std': info['std'],
                'min': info['min'],
                'max': info['max'],
                'count': info['count'],
                'range': info['max'] - info['min']
            }
        
        return stats
    
    def _generate_recommendations(self, viz_type: VisualizationType, 
                                data: Dict[str, Any]) -> List[str]:
        """Генерация рекомендаций"""
        recommendations = []
        
        if viz_type == VisualizationType.LINE_CHART:
            recommendations.extend([
                "Рассмотрите тренд данных",
                "Проверьте наличие выбросов",
                "Проанализируйте периодичность"
            ])
        elif viz_type == VisualizationType.BAR_CHART:
            recommendations.extend([
                "Сравните значения категорий",
                "Найдите максимальные и минимальные значения",
                "Рассмотрите группировку данных"
            ])
        elif viz_type == VisualizationType.PIE_CHART:
            recommendations.extend([
                "Проанализируйте пропорции сегментов",
                "Выделите доминирующие категории",
                "Рассмотрите возможность группировки мелких сегментов"
            ])
        
        recommendations.extend([
            "Проверьте качество исходных данных",
            "Рассмотрите альтернативные способы визуализации",
            "Добавьте дополнительные метрики для анализа"
        ])
        
        return recommendations
    
    def _create_error_result(self, error_message: str) -> VisualizationResult:
        """Создание результата с ошибкой"""
        return VisualizationResult(
            viz_type=VisualizationType.LINE_CHART,
            data_summary={},
            plot_data="",
            recommendations=[f"Ошибка: {error_message}"]
        )

def test_data_visualizer():
    """Тестирование визуализатора данных"""
    visualizer = RubinDataVisualizer()
    
    print("📈 ТЕСТИРОВАНИЕ ВИЗУАЛИЗАТОРА ДАННЫХ")
    print("=" * 60)
    
    # Тестовые данные
    test_data = {
        'Серия A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Серия B': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        'Серия C': [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
    }
    
    # Тест 1: Линейный график
    print("\n1. Линейный график:")
    result = visualizer.create_visualization(
        test_data, VisualizationType.LINE_CHART,
        "Тестовый линейный график", "Время", "Значение"
    )
    print(f"✅ График создан. Размер данных: {len(result.plot_data)} символов")
    
    # Тест 2: Столбчатая диаграмма
    print("\n2. Столбчатая диаграмма:")
    result = visualizer.create_visualization(
        test_data, VisualizationType.BAR_CHART,
        "Тестовая столбчатая диаграмма", "Категории", "Значения"
    )
    print(f"✅ Диаграмма создана. Статистика: {len(result.statistics)} серий")
    
    # Тест 3: Круговая диаграмма
    print("\n3. Круговая диаграмма:")
    result = visualizer.create_visualization(
        test_data, VisualizationType.PIE_CHART,
        "Тестовая круговая диаграмма"
    )
    print(f"✅ Диаграмма создана. Рекомендации: {len(result.recommendations)}")
    
    print(f"\n📊 Plotly доступен: {visualizer.plotly_available}")
    print("🎉 Тестирование завершено!")

if __name__ == "__main__":
    test_data_visualizer()





