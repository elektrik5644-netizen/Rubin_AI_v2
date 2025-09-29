#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Визуализатор аналитики нейронной сети Rubin AI
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np
import json
import requests
from typing import Dict, Any, List
import seaborn as sns
from collections import Counter
import pandas as pd

# Настройка стиля графиков
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class NeuralAnalyticsVisualizer:
    """Класс для визуализации аналитических данных нейронной сети"""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.analytics_data = None
    
    def fetch_analytics(self) -> bool:
        """Получение аналитических данных с сервера"""
        try:
            response = requests.get(f"{self.base_url}/api/analytics")
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    self.analytics_data = data['analytics']
                    return True
                else:
                    print(f"Ошибка получения аналитики: {data.get('error')}")
                    return False
            else:
                print(f"HTTP ошибка: {response.status_code}")
                return False
        except Exception as e:
            print(f"Ошибка соединения: {e}")
            return False
    
    def create_performance_dashboard(self) -> None:
        """Создание дашборда производительности"""
        if not self.analytics_data:
            print("Нет данных для визуализации")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('📊 Дашборд производительности нейронной сети Rubin AI', fontsize=16, fontweight='bold')
        
        # 1. Общая статистика
        summary = self.analytics_data['summary']
        ax1 = axes[0, 0]
        
        stats_labels = ['Всего запросов', 'Успешных', 'Неудачных', 'Время работы (ч)']
        stats_values = [
            summary['total_requests'],
            summary['successful_requests'],
            summary['failed_requests'],
            summary['uptime_seconds'] / 3600
        ]
        
        bars = ax1.bar(stats_labels, stats_values, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
        ax1.set_title('📈 Общая статистика')
        ax1.set_ylabel('Количество')
        
        # Добавляем значения на столбцы
        for bar, value in zip(bars, stats_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{value:.1f}', ha='center', va='bottom')
        
        # 2. Использование категорий
        ax2 = axes[0, 1]
        categories = self.analytics_data['categories']['usage']
        
        if categories:
            labels = list(categories.keys())
            values = list(categories.values())
            
            # Создаем круговую диаграмму
            wedges, texts, autotexts = ax2.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
            ax2.set_title('🎯 Использование категорий')
            
            # Улучшаем читаемость
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        
        # 3. Время обработки
        ax3 = axes[1, 0]
        performance = self.analytics_data['performance']
        
        time_labels = ['Мин', 'Макс', 'Среднее', 'Медиана']
        time_values = [
            performance['min'],
            performance['max'],
            performance['mean'],
            performance['median']
        ]
        
        bars = ax3.bar(time_labels, time_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax3.set_title('⏱️ Время обработки (сек)')
        ax3.set_ylabel('Секунды')
        
        for bar, value in zip(bars, time_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # 4. Уверенность по категориям
        ax4 = axes[1, 1]
        confidence = self.analytics_data['categories']['avg_confidence']
        
        if confidence:
            cat_labels = list(confidence.keys())
            conf_values = list(confidence.values())
            
            bars = ax4.bar(cat_labels, conf_values, color='#9B59B6')
            ax4.set_title('🎯 Средняя уверенность по категориям')
            ax4.set_ylabel('Уверенность')
            ax4.set_ylim(0, 1)
            
            # Поворачиваем подписи для лучшей читаемости
            ax4.tick_params(axis='x', rotation=45)
            
            for bar, value in zip(bars, conf_values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def create_temporal_analysis(self) -> None:
        """Анализ временных паттернов"""
        if not self.analytics_data:
            print("Нет данных для визуализации")
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle('📅 Временной анализ использования нейронной сети', fontsize=16, fontweight='bold')
        
        # 1. Запросы по часам
        ax1 = axes[0]
        hourly_data = self.analytics_data['requests']['hourly']
        
        if hourly_data:
            # Преобразуем в DataFrame для удобства
            hours = []
            counts = []
            
            for hour_str, count in hourly_data.items():
                try:
                    hour_dt = datetime.strptime(hour_str, '%Y-%m-%d %H:%M')
                    hours.append(hour_dt)
                    counts.append(count)
                except ValueError:
                    continue
            
            if hours:
                ax1.plot(hours, counts, marker='o', linewidth=2, markersize=6, color='#E74C3C')
                ax1.set_title('📈 Запросы по часам')
                ax1.set_ylabel('Количество запросов')
                ax1.grid(True, alpha=0.3)
                
                # Форматируем ось X
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
                ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))
                plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # 2. Запросы по дням
        ax2 = axes[1]
        daily_data = self.analytics_data['requests']['daily']
        
        if daily_data:
            days = []
            counts = []
            
            for day_str, count in daily_data.items():
                try:
                    day_dt = datetime.strptime(day_str, '%Y-%m-%d')
                    days.append(day_dt)
                    counts.append(count)
                except ValueError:
                    continue
            
            if days:
                bars = ax2.bar(days, counts, color='#3498DB', alpha=0.7)
                ax2.set_title('📊 Запросы по дням')
                ax2.set_ylabel('Количество запросов')
                ax2.grid(True, alpha=0.3)
                
                # Форматируем ось X
                ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
                ax2.xaxis.set_major_locator(mdates.DayLocator())
                plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def create_error_analysis(self) -> None:
        """Анализ ошибок"""
        if not self.analytics_data:
            print("Нет данных для визуализации")
            return
        
        errors = self.analytics_data['errors']
        
        if not errors['types']:
            print("Нет данных об ошибках для анализа")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('🚨 Анализ ошибок нейронной сети', fontsize=16, fontweight='bold')
        
        # 1. Типы ошибок
        ax1 = axes[0]
        error_types = errors['types']
        
        if error_types:
            labels = list(error_types.keys())
            values = list(error_types.values())
            
            bars = ax1.bar(labels, values, color='#E74C3C')
            ax1.set_title('📊 Типы ошибок')
            ax1.set_ylabel('Количество')
            ax1.tick_params(axis='x', rotation=45)
            
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        str(value), ha='center', va='bottom')
        
        # 2. Временная линия ошибок
        ax2 = axes[1]
        recent_errors = errors['recent_errors']
        
        if recent_errors:
            timestamps = []
            error_counts = []
            
            for error in recent_errors:
                try:
                    ts = datetime.fromisoformat(error['timestamp'])
                    timestamps.append(ts)
                    error_counts.append(1)
                except ValueError:
                    continue
            
            if timestamps:
                ax2.scatter(timestamps, error_counts, color='#E74C3C', s=100, alpha=0.7)
                ax2.set_title('⏰ Временная линия ошибок')
                ax2.set_ylabel('Ошибки')
                ax2.set_ylim(0, 2)
                
                # Форматируем ось X
                ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def create_training_analysis(self) -> None:
        """Анализ обучения"""
        if not self.analytics_data:
            print("Нет данных для визуализации")
            return
        
        training = self.analytics_data['training']
        
        if training['sessions'] == 0:
            print("Нет данных об обучении для анализа")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('🎓 Анализ обучения нейронной сети', fontsize=16, fontweight='bold')
        
        # 1. Статистика обучения
        ax1 = axes[0]
        training_stats = ['Сессии', 'Точек данных']
        training_values = [training['sessions'], training['data_points']]
        
        bars = ax1.bar(training_stats, training_values, color=['#27AE60', '#F39C12'])
        ax1.set_title('📚 Статистика обучения')
        ax1.set_ylabel('Количество')
        
        for bar, value in zip(bars, training_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    str(value), ha='center', va='bottom')
        
        # 2. Улучшения модели
        ax2 = axes[1]
        improvements = training['recent_improvements']
        
        if improvements:
            timestamps = []
            improvement_values = []
            
            for improvement in improvements:
                try:
                    ts = datetime.fromisoformat(improvement['timestamp'])
                    timestamps.append(ts)
                    improvement_values.append(improvement['improvement'])
                except ValueError:
                    continue
            
            if timestamps:
                ax2.plot(timestamps, improvement_values, marker='o', linewidth=2, color='#27AE60')
                ax2.set_title('📈 Улучшения модели')
                ax2.set_ylabel('Улучшение')
                ax2.grid(True, alpha=0.3)
                
                # Форматируем ось X
                ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def create_full_report(self) -> None:
        """Создание полного отчета"""
        print("🔄 Получение аналитических данных...")
        
        if not self.fetch_analytics():
            print("❌ Не удалось получить данные аналитики")
            return
        
        print("📊 Создание дашборда производительности...")
        self.create_performance_dashboard()
        
        print("📅 Создание временного анализа...")
        self.create_temporal_analysis()
        
        print("🚨 Создание анализа ошибок...")
        self.create_error_analysis()
        
        print("🎓 Создание анализа обучения...")
        self.create_training_analysis()
        
        print("✅ Полный отчет создан!")

def main():
    """Основная функция"""
    print("🚀 Запуск визуализатора аналитики нейронной сети Rubin AI")
    
    visualizer = NeuralAnalyticsVisualizer()
    
    # Создаем полный отчет
    visualizer.create_full_report()

if __name__ == "__main__":
    main()


