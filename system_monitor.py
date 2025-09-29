#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль мониторинга системы Rubin AI v2
"""

import json
import time
import logging
import psutil
import requests
from datetime import datetime
from typing import Dict, List, Any
import os
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class RubinSystemMonitor:
    """Класс для мониторинга системы Rubin AI"""
    
    def __init__(self, base_url="http://127.0.0.1:8081"):
        self.base_url = base_url
        self.metrics_buffer = deque(maxlen=10000)
        self.alerts = []
        self.monitoring_active = False
        
    def collect_system_metrics(self):
        """Сбор системных метрик"""
        try:
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "system": {
                    "cpu_percent": psutil.cpu_percent(interval=1),
                    "memory_percent": psutil.virtual_memory().percent,
                    "disk_percent": psutil.disk_usage('/').percent
                }
            }
            
            self.metrics_buffer.append(metrics)
            return metrics
            
        except Exception as e:
            logger.error(f"Ошибка сбора метрик: {e}")
            return {"error": str(e)}
    
    def collect_api_metrics(self):
        """Сбор метрик API"""
        try:
            api_metrics = {
                "timestamp": datetime.now().isoformat(),
                "api": {}
            }
            
            # Проверяем здоровье API
            try:
                start_time = time.time()
                response = requests.get(f"{self.base_url}/api/health", timeout=5)
                response_time = time.time() - start_time
                
                api_metrics["api"]["health"] = {
                    "status": response.status_code == 200,
                    "response_time": response_time
                }
            except Exception as e:
                api_metrics["api"]["health"] = {
                    "status": False,
                    "error": str(e)
                }
            
            return api_metrics
            
        except Exception as e:
            logger.error(f"Ошибка сбора API метрик: {e}")
            return {"error": str(e)}
    
    def check_alerts(self, metrics):
        """Проверка алертов"""
        try:
            alerts = []
            
            if "system" in metrics:
                system = metrics["system"]
                
                if system.get("cpu_percent", 0) > 80:
                    alerts.append({
                        "type": "warning",
                        "metric": "cpu",
                        "value": system["cpu_percent"],
                        "message": "Высокая загрузка CPU"
                    })
                
                if system.get("memory_percent", 0) > 80:
                    alerts.append({
                        "type": "warning", 
                        "metric": "memory",
                        "value": system["memory_percent"],
                        "message": "Высокая загрузка памяти"
                    })
            
            self.alerts.extend(alerts)
            return alerts
            
        except Exception as e:
            logger.error(f"Ошибка проверки алертов: {e}")
            return []
    
    def generate_monitoring_report(self):
        """Генерация отчета мониторинга"""
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "summary": {},
                "system_metrics": {},
                "api_metrics": {},
                "alerts_summary": {},
                "recommendations": []
            }
            
            # Собираем метрики
            system_metrics = self.collect_system_metrics()
            api_metrics = self.collect_api_metrics()
            
            # Проверяем алерты
            all_metrics = {**system_metrics, **api_metrics}
            alerts = self.check_alerts(all_metrics)
            
            # Создаем сводку
            report["summary"] = {
                "total_metrics_collected": len(self.metrics_buffer),
                "active_alerts": len([a for a in self.alerts if a["type"] == "warning"]),
                "system_status": "healthy" if len(alerts) == 0 else "warning"
            }
            
            # Детальные метрики
            report["system_metrics"] = system_metrics
            report["api_metrics"] = api_metrics
            
            # Сводка алертов
            alert_summary = defaultdict(int)
            for alert in self.alerts:
                alert_summary[alert["type"]] += 1
            report["alerts_summary"] = dict(alert_summary)
            
            # Рекомендации
            recommendations = []
            if system_metrics.get("system", {}).get("cpu_percent", 0) > 80:
                recommendations.append("Оптимизировать CPU нагрузку")
            if system_metrics.get("system", {}).get("memory_percent", 0) > 80:
                recommendations.append("Оптимизировать использование памяти")
            if not api_metrics.get("api", {}).get("health", {}).get("status", False):
                recommendations.append("Исправить проблемы с API")
            
            report["recommendations"] = recommendations
            
            return report
            
        except Exception as e:
            logger.error(f"Ошибка генерации отчета: {e}")
            return {"error": str(e)}

def main():
    """Основная функция для тестирования мониторинга"""
    
    print("МОНИТОРИНГ СИСТЕМЫ RUBIN AI v2")
    print("=" * 50)
    
    # Создаем экземпляр
    monitor = RubinSystemMonitor()
    
    # Генерируем отчет
    print("\nГенерация отчета мониторинга...")
    report = monitor.generate_monitoring_report()
    
    # Сохраняем отчет
    with open('monitoring_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"Отчет сохранен в: monitoring_report.json")
    
    # Показываем сводку
    if "summary" in report:
        summary = report["summary"]
        print(f"\nСВОДКА МОНИТОРИНГА:")
        print(f"  Статус системы: {summary.get('system_status', 'unknown')}")
        print(f"  Собрано метрик: {summary.get('total_metrics_collected', 0)}")
        print(f"  Активных алертов: {summary.get('active_alerts', 0)}")
    
    # Рекомендации
    if "recommendations" in report:
        recommendations = report["recommendations"]
        if recommendations:
            print(f"\nРЕКОМЕНДАЦИИ:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        else:
            print(f"\nСистема работает стабильно")
    
    return report

if __name__ == "__main__":
    main()