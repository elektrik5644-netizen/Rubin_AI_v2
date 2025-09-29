#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Система мониторинга и автоматических обновлений Rubin AI
"""

import sqlite3
import json
import time
import threading
import schedule
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging
import os
import sys

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('system_monitor.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SystemMonitor:
    """Класс для мониторинга системы Rubin AI"""
    
    def __init__(self, db_path: str = "rubin_ai_v2.db"):
        self.db_path = db_path
        self.monitoring_active = False
        self.stats = {
            'last_check': None,
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0,
            'database_size': 0,
            'document_count': 0,
            'synonym_count': 0,
            'system_health': 'unknown'
        }
    
    def check_database_health(self) -> Dict[str, Any]:
        """Проверка здоровья базы данных"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Проверяем размер базы данных
            db_size = os.path.getsize(self.db_path)
            
            # Подсчитываем документы
            cursor.execute("SELECT COUNT(*) FROM documents")
            doc_count = cursor.fetchone()[0]
            
            # Подсчитываем синонимы
            cursor.execute("SELECT COUNT(*) FROM technical_synonyms")
            synonym_count = cursor.fetchone()[0]
            
            # Проверяем целостность
            cursor.execute("PRAGMA integrity_check")
            integrity = cursor.fetchone()[0]
            
            conn.close()
            
            health_status = {
                'database_size_mb': round(db_size / (1024 * 1024), 2),
                'document_count': doc_count,
                'synonym_count': synonym_count,
                'integrity_check': integrity,
                'status': 'healthy' if integrity == 'ok' else 'warning',
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"✅ Проверка БД: {doc_count} документов, {synonym_count} синонимов")
            return health_status
            
        except Exception as e:
            logger.error(f"❌ Ошибка проверки БД: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def check_api_health(self) -> Dict[str, Any]:
        """Проверка здоровья API"""
        try:
            # Проверяем основной API
            response = requests.get('http://localhost:8084/health', timeout=5)
            main_api_status = response.status_code == 200
            
            # Проверяем API документов
            try:
                response = requests.get('http://localhost:8088/health', timeout=5)
                docs_api_status = response.status_code == 200
            except:
                docs_api_status = False
            
            # Проверяем API словаря
            try:
                response = requests.get('http://localhost:8085/api/vocabulary/stats', timeout=5)
                vocab_api_status = response.status_code == 200
            except:
                vocab_api_status = False
            
            api_health = {
                'main_api': main_api_status,
                'docs_api': docs_api_status,
                'vocab_api': vocab_api_status,
                'overall_status': 'healthy' if main_api_status else 'degraded',
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"🔍 API статус: Основной={main_api_status}, Документы={docs_api_status}, Словарь={vocab_api_status}")
            return api_health
            
        except Exception as e:
            logger.error(f"❌ Ошибка проверки API: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def check_search_performance(self) -> Dict[str, Any]:
        """Проверка производительности поиска"""
        try:
            # Тестируем поиск с разными запросами
            test_queries = [
                "ПИД регулятор",
                "автоматизация",
                "электротехника",
                "программирование"
            ]
            
            performance_data = []
            
            for query in test_queries:
                start_time = time.time()
                try:
                    response = requests.post(
                        'http://localhost:8084/api/chat',
                        json={'message': query},
                        timeout=10
                    )
                    response_time = time.time() - start_time
                    
                    performance_data.append({
                        'query': query,
                        'response_time': round(response_time, 3),
                        'status': 'success' if response.status_code == 200 else 'failed',
                        'status_code': response.status_code
                    })
                    
                except Exception as e:
                    performance_data.append({
                        'query': query,
                        'response_time': -1,
                        'status': 'error',
                        'error': str(e)
                    })
            
            avg_response_time = sum([p['response_time'] for p in performance_data if p['response_time'] > 0]) / len([p for p in performance_data if p['response_time'] > 0])
            
            performance = {
                'test_queries': performance_data,
                'average_response_time': round(avg_response_time, 3),
                'success_rate': len([p for p in performance_data if p['status'] == 'success']) / len(performance_data),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"⚡ Производительность: среднее время {avg_response_time:.3f}с")
            return performance
            
        except Exception as e:
            logger.error(f"❌ Ошибка проверки производительности: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def update_system_stats(self):
        """Обновление статистики системы"""
        try:
            db_health = self.check_database_health()
            api_health = self.check_api_health()
            performance = self.check_search_performance()
            
            self.stats.update({
                'last_check': datetime.now().isoformat(),
                'database_health': db_health,
                'api_health': api_health,
                'performance': performance,
                'system_health': self.calculate_overall_health(db_health, api_health, performance)
            })
            
            # Сохраняем статистику в файл
            with open('system_stats.json', 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, ensure_ascii=False, indent=2)
            
            logger.info(f"📊 Статистика обновлена: {self.stats['system_health']}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка обновления статистики: {e}")
    
    def calculate_overall_health(self, db_health: Dict, api_health: Dict, performance: Dict) -> str:
        """Расчет общего состояния системы"""
        try:
            health_score = 0
            
            # Оценка БД (40%)
            if db_health.get('status') == 'healthy':
                health_score += 40
            elif db_health.get('status') == 'warning':
                health_score += 20
            
            # Оценка API (40%)
            if api_health.get('overall_status') == 'healthy':
                health_score += 40
            elif api_health.get('overall_status') == 'degraded':
                health_score += 20
            
            # Оценка производительности (20%)
            if performance.get('success_rate', 0) > 0.8:
                health_score += 20
            elif performance.get('success_rate', 0) > 0.5:
                health_score += 10
            
            if health_score >= 80:
                return 'excellent'
            elif health_score >= 60:
                return 'good'
            elif health_score >= 40:
                return 'fair'
            else:
                return 'poor'
                
        except Exception as e:
            logger.error(f"❌ Ошибка расчета здоровья: {e}")
            return 'unknown'

class AutoUpdater:
    """Класс для автоматических обновлений"""
    
    def __init__(self, db_path: str = "rubin_ai_v2.db"):
        self.db_path = db_path
        self.update_log = []
    
    def update_synonym_database(self):
        """Обновление базы синонимов"""
        try:
            logger.info("🔄 Начинаем обновление базы синонимов...")
            
            # Здесь можно добавить логику для автоматического обновления синонимов
            # Например, загрузка новых терминов из внешних источников
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Проверяем, нужно ли обновление
            cursor.execute("SELECT COUNT(*) FROM technical_synonyms")
            current_count = cursor.fetchone()[0]
            
            # Добавляем новые синонимы если нужно
            new_synonyms = [
                ("симистр", "тиристор", "электротехника"),
                ("инвертор", "преобразователь частоты", "электротехника"),
                ("энкодер", "датчик положения", "автоматизация"),
                ("сервопривод", "серводвигатель", "автоматизация"),
                ("частотник", "частотный преобразователь", "электротехника")
            ]
            
            added_count = 0
            for main_term, synonym, category in new_synonyms:
                try:
                    cursor.execute("""
                        INSERT OR IGNORE INTO technical_synonyms 
                        (main_term, synonym, category, confidence)
                        VALUES (?, ?, ?, ?)
                    """, (main_term, synonym, category, 1.0))
                    added_count += 1
                except:
                    pass
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ Обновление синонимов завершено: добавлено {added_count} новых")
            
            self.update_log.append({
                'type': 'synonym_update',
                'timestamp': datetime.now().isoformat(),
                'added_count': added_count,
                'status': 'success'
            })
            
        except Exception as e:
            logger.error(f"❌ Ошибка обновления синонимов: {e}")
            self.update_log.append({
                'type': 'synonym_update',
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'status': 'failed'
            })
    
    def optimize_database(self):
        """Оптимизация базы данных"""
        try:
            logger.info("🔧 Начинаем оптимизацию базы данных...")
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Анализируем базу данных
            cursor.execute("ANALYZE")
            
            # Очищаем старые записи кэша
            cursor.execute("DELETE FROM vocabulary_search_cache WHERE last_accessed < datetime('now', '-7 days')")
            deleted_cache = cursor.rowcount
            
            # Обновляем статистику
            cursor.execute("UPDATE sqlite_stat1 SET stat = (SELECT COUNT(*) FROM documents)")
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ Оптимизация завершена: очищено {deleted_cache} записей кэша")
            
            self.update_log.append({
                'type': 'database_optimization',
                'timestamp': datetime.now().isoformat(),
                'deleted_cache_entries': deleted_cache,
                'status': 'success'
            })
            
        except Exception as e:
            logger.error(f"❌ Ошибка оптимизации БД: {e}")
            self.update_log.append({
                'type': 'database_optimization',
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'status': 'failed'
            })
    
    def backup_database(self):
        """Создание резервной копии базы данных"""
        try:
            logger.info("💾 Создаем резервную копию базы данных...")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"backups/rubin_ai_backup_{timestamp}.db"
            
            # Создаем папку для бэкапов если не существует
            os.makedirs("backups", exist_ok=True)
            
            # Копируем базу данных
            import shutil
            shutil.copy2(self.db_path, backup_path)
            
            # Удаляем старые бэкапы (оставляем только последние 7)
            backup_files = [f for f in os.listdir("backups") if f.startswith("rubin_ai_backup_")]
            backup_files.sort(reverse=True)
            
            for old_backup in backup_files[7:]:
                os.remove(f"backups/{old_backup}")
            
            logger.info(f"✅ Резервная копия создана: {backup_path}")
            
            self.update_log.append({
                'type': 'database_backup',
                'timestamp': datetime.now().isoformat(),
                'backup_path': backup_path,
                'status': 'success'
            })
            
        except Exception as e:
            logger.error(f"❌ Ошибка создания бэкапа: {e}")
            self.update_log.append({
                'type': 'database_backup',
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'status': 'failed'
            })

class SystemScheduler:
    """Планировщик задач системы"""
    
    def __init__(self):
        self.monitor = SystemMonitor()
        self.updater = AutoUpdater()
        self.running = False
    
    def start_monitoring(self):
        """Запуск мониторинга"""
        logger.info("🚀 Запуск системы мониторинга и обновлений...")
        
        # Планируем задачи
        schedule.every(5).minutes.do(self.monitor.update_system_stats)
        schedule.every(1).hours.do(self.updater.update_synonym_database)
        schedule.every(6).hours.do(self.updater.optimize_database)
        schedule.every(24).hours.do(self.updater.backup_database)
        
        self.running = True
        
        # Запускаем мониторинг в отдельном потоке
        monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        monitor_thread.start()
        
        logger.info("✅ Система мониторинга запущена!")
        logger.info("📋 Расписание:")
        logger.info("  - Проверка системы: каждые 5 минут")
        logger.info("  - Обновление синонимов: каждый час")
        logger.info("  - Оптимизация БД: каждые 6 часов")
        logger.info("  - Резервное копирование: каждый день")
        
        return monitor_thread
    
    def _monitoring_loop(self):
        """Основной цикл мониторинга"""
        while self.running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Проверяем каждую минуту
            except Exception as e:
                logger.error(f"❌ Ошибка в цикле мониторинга: {e}")
                time.sleep(60)
    
    def stop_monitoring(self):
        """Остановка мониторинга"""
        self.running = False
        logger.info("🛑 Система мониторинга остановлена")
    
    def get_status_report(self) -> Dict[str, Any]:
        """Получение отчета о состоянии системы"""
        try:
            # Загружаем последнюю статистику
            if os.path.exists('system_stats.json'):
                with open('system_stats.json', 'r', encoding='utf-8') as f:
                    stats = json.load(f)
            else:
                stats = self.monitor.stats
            
            # Добавляем информацию о планировщике
            report = {
                'system_status': stats,
                'scheduler_status': {
                    'running': self.running,
                    'next_jobs': [str(job) for job in schedule.jobs],
                    'update_log': self.updater.update_log[-10:]  # Последние 10 обновлений
                },
                'timestamp': datetime.now().isoformat()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения отчета: {e}")
            return {'error': str(e)}

def main():
    """Основная функция"""
    print("🚀 СИСТЕМА МОНИТОРИНГА И ОБНОВЛЕНИЙ RUBIN AI")
    print("=" * 60)
    
    scheduler = SystemScheduler()
    
    try:
        # Запускаем мониторинг
        monitor_thread = scheduler.start_monitoring()
        
        # Выводим начальный отчет
        print("\n📊 НАЧАЛЬНЫЙ ОТЧЕТ СИСТЕМЫ:")
        initial_report = scheduler.get_status_report()
        print(f"  - Состояние системы: {initial_report['system_status'].get('system_health', 'unknown')}")
        print(f"  - Документов в БД: {initial_report['system_status'].get('database_health', {}).get('document_count', 0)}")
        print(f"  - Синонимов в БД: {initial_report['system_status'].get('database_health', {}).get('synonym_count', 0)}")
        
        print("\n🔄 Система мониторинга работает в фоновом режиме...")
        print("📝 Логи сохраняются в system_monitor.log")
        print("📊 Статистика сохраняется в system_stats.json")
        print("💾 Резервные копии создаются в папке backups/")
        print("\nНажмите Ctrl+C для остановки...")
        
        # Ждем завершения
        monitor_thread.join()
        
    except KeyboardInterrupt:
        print("\n🛑 Получен сигнал остановки...")
        scheduler.stop_monitoring()
        print("✅ Система мониторинга остановлена")
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
        scheduler.stop_monitoring()

if __name__ == "__main__":
    main()






















