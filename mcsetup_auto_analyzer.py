#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Автоматический анализатор MCSetup
Мониторит изменения в MCSetup и автоматически анализирует графики
"""

import os
import time
import json
import requests
import logging
from datetime import datetime
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import xml.etree.ElementTree as ET

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Конфигурация
MCSETUP_PATH = r"C:\Users\elekt\OneDrive\Desktop\MCSetup_V1_9_0\MCSetup_V1_9_0"
MCSETUP_BRIDGE_URL = "http://localhost:8096"
RUBIN_SMART_DISPATCHER_URL = "http://localhost:8080"
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')

class MCSetupFileHandler(FileSystemEventHandler):
    """Обработчик изменений файлов MCSetup"""
    
    def __init__(self):
        self.last_analysis_time = {}
        self.analysis_cooldown = 5  # секунд между анализами одного файла
        
    def on_modified(self, event):
        """Обрабатывает изменения файлов"""
        if event.is_directory:
            return
            
        file_path = Path(event.src_path)
        
        # Проверяем, что это XML файл графика
        if file_path.suffix.lower() == '.xml' and 'plot' in file_path.name.lower():
            self.handle_plot_file_change(file_path)
    
    def handle_plot_file_change(self, file_path):
        """Обрабатывает изменения файла графика"""
        try:
            current_time = time.time()
            file_key = str(file_path)
            
            # Проверяем cooldown
            if file_key in self.last_analysis_time:
                if current_time - self.last_analysis_time[file_key] < self.analysis_cooldown:
                    return
            
            self.last_analysis_time[file_key] = current_time
            
            logger.info(f"📊 Обнаружено изменение графика: {file_path.name}")
            
            # Ждем немного, чтобы файл полностью сохранился
            time.sleep(2)
            
            # Анализируем изменения
            self.analyze_plot_changes(file_path)
            
        except Exception as e:
            logger.error(f"❌ Ошибка обработки изменения файла {file_path}: {e}")
    
    def analyze_plot_changes(self, file_path):
        """Анализирует изменения в файле графика"""
        try:
            # Читаем XML файл
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Извлекаем информацию о графике
            plot_info = self.extract_plot_info(root, file_path.name)
            
            # Отправляем на анализ в MCSetup Bridge
            analysis_result = self.send_for_analysis(plot_info)
            
            if analysis_result:
                # Отправляем рекомендации в Telegram
                self.send_recommendations_to_telegram(plot_info, analysis_result)
                
        except Exception as e:
            logger.error(f"❌ Ошибка анализа графика {file_path}: {e}")
    
    def extract_plot_info(self, root, filename):
        """Извлекает информацию о графике из XML"""
        plot_info = {
            'filename': filename,
            'timestamp': datetime.now().isoformat(),
            'gather_period': '10',
            'sources': [],
            'changes_detected': True
        }
        
        try:
            # Извлекаем период сбора данных
            gather_period = root.find('.//param[@name="gatherPeriod"]')
            if gather_period is not None:
                plot_info['gather_period'] = gather_period.get('value', '10')
            
            # Извлекаем источники данных
            for item in root.findall('.//group[@name="itemToGather"]'):
                source = {
                    'address': item.find('param[@name="address"]').get('value', '') if item.find('param[@name="address"]') is not None else '',
                    'name': item.find('param[@name="name"]').get('value', '') if item.find('param[@name="name"]') is not None else '',
                    'enabled': item.find('param[@name="enable"]').get('value', 'false') == 'true' if item.find('param[@name="enable"]') is not None else False
                }
                plot_info['sources'].append(source)
                
        except Exception as e:
            logger.warning(f"⚠️ Ошибка извлечения информации из {filename}: {e}")
        
        return plot_info
    
    def send_for_analysis(self, plot_info):
        """Отправляет данные на анализ в MCSetup Bridge"""
        try:
            # Отправляем в MCSetup Bridge для анализа
            response = requests.post(
                f"{MCSETUP_BRIDGE_URL}/api/mcsetup/analyze/graphs",
                json={
                    'graph_name': plot_info['filename'],
                    'plot_info': plot_info,
                    'auto_analysis': True
                },
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"❌ Ошибка анализа в MCSetup Bridge: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Ошибка отправки на анализ: {e}")
            return None
    
    def send_recommendations_to_telegram(self, plot_info, analysis_result):
        """Отправляет рекомендации в Telegram"""
        try:
            if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
                logger.warning("⚠️ Telegram не настроен")
                return
            
            # Формируем сообщение
            message = f"📊 **Автоматический анализ MCSetup**\n\n"
            message += f"**График:** {plot_info['filename']}\n"
            message += f"**Время:** {datetime.now().strftime('%H:%M:%S')}\n\n"
            
            # Добавляем информацию об анализе
            if 'analysis' in analysis_result:
                analysis = analysis_result['analysis']
                
                if 'insights' in analysis:
                    message += "**Обнаруженные изменения:**\n"
                    for insight in analysis['insights'][:3]:  # Первые 3 инсайта
                        message += f"• {insight}\n"
                    message += "\n"
                
                if 'graph_analysis' in analysis:
                    graph_analysis = analysis['graph_analysis']
                    if plot_info['filename'] in graph_analysis:
                        motor_data = graph_analysis[plot_info['filename']]
                        message += f"**Активных источников:** {motor_data.get('active_sources', 0)}\n"
                        message += f"**Период сбора:** {motor_data.get('gather_period', 'N/A')}мс\n\n"
            
            # Отправляем запрос в Rubin AI для рекомендаций
            rubin_recommendations = self.get_rubin_recommendations(plot_info)
            if rubin_recommendations:
                message += f"**Рекомендации Rubin AI:**\n{rubin_recommendations}\n\n"
            
            message += "🤖 *Автоматический анализ завершен*"
            
            # Отправляем в Telegram
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = {
                'chat_id': TELEGRAM_CHAT_ID,
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                logger.info(f"✅ Рекомендации отправлены в Telegram для {plot_info['filename']}")
            else:
                logger.error(f"❌ Ошибка отправки в Telegram: {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ Ошибка отправки рекомендаций: {e}")
    
    def get_rubin_recommendations(self, plot_info):
        """Получает рекомендации от Rubin AI"""
        try:
            # Формируем запрос для Rubin AI
            query = f"Анализ графика {plot_info['filename']} с периодом сбора {plot_info['gather_period']}мс. "
            query += f"Активных источников: {len([s for s in plot_info['sources'] if s['enabled']])}. "
            query += "Дай рекомендации по оптимизации настроек."
            
            # Отправляем в Smart Dispatcher
            response = requests.post(
                f"{RUBIN_SMART_DISPATCHER_URL}/api/chat",
                json={
                    'message': query,
                    'context': {
                        'source': 'mcsetup_auto_analyzer',
                        'plot_info': plot_info
                    }
                },
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'response' in result:
                    if isinstance(result['response'], dict):
                        return result['response'].get('explanation', str(result['response']))
                    else:
                        return str(result['response'])
                return str(result)
            else:
                logger.error(f"❌ Ошибка получения рекомендаций от Rubin AI: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Ошибка запроса рекомендаций: {e}")
            return None

class MCSetupAutoAnalyzer:
    """Автоматический анализатор MCSetup"""
    
    def __init__(self):
        self.observer = Observer()
        self.event_handler = MCSetupFileHandler()
        self.mcsetup_path = Path(MCSETUP_PATH)
        
    def start_monitoring(self):
        """Запускает мониторинг файлов MCSetup"""
        try:
            # Проверяем существование пути MCSetup
            if not self.mcsetup_path.exists():
                logger.error(f"❌ Путь MCSetup не найден: {self.mcsetup_path}")
                return False
            
            # Находим директорию с графиками
            plot_dir = self.mcsetup_path / "Plot, Watch windows" / "Plot"
            if not plot_dir.exists():
                logger.error(f"❌ Директория графиков не найдена: {plot_dir}")
                return False
            
            # Запускаем мониторинг
            self.observer.schedule(self.event_handler, str(plot_dir), recursive=False)
            self.observer.start()
            
            logger.info(f"✅ Мониторинг MCSetup запущен: {plot_dir}")
            logger.info("📊 Ожидание изменений в графиках...")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка запуска мониторинга: {e}")
            return False
    
    def stop_monitoring(self):
        """Останавливает мониторинг"""
        try:
            if self.observer.is_alive():
                self.observer.stop()
                self.observer.join()
            logger.info("⏹️ Мониторинг MCSetup остановлен")
        except Exception as e:
            logger.error(f"❌ Ошибка остановки мониторинга: {e}")
    
    def run(self):
        """Запускает автоматический анализатор"""
        try:
            if not self.start_monitoring():
                return
            
            logger.info("🚀 Автоматический анализатор MCSetup запущен")
            logger.info("Нажмите Ctrl+C для остановки")
            
            # Основной цикл
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("⏹️ Получен сигнал остановки")
        except Exception as e:
            logger.error(f"❌ Критическая ошибка: {e}")
        finally:
            self.stop_monitoring()

def main():
    """Главная функция"""
    print("🤖 Автоматический анализатор MCSetup")
    print("=" * 50)
    
    analyzer = MCSetupAutoAnalyzer()
    analyzer.run()

if __name__ == '__main__':
    main()
