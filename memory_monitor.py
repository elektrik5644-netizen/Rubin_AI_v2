#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 Memory Monitor - Мониторинг потребления памяти серверами
"""

import psutil
import time
import json
from datetime import datetime

def get_memory_usage():
    """Получить информацию о потреблении памяти"""
    memory = psutil.virtual_memory()
    return {
        "total": memory.total,
        "available": memory.available,
        "used": memory.used,
        "percentage": memory.percent,
        "free": memory.free
    }

def get_processes_memory():
    """Получить информацию о потреблении памяти процессами Python"""
    python_processes = []
    
    for proc in psutil.process_iter(['pid', 'name', 'memory_info', 'cmdline']):
        try:
            if proc.info['name'] and 'python' in proc.info['name'].lower():
                memory_info = proc.info['memory_info']
                cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                
                python_processes.append({
                    "pid": proc.info['pid'],
                    "name": proc.info['name'],
                    "memory_rss": memory_info.rss,  # Resident Set Size
                    "memory_vms": memory_info.vms,  # Virtual Memory Size
                    "memory_percent": proc.memory_percent(),
                    "cmdline": cmdline
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    return python_processes

def monitor_memory():
    """Мониторинг памяти в реальном времени"""
    print("📊 Memory Monitor запущен")
    print("=" * 60)
    
    while True:
        try:
            # Общая информация о памяти
            memory = get_memory_usage()
            processes = get_processes_memory()
            
            print(f"\n🕐 {datetime.now().strftime('%H:%M:%S')}")
            print(f"💾 Общая память: {memory['used'] // (1024**3):.1f}GB / {memory['total'] // (1024**3):.1f}GB ({memory['percentage']:.1f}%)")
            print(f"🆓 Доступно: {memory['available'] // (1024**3):.1f}GB")
            
            if memory['percentage'] > 80:
                print("⚠️  ВНИМАНИЕ: Использование памяти превышает 80%!")
            
            # Топ процессов по потреблению памяти
            processes.sort(key=lambda x: x['memory_rss'], reverse=True)
            print(f"\n🐍 Python процессы (топ-5):")
            for i, proc in enumerate(processes[:5]):
                memory_mb = proc['memory_rss'] // (1024**2)
                print(f"  {i+1}. PID {proc['pid']}: {memory_mb}MB - {proc['cmdline'][:50]}...")
            
            time.sleep(10)  # Проверка каждые 10 секунд
            
        except KeyboardInterrupt:
            print("\n🛑 Мониторинг остановлен")
            break
        except Exception as e:
            print(f"❌ Ошибка мониторинга: {e}")
            time.sleep(5)

if __name__ == '__main__':
    monitor_memory()








