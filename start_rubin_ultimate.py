#!/usr/bin/env python3
"""
Запуск Ultimate версии Rubin AI с полной базой знаний и улучшенной математикой
"""

import subprocess
import sys
import os
import time
import webbrowser
from threading import Thread

def start_rubin_ultimate():
    """Запуск Ultimate версии Rubin AI"""
    
    print("🚀 ЗАПУСК RUBIN AI - ULTIMATE ВЕРСИЯ")
    print("=" * 50)
    
    # Проверяем, что файлы существуют
    required_files = [
        'rubin_ultimate_system.py',
        'rubin_math_solver.py',
        'rubin_final_interface.html'
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            print(f"❌ Файл {file} не найден!")
            return False
    
    print("✅ Все файлы найдены")
    
    # Проверяем базы данных
    db_files = [
        'rubin_ai_v2.db',
        'rubin_ai_documents.db',
        'rubin_knowledge_base.db'
    ]
    
    print("🗄️ Проверка баз данных:")
    for db_file in db_files:
        if os.path.exists(db_file):
            size = os.path.getsize(db_file)
            print(f"  ✅ {db_file}: {size:,} байт")
        else:
            print(f"  ❌ {db_file}: не найден")
    
    # Проверяем тестовые документы
    test_docs_dir = 'test_documents'
    if os.path.exists(test_docs_dir):
        files = os.listdir(test_docs_dir)
        print(f"📚 Тестовые документы: {len(files)} файлов")
        for file in files:
            size = os.path.getsize(os.path.join(test_docs_dir, file))
            print(f"  📄 {file}: {size:,} байт")
    else:
        print("❌ Папка test_documents не найдена")
    
    # Запускаем сервер
    print("\n🌐 Запуск Ultimate сервера на порту 8088...")
    try:
        # Запускаем сервер в отдельном процессе
        server_process = subprocess.Popen([
            sys.executable, 'rubin_ultimate_system.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Ждем немного для запуска
        time.sleep(3)
        
        # Проверяем, что сервер запустился
        if server_process.poll() is None:
            print("✅ Ultimate сервер запущен успешно")
            
            # Открываем браузер
            def open_browser():
                time.sleep(2)
                webbrowser.open('file://' + os.path.abspath('rubin_final_interface.html'))
            
            browser_thread = Thread(target=open_browser)
            browser_thread.daemon = True
            browser_thread.start()
            
            print("🌐 Открываем интерфейс в браузере...")
            print("\n📡 СИСТЕМА ГОТОВА К РАБОТЕ!")
            print("🔗 API: http://localhost:8088")
            print("🌐 Интерфейс: rubin_final_interface.html")
            print("\n💡 Возможности Ultimate системы:")
            print("  • Поиск в полной базе знаний")
            print("  • 7 тестовых документов")
            print("  • Множественные базы данных")
            print("  • Улучшенные математические вычисления")
            print("  • Задачи на движение, углы, количество")
            print("  • Умные ответы с источниками")
            print("  • Современный веб-интерфейс")
            print("\n🛑 Для остановки нажмите Ctrl+C")
            
            try:
                # Ждем завершения процесса
                server_process.wait()
            except KeyboardInterrupt:
                print("\n🛑 Остановка сервера...")
                server_process.terminate()
                server_process.wait()
                print("✅ Сервер остановлен")
            
            return True
        else:
            stdout, stderr = server_process.communicate()
            print(f"❌ Ошибка запуска сервера:")
            print(f"STDOUT: {stdout.decode()}")
            print(f"STDERR: {stderr.decode()}")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False

def show_system_info():
    """Показать информацию о системе"""
    print("\n📊 ИНФОРМАЦИЯ О ULTIMATE СИСТЕМЕ:")
    print("=" * 40)
    
    # Размер баз данных
    total_db_size = 0
    for db_file in ['rubin_ai_v2.db', 'rubin_ai_documents.db', 'rubin_knowledge_base.db']:
        if os.path.exists(db_file):
            size = os.path.getsize(db_file)
            total_db_size += size
            print(f"📊 {db_file}: {size:,} байт")
    
    print(f"📊 Общий размер БД: {total_db_size:,} байт")
    
    # Тестовые документы
    test_docs_dir = 'test_documents'
    if os.path.exists(test_docs_dir):
        files = os.listdir(test_docs_dir)
        total_docs_size = 0
        for file in files:
            size = os.path.getsize(os.path.join(test_docs_dir, file))
            total_docs_size += size
        print(f"📚 Тестовые документы: {len(files)} файлов, {total_docs_size:,} байт")
    
    print(f"💾 Общий объем данных: {total_db_size + total_docs_size:,} байт")
    
    # Проверяем математический решатель
    if os.path.exists('rubin_math_solver.py'):
        print("🧮 Улучшенный математический решатель: ✅ Доступен")
    else:
        print("🧮 Улучшенный математический решатель: ❌ Недоступен")

if __name__ == "__main__":
    print("🤖 RUBIN AI - ULTIMATE ВЕРСИЯ")
    print("=" * 50)
    
    show_system_info()
    
    # Запускаем систему
    if start_rubin_ultimate():
        print("\n✅ Ultimate система запущена успешно!")
        print("🎉 Rubin AI готов к работе с полной базой знаний и улучшенной математикой!")
    else:
        print("\n❌ Ошибка запуска системы")
        sys.exit(1)












