#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Загрузка технической литературы в Rubin AI v2.0
"""

import os
import sys
from pathlib import Path

def main():
    """Главная функция"""
    print("📚 ЗАГРУЗКА ТЕХНИЧЕСКОЙ ЛИТЕРАТУРЫ В RUBIN AI v2.0")
    print("=" * 60)
    
    # Путь к папке с технической литературой
    tech_literature_path = r"E:\03.Тех.литература"
    
    print(f"📁 Проверяем папку: {tech_literature_path}")
    
    # Проверяем существование папки
    if not os.path.exists(tech_literature_path):
        print(f"❌ Папка не найдена: {tech_literature_path}")
        print("\n🔧 Возможные решения:")
        print("   1. Проверьте правильность пути")
        print("   2. Убедитесь, что диск E: подключен")
        print("   3. Проверьте права доступа к папке")
        
        # Предлагаем альтернативные пути
        alternative_paths = [
            r"C:\Users\{}\Documents\03.Тех.литература".format(os.getenv('USERNAME')),
            r"C:\03.Тех.литература",
            r"D:\03.Тех.литература",
            "sample_documents"  # Папка с образцами
        ]
        
        print(f"\n📂 Альтернативные пути:")
        for i, path in enumerate(alternative_paths, 1):
            if os.path.exists(path):
                print(f"   {i}. ✅ {path} (найден)")
            else:
                print(f"   {i}. ❌ {path} (не найден)")
        
        # Создаем папку с образцами, если ничего не найдено
        print(f"\n🎯 Создаем папку с образцами документов...")
        from start_documents_system import create_sample_documents
        docs_dir = create_sample_documents()
        tech_literature_path = str(docs_dir)
        print(f"✅ Создана папка с образцами: {tech_literature_path}")
    
    else:
        print(f"✅ Папка найдена: {tech_literature_path}")
        
        # Показываем содержимое папки
        try:
            files = list(Path(tech_literature_path).rglob('*'))
            file_count = len([f for f in files if f.is_file()])
            print(f"📄 Найдено файлов: {file_count}")
            
            # Показываем первые 10 файлов
            print(f"\n📋 Первые 10 файлов:")
            for i, file_path in enumerate(files[:10]):
                if file_path.is_file():
                    size = file_path.stat().st_size
                    print(f"   {i+1}. {file_path.name} ({size} байт)")
            
            if file_count > 10:
                print(f"   ... и еще {file_count - 10} файлов")
                
        except Exception as e:
            print(f"⚠️ Ошибка чтения папки: {e}")
    
    # Запускаем загрузку
    print(f"\n📥 Начинаем загрузку документов...")
    
    try:
        from document_loader import DocumentLoader
        loader = DocumentLoader()
        
        success = loader.load_directory(tech_literature_path)
        
        if success:
            print(f"\n✅ ЗАГРУЗКА ЗАВЕРШЕНА УСПЕШНО!")
            print("=" * 50)
            
            # Показываем статистику
            stats = loader.get_document_stats()
            if stats:
                print(f"📊 СТАТИСТИКА ЗАГРУЗКИ:")
                print(f"   📚 Всего документов: {stats['total_documents']}")
                print(f"   📂 Категорий: {len(stats['categories'])}")
                print(f"   📄 Типов файлов: {len(stats['file_types'])}")
                
                print(f"\n📂 КАТЕГОРИИ ДОКУМЕНТОВ:")
                for category, count in stats['categories']:
                    print(f"   • {category}: {count} документов")
                
                print(f"\n📄 ТИПЫ ФАЙЛОВ:")
                for file_type, count in stats['file_types']:
                    print(f"   • {file_type}: {count} файлов")
            
            print(f"\n🌐 СЛЕДУЮЩИЕ ШАГИ:")
            print(f"   1. Запустите API документов:")
            print(f"      python api/documents_api.py")
            print(f"   2. Откройте веб-интерфейс:")
            print(f"      http://localhost:8088/DocumentsManager.html")
            print(f"   3. Или используйте поиск через Rubin AI")
            
            print(f"\n🔍 ПРИМЕРЫ ПОИСКОВЫХ ЗАПРОСОВ:")
            print(f"   • 'закон ома' - найдет документы по электротехнике")
            print(f"   • 'алгоритм' - найдет документы по программированию")
            print(f"   • 'pid регулятор' - найдет документы по автоматизации")
            print(f"   • 'модуляция' - найдет документы по радиотехнике")
            
        else:
            print(f"❌ Ошибка при загрузке документов")
            print(f"Проверьте логи в файле document_loader.log")
            
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        print(f"Убедитесь, что все зависимости установлены:")
        print(f"   pip install flask flask-cors")

if __name__ == "__main__":
    main()


















