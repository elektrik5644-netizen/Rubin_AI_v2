import sqlite3
import os

def check_encoder_documents():
    db_path = "rubin_ai_documents.db"
    if not os.path.exists(db_path):
        print(f"❌ База данных не найдена: {db_path}")
        return
    
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Ищем документы, связанные с энкодерами
        cursor.execute("""
            SELECT id, file_name, content, category 
            FROM documents 
            WHERE file_name LIKE '%энкодер%' 
               OR file_name LIKE '%encoder%'
               OR content LIKE '%энкодер%'
               OR content LIKE '%encoder%'
               OR file_name LIKE '%мотор%'
               OR file_name LIKE '%motor%'
            LIMIT 5
        """)
        
        documents = cursor.fetchall()
        print(f"📊 Найдено документов: {len(documents)}")
        
        for doc in documents:
            doc_id, file_name, content, category = doc
            print(f"\n🔸 Документ {doc_id}:")
            print(f"   Файл: {file_name}")
            print(f"   Категория: {category}")
            print(f"   Длина содержимого: {len(content)} символов")
            print(f"   Первые 200 символов:")
            print(f"   {repr(content[:200])}")
            print(f"   Последние 200 символов:")
            print(f"   {repr(content[-200:])}")
        
        # Проверим общее количество документов
        cursor.execute("SELECT COUNT(*) FROM documents")
        total_count = cursor.fetchone()[0]
        print(f"\n📊 Всего документов в базе: {total_count}")
        
    except sqlite3.Error as e:
        print(f"❌ Ошибка SQLite: {e}")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    check_encoder_documents()






















