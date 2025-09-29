import sqlite3
import os

def check_endat_document():
    db_path = "rubin_ai_documents.db"
    if not os.path.exists(db_path):
        print(f"❌ База данных не найдена: {db_path}")
        return
    
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Найдем документ с EnDat
        cursor.execute("""
            SELECT id, file_name, content, category 
            FROM documents 
            WHERE file_name LIKE '%Endat%' OR file_name LIKE '%endat%'
        """)
        
        documents = cursor.fetchall()
        print(f"📊 Найдено документов с Endat: {len(documents)}")
        
        for doc in documents:
            doc_id, file_name, content, category = doc
            print(f"\n🔸 Документ {doc_id}:")
            print(f"   Файл: {file_name}")
            print(f"   Категория: {category}")
            print(f"   Длина содержимого: {len(content)} символов")
            
            # Ищем упоминания энкодеров в тексте
            if 'энкодер' in content.lower() or 'encoder' in content.lower():
                print("   ✅ Содержит упоминания энкодеров!")
            else:
                print("   ❌ Не содержит упоминаний энкодеров")
            
            print(f"   Первые 500 символов:")
            print(f"   {content[:500]}")
            
            # Ищем ключевые слова
            keywords = ['EnDat', 'энкодер', 'encoder', 'датчик', 'sensor', 'положение', 'position']
            found_keywords = []
            for keyword in keywords:
                if keyword.lower() in content.lower():
                    found_keywords.append(keyword)
            
            if found_keywords:
                print(f"   🔍 Найденные ключевые слова: {', '.join(found_keywords)}")
        
    except sqlite3.Error as e:
        print(f"❌ Ошибка SQLite: {e}")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    check_endat_document()






















