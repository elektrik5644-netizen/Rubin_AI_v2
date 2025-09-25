import sqlite3
import os

def check_vector_indexing():
    db_path = "rubin_ai_documents.db"
    if not os.path.exists(db_path):
        print(f"❌ База данных не найдена: {db_path}")
        return
    
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Проверим таблицу документов
        cursor.execute("SELECT COUNT(*) FROM documents")
        doc_count = cursor.fetchone()[0]
        print(f"📊 Всего документов в таблице documents: {doc_count}")
        
        # Проверим таблицу векторного поиска
        cursor.execute("SELECT COUNT(*) FROM document_embeddings")
        vector_count = cursor.fetchone()[0]
        print(f"📊 Всего документов в таблице document_embeddings: {vector_count}")
        
        # Проверим конкретный документ с EnDat
        cursor.execute("""
            SELECT de.document_id, d.file_name, d.category
            FROM document_embeddings de
            JOIN documents d ON de.document_id = d.id
            WHERE d.file_name LIKE '%Endat%'
        """)
        
        endat_docs = cursor.fetchall()
        print(f"\n📊 Документы с Endat в векторном индексе: {len(endat_docs)}")
        
        for doc in endat_docs:
            doc_id, file_name, category = doc
            print(f"   ✅ Документ {doc_id}: {file_name} ({category})")
        
        # Проверим, какие документы НЕ проиндексированы
        cursor.execute("""
            SELECT d.id, d.file_name, d.category
            FROM documents d
            LEFT JOIN document_embeddings de ON d.id = de.document_id
            WHERE de.document_id IS NULL
        """)
        
        unindexed_docs = cursor.fetchall()
        print(f"\n📊 НЕ проиндексированные документы: {len(unindexed_docs)}")
        
        for doc in unindexed_docs[:10]:  # Показываем первые 10
            doc_id, file_name, category = doc
            print(f"   ❌ Документ {doc_id}: {file_name} ({category})")
        
        if len(unindexed_docs) > 10:
            print(f"   ... и еще {len(unindexed_docs) - 10} документов")
        
    except sqlite3.Error as e:
        print(f"❌ Ошибка SQLite: {e}")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    check_vector_indexing()

















