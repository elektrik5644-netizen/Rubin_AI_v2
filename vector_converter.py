#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Конвертер векторов в базе данных Rubin AI v2.0
Поддерживает различные форматы конвертации векторов
"""

import os
import sys
import sqlite3
import numpy as np
import logging
import json
from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path
import hashlib
from datetime import datetime
import argparse

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️ sentence-transformers не установлен. Установите: pip install sentence-transformers")

class VectorConverter:
    """Конвертер векторов для различных форматов и размерностей"""
    
    def __init__(self, db_path="rubin_ai_documents.db", backup_path=None):
        self.db_path = db_path
        self.backup_path = backup_path or f"{db_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.setup_logging()
        
    def setup_logging(self):
        """Настройка логирования"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('vector_converter.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def backup_database(self) -> bool:
        """Создание резервной копии базы данных"""
        try:
            if not os.path.exists(self.db_path):
                self.logger.error(f"❌ База данных {self.db_path} не найдена")
                return False
                
            import shutil
            shutil.copy2(self.db_path, self.backup_path)
            self.logger.info(f"✅ Резервная копия создана: {self.backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка создания резервной копии: {e}")
            return False
            
    def analyze_vectors(self) -> Dict:
        """Анализ текущих векторов в базе данных"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Проверка существования таблиц
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='document_vectors'")
            if not cursor.fetchone():
                self.logger.warning("⚠️ Таблица document_vectors не найдена")
                return {}
                
            # Подсчет векторов
            cursor.execute("SELECT COUNT(*) FROM document_vectors")
            vector_count = cursor.fetchone()[0]
            
            # Анализ размерности векторов
            cursor.execute("SELECT vector_data FROM document_vectors LIMIT 1")
            sample_vector = cursor.fetchone()
            
            if sample_vector:
                vector_data = np.frombuffer(sample_vector[0], dtype=np.float32)
                dimension = len(vector_data)
            else:
                dimension = 0
                
            # Анализ метаданных
            cursor.execute("SELECT COUNT(*) FROM vector_metadata")
            metadata_count = cursor.fetchone()[0]
            
            # Анализ документов
            cursor.execute("SELECT COUNT(*) FROM documents")
            document_count = cursor.fetchone()[0]
            
            conn.close()
            
            analysis = {
                'vector_count': vector_count,
                'dimension': dimension,
                'metadata_count': metadata_count,
                'document_count': document_count,
                'vector_size_bytes': dimension * 4 if dimension > 0 else 0,  # float32 = 4 bytes
                'total_vector_size_mb': (vector_count * dimension * 4) / (1024 * 1024) if dimension > 0 else 0
            }
            
            self.logger.info("📊 Анализ векторов:")
            for key, value in analysis.items():
                self.logger.info(f"  {key}: {value}")
                
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка анализа векторов: {e}")
            return {}
            
    def convert_vectors_to_csv(self, output_file: str = "vectors_export.csv") -> bool:
        """Конвертация векторов в CSV формат"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT dv.id, dv.document_id, dv.vector_data, dv.vector_hash, dv.content_hash,
                       d.file_name, d.content, vm.content_preview, vm.keywords, vm.category
                FROM document_vectors dv
                LEFT JOIN documents d ON dv.document_id = d.id
                LEFT JOIN vector_metadata vm ON dv.document_id = vm.document_id
            """)
            
            vectors = cursor.fetchall()
            
            if not vectors:
                self.logger.warning("⚠️ Векторы не найдены")
                return False
                
            # Подготовка CSV
            import csv
            
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Заголовки
                sample_vector = np.frombuffer(vectors[0][2], dtype=np.float32)
                headers = ['id', 'document_id', 'vector_hash', 'content_hash', 'file_name', 'content_preview', 'keywords', 'category']
                headers.extend([f'vector_{i}' for i in range(len(sample_vector))])
                writer.writerow(headers)
                
                # Данные
                for row in vectors:
                    vector_data = np.frombuffer(row[2], dtype=np.float32)
                    csv_row = [
                        row[0],  # id
                        row[1],  # document_id
                        row[3],  # vector_hash
                        row[4],  # content_hash
                        row[5] or '',  # title
                        row[7] or '',  # content_preview
                        row[8] or '',  # keywords
                        row[9] or ''   # category
                    ]
                    csv_row.extend(vector_data.tolist())
                    writer.writerow(csv_row)
                    
            conn.close()
            self.logger.info(f"✅ Векторы экспортированы в {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка экспорта в CSV: {e}")
            return False
            
    def convert_vectors_to_json(self, output_file: str = "vectors_export.json") -> bool:
        """Конвертация векторов в JSON формат"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT dv.id, dv.document_id, dv.vector_data, dv.vector_hash, dv.content_hash,
                       d.file_name, d.content, vm.content_preview, vm.keywords, vm.category
                FROM document_vectors dv
                LEFT JOIN documents d ON dv.document_id = d.id
                LEFT JOIN vector_metadata vm ON dv.document_id = vm.document_id
            """)
            
            vectors = cursor.fetchall()
            
            if not vectors:
                self.logger.warning("⚠️ Векторы не найдены")
                return False
                
            # Подготовка JSON
            json_data = []
            
            for row in vectors:
                vector_data = np.frombuffer(row[2], dtype=np.float32)
                
                vector_info = {
                    'id': row[0],
                    'document_id': row[1],
                    'vector_hash': row[3],
                    'content_hash': row[4],
                    'file_name': row[5] or '',
                    'content_preview': row[7] or '',
                    'keywords': row[8] or '',
                    'category': row[9] or '',
                    'vector': vector_data.tolist(),
                    'dimension': len(vector_data)
                }
                json_data.append(vector_info)
                
            with open(output_file, 'w', encoding='utf-8') as jsonfile:
                json.dump(json_data, jsonfile, ensure_ascii=False, indent=2)
                
            conn.close()
            self.logger.info(f"✅ Векторы экспортированы в {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка экспорта в JSON: {e}")
            return False
            
    def convert_vectors_to_numpy(self, output_file: str = "vectors_export.npz") -> bool:
        """Конвертация векторов в NumPy формат"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT dv.id, dv.document_id, dv.vector_data, dv.vector_hash, dv.content_hash,
                       d.file_name, d.content, vm.content_preview, vm.keywords, vm.category
                FROM document_vectors dv
                LEFT JOIN documents d ON dv.document_id = d.id
                LEFT JOIN vector_metadata vm ON dv.document_id = vm.document_id
            """)
            
            vectors = cursor.fetchall()
            
            if not vectors:
                self.logger.warning("⚠️ Векторы не найдены")
                return False
                
            # Подготовка NumPy массивов
            vector_list = []
            metadata_list = []
            
            for row in vectors:
                vector_data = np.frombuffer(row[2], dtype=np.float32)
                vector_list.append(vector_data)
                
                metadata = {
                    'id': row[0],
                    'document_id': row[1],
                    'vector_hash': row[3],
                    'content_hash': row[4],
                    'file_name': row[5] or '',
                    'content_preview': row[7] or '',
                    'keywords': row[8] or '',
                    'category': row[9] or ''
                }
                metadata_list.append(metadata)
                
            # Сохранение в NPZ
            vectors_array = np.array(vector_list)
            np.savez_compressed(
                output_file,
                vectors=vectors_array,
                metadata=metadata_list
            )
            
            conn.close()
            self.logger.info(f"✅ Векторы экспортированы в {output_file}")
            self.logger.info(f"📊 Размер массива векторов: {vectors_array.shape}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка экспорта в NumPy: {e}")
            return False
            
    def convert_vectors_to_text(self, output_file: str = "vectors_export.txt") -> bool:
        """Конвертация векторов в текстовый формат"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT dv.id, dv.document_id, dv.vector_data, dv.vector_hash, dv.content_hash,
                       d.file_name, d.content, vm.content_preview, vm.keywords, vm.category
                FROM document_vectors dv
                LEFT JOIN documents d ON dv.document_id = d.id
                LEFT JOIN vector_metadata vm ON dv.document_id = vm.document_id
            """)
            
            vectors = cursor.fetchall()
            
            if not vectors:
                self.logger.warning("⚠️ Векторы не найдены")
                return False
                
            with open(output_file, 'w', encoding='utf-8') as txtfile:
                txtfile.write("# Rubin AI Vectors Export\n")
                txtfile.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                txtfile.write(f"# Total vectors: {len(vectors)}\n\n")
                
                for row in vectors:
                    vector_data = np.frombuffer(row[2], dtype=np.float32)
                    
                    txtfile.write(f"=== Vector ID: {row[0]} ===\n")
                    txtfile.write(f"Document ID: {row[1]}\n")
                    txtfile.write(f"Vector Hash: {row[3]}\n")
                    txtfile.write(f"Content Hash: {row[4]}\n")
                    txtfile.write(f"File Name: {row[5] or 'N/A'}\n")
                    txtfile.write(f"Content Preview: {row[7] or 'N/A'}\n")
                    txtfile.write(f"Keywords: {row[8] or 'N/A'}\n")
                    txtfile.write(f"Category: {row[9] or 'N/A'}\n")
                    txtfile.write(f"Dimension: {len(vector_data)}\n")
                    txtfile.write(f"Vector: {vector_data.tolist()}\n\n")
                    
            conn.close()
            self.logger.info(f"✅ Векторы экспортированы в {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка экспорта в текст: {e}")
            return False
            
    def convert_dimension(self, new_dimension: int, model_name: str = None) -> bool:
        """Конвертация векторов в новую размерность"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            self.logger.error("❌ sentence-transformers недоступен для конвертации размерности")
            return False
            
        try:
            # Загрузка новой модели
            if model_name:
                self.logger.info(f"🔄 Загрузка модели для размерности {new_dimension}: {model_name}")
                new_model = SentenceTransformer(model_name)
            else:
                # Выбор модели по размерности
                if new_dimension == 384:
                    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                elif new_dimension == 768:
                    model_name = "sentence-transformers/all-MiniLM-L6-v2"
                elif new_dimension == 1536:
                    model_name = "sentence-transformers/all-mpnet-base-v2"
                else:
                    self.logger.error(f"❌ Неизвестная размерность: {new_dimension}")
                    return False
                    
                self.logger.info(f"🔄 Загрузка модели: {model_name}")
                new_model = SentenceTransformer(model_name)
                
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Получение документов для переиндексации
            cursor.execute("""
                SELECT d.id, d.content
                FROM documents d
                INNER JOIN document_vectors dv ON d.id = dv.document_id
            """)
            
            documents = cursor.fetchall()
            
            if not documents:
                self.logger.warning("⚠️ Документы для переиндексации не найдены")
                return False
                
            # Переиндексация векторов
            updated_count = 0
            
            for doc_id, content in documents:
                try:
                    # Генерация нового вектора
                    new_embedding = new_model.encode(content)
                    
                    # Конвертация в BLOB
                    new_vector_blob = new_embedding.tobytes()
                    new_vector_hash = hashlib.md5(new_vector_blob).hexdigest()
                    
                    # Обновление в базе данных
                    cursor.execute("""
                        UPDATE document_vectors 
                        SET vector_data = ?, vector_hash = ?
                        WHERE document_id = ?
                    """, (new_vector_blob, new_vector_hash, doc_id))
                    
                    updated_count += 1
                    
                    if updated_count % 10 == 0:
                        self.logger.info(f"🔄 Обновлено {updated_count} векторов...")
                        
                except Exception as e:
                    self.logger.error(f"❌ Ошибка обновления вектора для документа {doc_id}: {e}")
                    
            conn.commit()
            conn.close()
            
            self.logger.info(f"✅ Конвертация завершена. Обновлено {updated_count} векторов")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка конвертации размерности: {e}")
            return False
            
    def restore_from_backup(self, backup_path: str = None) -> bool:
        """Восстановление базы данных из резервной копии"""
        try:
            backup_file = backup_path or self.backup_path
            
            if not os.path.exists(backup_file):
                self.logger.error(f"❌ Резервная копия {backup_file} не найдена")
                return False
                
            import shutil
            shutil.copy2(backup_file, self.db_path)
            self.logger.info(f"✅ База данных восстановлена из {backup_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка восстановления: {e}")
            return False

def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(description="Конвертер векторов Rubin AI v2.0")
    parser.add_argument("--db", default="rubin_ai_documents.db", help="Путь к базе данных")
    parser.add_argument("--action", choices=[
        "analyze", "csv", "json", "numpy", "text", "dimension", "backup", "restore"
    ], required=True, help="Действие для выполнения")
    parser.add_argument("--output", help="Файл для экспорта")
    parser.add_argument("--dimension", type=int, help="Новая размерность векторов")
    parser.add_argument("--model", help="Модель для конвертации размерности")
    parser.add_argument("--backup", help="Путь к резервной копии")
    
    args = parser.parse_args()
    
    converter = VectorConverter(args.db, args.backup)
    
    if args.action == "analyze":
        converter.analyze_vectors()
        
    elif args.action == "backup":
        converter.backup_database()
        
    elif args.action == "restore":
        converter.restore_from_backup(args.backup)
        
    elif args.action == "csv":
        output_file = args.output or "vectors_export.csv"
        converter.convert_vectors_to_csv(output_file)
        
    elif args.action == "json":
        output_file = args.output or "vectors_export.json"
        converter.convert_vectors_to_json(output_file)
        
    elif args.action == "numpy":
        output_file = args.output or "vectors_export.npz"
        converter.convert_vectors_to_numpy(output_file)
        
    elif args.action == "text":
        output_file = args.output or "vectors_export.txt"
        converter.convert_vectors_to_text(output_file)
        
    elif args.action == "dimension":
        if not args.dimension:
            print("❌ Требуется параметр --dimension")
            return
        converter.convert_dimension(args.dimension, args.model)

if __name__ == "__main__":
    main()
