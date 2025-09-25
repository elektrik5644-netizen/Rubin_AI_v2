"""
Сервис для работы с Qdrant векторной базой данных
"""

import httpx
import logging
import time
from typing import Dict, List, Optional, Any
from app.core.config import settings

logger = logging.getLogger(__name__)

class QdrantService:
    """Сервис для работы с Qdrant"""
    
    def __init__(self):
        self.base_url = settings.qdrant_url
        self.collection_name = "rubin_documents"
        self.vector_size = 384  # Размер вектора для sentence-transformers
        self.timeout = 10.0
    
    async def health_check(self) -> Dict[str, Any]:
        """Проверка здоровья Qdrant"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/health",
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    return {
                        "status": "healthy",
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "error": f"HTTP {response.status_code}",
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
                    }
                    
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
            }
    
    async def create_collection(self, collection_name: str = None) -> bool:
        """Создание коллекции"""
        try:
            collection_name = collection_name or self.collection_name
            
            collection_config = {
                "vectors": {
                    "size": self.vector_size,
                    "distance": "Cosine"
                }
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{self.base_url}/collections/{collection_name}",
                    json=collection_config,
                    timeout=self.timeout
                )
                
                if response.status_code in [200, 201]:
                    logger.info(f"Collection {collection_name} created successfully")
                    return True
                else:
                    logger.error(f"Error creating collection: {response.status_code}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            return False
    
    async def list_collections(self) -> List[str]:
        """Получение списка коллекций"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/collections",
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return [col["name"] for col in result.get("collections", [])]
                else:
                    logger.error(f"Error listing collections: {response.status_code}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return []
    
    async def add_documents(self, documents: List[Dict[str, Any]], collection_name: str = None) -> bool:
        """Добавление документов в коллекцию"""
        try:
            collection_name = collection_name or self.collection_name
            
            # Подготовка точек для добавления
            points = []
            for i, doc in enumerate(documents):
                point = {
                    "id": doc.get("id", i),
                    "vector": doc.get("vector", []),
                    "payload": {
                        "title": doc.get("title", ""),
                        "content": doc.get("content", ""),
                        "source": doc.get("source", ""),
                        "metadata": doc.get("metadata", {}),
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
                    }
                }
                points.append(point)
            
            # Добавление точек
            upsert_data = {
                "points": points
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{self.base_url}/collections/{collection_name}/points",
                    json=upsert_data,
                    timeout=self.timeout
                )
                
                if response.status_code in [200, 201]:
                    logger.info(f"Added {len(documents)} documents to collection {collection_name}")
                    return True
                else:
                    logger.error(f"Error adding documents: {response.status_code}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False
    
    async def search_similar(
        self, 
        query: str, 
        limit: int = 5, 
        collection_name: str = None,
        score_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Поиск похожих документов"""
        try:
            collection_name = collection_name or self.collection_name
            
            # Векторизация запроса (заглушка - в реальности нужен sentence-transformer)
            query_vector = await self._vectorize_text(query)
            
            search_data = {
                "vector": query_vector,
                "limit": limit,
                "with_payload": True,
                "score_threshold": score_threshold
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/collections/{collection_name}/points/search",
                    json=search_data,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    documents = []
                    
                    for point in result.get("result", []):
                        doc = {
                            "id": point.get("id"),
                            "score": point.get("score", 0),
                            "title": point.get("payload", {}).get("title", ""),
                            "content": point.get("payload", {}).get("content", ""),
                            "source": point.get("payload", {}).get("source", ""),
                            "metadata": point.get("payload", {}).get("metadata", {}),
                            "timestamp": point.get("payload", {}).get("timestamp", "")
                        }
                        documents.append(doc)
                    
                    logger.info(f"Found {len(documents)} similar documents")
                    return documents
                else:
                    logger.error(f"Error searching documents: {response.status_code}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    async def get_document(self, document_id: str, collection_name: str = None) -> Optional[Dict[str, Any]]:
        """Получение документа по ID"""
        try:
            collection_name = collection_name or self.collection_name
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/collections/{collection_name}/points/{document_id}",
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    point = result.get("result", {})
                    
                    return {
                        "id": point.get("id"),
                        "title": point.get("payload", {}).get("title", ""),
                        "content": point.get("payload", {}).get("content", ""),
                        "source": point.get("payload", {}).get("source", ""),
                        "metadata": point.get("payload", {}).get("metadata", {}),
                        "timestamp": point.get("payload", {}).get("timestamp", "")
                    }
                else:
                    logger.error(f"Error getting document: {response.status_code}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting document: {e}")
            return None
    
    async def delete_document(self, document_id: str, collection_name: str = None) -> bool:
        """Удаление документа"""
        try:
            collection_name = collection_name or self.collection_name
            
            delete_data = {
                "points": [document_id]
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/collections/{collection_name}/points/delete",
                    json=delete_data,
                    timeout=self.timeout
                )
                
                if response.status_code in [200, 202]:
                    logger.info(f"Document {document_id} deleted successfully")
                    return True
                else:
                    logger.error(f"Error deleting document: {response.status_code}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False
    
    async def get_collection_info(self, collection_name: str = None) -> Dict[str, Any]:
        """Получение информации о коллекции"""
        try:
            collection_name = collection_name or self.collection_name
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/collections/{collection_name}",
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Error getting collection info: {response.status_code}")
                    return {}
                    
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}
    
    async def _vectorize_text(self, text: str) -> List[float]:
        """Векторизация текста (заглушка)"""
        # В реальной реализации здесь должен быть sentence-transformer
        # Пока возвращаем случайный вектор
        import random
        random.seed(hash(text))
        return [random.random() for _ in range(self.vector_size)]
    
    async def batch_vectorize_documents(self, documents: List[str]) -> List[List[float]]:
        """Пакетная векторизация документов"""
        vectors = []
        for doc in documents:
            vector = await self._vectorize_text(doc)
            vectors.append(vector)
        return vectors
    
    async def update_document(self, document_id: str, content: str, metadata: Dict = None, collection_name: str = None) -> bool:
        """Обновление документа"""
        try:
            collection_name = collection_name or self.collection_name
            
            # Векторизация нового контента
            vector = await self._vectorize_text(content)
            
            point = {
                "id": document_id,
                "vector": vector,
                "payload": {
                    "content": content,
                    "metadata": metadata or {},
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
                }
            }
            
            upsert_data = {
                "points": [point]
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.put(
                    f"{self.base_url}/collections/{collection_name}/points",
                    json=upsert_data,
                    timeout=self.timeout
                )
                
                if response.status_code in [200, 201]:
                    logger.info(f"Document {document_id} updated successfully")
                    return True
                else:
                    logger.error(f"Error updating document: {response.status_code}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error updating document: {e}")
            return False
