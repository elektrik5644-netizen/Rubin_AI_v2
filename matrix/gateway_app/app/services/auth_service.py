"""
Сервис аутентификации и авторизации
"""

import jwt
import hashlib
import logging
import time
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from app.core.config import settings

logger = logging.getLogger(__name__)

class AuthService:
    """Сервис аутентификации"""
    
    def __init__(self):
        self.secret_key = settings.JWT_SECRET
        self.algorithm = settings.JWT_ALGORITHM
        self.expire_minutes = settings.JWT_EXPIRE_MINUTES
    
    def hash_password(self, password: str) -> str:
        """Хеширование пароля"""
        # В реальной реализации используйте bcrypt
        return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Проверка пароля"""
        return self.hash_password(plain_password) == hashed_password
    
    def create_access_token(self, data: Dict[str, Any]) -> str:
        """Создание JWT токена"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=self.expire_minutes)
        to_encode.update({"exp": expire})
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Проверка JWT токена"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.JWTError:
            logger.warning("Invalid token")
            return None
    
    async def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Аутентификация пользователя"""
        try:
            # В реальной реализации здесь будет запрос к базе данных
            # Пока используем заглушку
            if username == "admin" and password == "admin":
                return {
                    "id": 1,
                    "username": "admin",
                    "email": "admin@rubin-matrix.com",
                    "role": "admin",
                    "created_at": "2025-01-01T00:00:00Z"
                }
            elif username == "user" and password == "user":
                return {
                    "id": 2,
                    "username": "user",
                    "email": "user@rubin-matrix.com",
                    "role": "user",
                    "created_at": "2025-01-01T00:00:00Z"
                }
            else:
                return None
                
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return None
    
    async def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Получение пользователя по имени"""
        try:
            # В реальной реализации здесь будет запрос к базе данных
            if username == "admin":
                return {
                    "id": 1,
                    "username": "admin",
                    "email": "admin@rubin-matrix.com",
                    "role": "admin"
                }
            elif username == "user":
                return {
                    "id": 2,
                    "username": "user",
                    "email": "user@rubin-matrix.com",
                    "role": "user"
                }
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error getting user: {e}")
            return None
    
    async def create_user(
        self, 
        username: str, 
        email: str, 
        password: str, 
        role: str = "user"
    ) -> int:
        """Создание пользователя"""
        try:
            # В реальной реализации здесь будет запрос к базе данных
            hashed_password = self.hash_password(password)
            
            # Имитация создания пользователя
            user_id = int(time.time()) % 10000
            
            logger.info(f"Created user {username} with ID {user_id}")
            return user_id
            
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            raise
    
    async def get_current_user(self, token: str) -> Optional[Dict[str, Any]]:
        """Получение текущего пользователя по токену"""
        try:
            payload = self.verify_token(token)
            if not payload:
                return None
            
            username = payload.get("sub")
            if not username:
                return None
            
            return await self.get_user_by_username(username)
            
        except Exception as e:
            logger.error(f"Error getting current user: {e}")
            return None
    
    async def logout_user(self, token: str):
        """Выход пользователя"""
        try:
            # В реальной реализации здесь будет добавление токена в черный список
            logger.info("User logged out successfully")
            
        except Exception as e:
            logger.error(f"Error during logout: {e}")
    
    async def refresh_token(self, token: str) -> Optional[str]:
        """Обновление токена"""
        try:
            payload = self.verify_token(token)
            if not payload:
                return None
            
            # Создание нового токена
            new_data = {
                "sub": payload.get("sub"),
                "user_id": payload.get("user_id")
            }
            
            return self.create_access_token(new_data)
            
        except Exception as e:
            logger.error(f"Error refreshing token: {e}")
            return None
