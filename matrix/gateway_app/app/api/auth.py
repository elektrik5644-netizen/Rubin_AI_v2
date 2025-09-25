"""
API для аутентификации и авторизации
"""

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional
import logging
import time

from app.services.auth_service import AuthService

logger = logging.getLogger(__name__)
router = APIRouter()
security = HTTPBearer()

class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    user_id: int
    username: str
    role: str

class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str
    role: Optional[str] = "user"

@router.post("/login", response_model=LoginResponse)
async def login(login_request: LoginRequest):
    """Аутентификация пользователя"""
    try:
        auth_service = AuthService()
        
        # Проверка учетных данных
        user = await auth_service.authenticate_user(
            login_request.username, 
            login_request.password
        )
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Неверные учетные данные"
            )
        
        # Создание токена
        access_token = auth_service.create_access_token(
            data={"sub": user["username"], "user_id": user["id"]}
        )
        
        return LoginResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=1800,  # 30 минут
            user_id=user["id"],
            username=user["username"],
            role=user["role"]
        )
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ошибка аутентификации"
        )

@router.post("/register")
async def register(register_request: RegisterRequest):
    """Регистрация нового пользователя"""
    try:
        auth_service = AuthService()
        
        # Проверка существования пользователя
        existing_user = await auth_service.get_user_by_username(register_request.username)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Пользователь с таким именем уже существует"
            )
        
        # Создание пользователя
        user_id = await auth_service.create_user(
            username=register_request.username,
            email=register_request.email,
            password=register_request.password,
            role=register_request.role
        )
        
        return {
            "message": "Пользователь успешно зарегистрирован",
            "user_id": user_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ошибка регистрации"
        )

@router.get("/me")
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Получение информации о текущем пользователе"""
    try:
        auth_service = AuthService()
        
        # Проверка токена
        user = await auth_service.get_current_user(credentials.credentials)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Недействительный токен"
            )
        
        return {
            "user_id": user["id"],
            "username": user["username"],
            "email": user["email"],
            "role": user["role"],
            "created_at": user["created_at"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get current user error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ошибка получения информации о пользователе"
        )

@router.post("/logout")
async def logout(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Выход из системы"""
    try:
        auth_service = AuthService()
        
        # Добавление токена в черный список (если используется)
        await auth_service.logout_user(credentials.credentials)
        
        return {"message": "Успешный выход из системы"}
        
    except Exception as e:
        logger.error(f"Logout error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ошибка выхода из системы"
        )

@router.post("/refresh")
async def refresh_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Обновление токена"""
    try:
        auth_service = AuthService()
        
        # Проверка токена и создание нового
        new_token = await auth_service.refresh_token(credentials.credentials)
        
        if not new_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Недействительный токен"
            )
        
        return {
            "access_token": new_token,
            "token_type": "bearer",
            "expires_in": 1800
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Refresh token error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ошибка обновления токена"
        )
