#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Telegram бот для Rubin AI v2.0
Интеграция с Enhanced Smart Dispatcher
"""

import os
import logging
import requests
import json
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Конфигурация
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
DISPATCHER_URL = 'http://rubin-smart-dispatcher:8080'

class RubinTelegramBot:
    def __init__(self):
        self.dispatcher_url = DISPATCHER_URL
        
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /start"""
        welcome_message = """
🤖 Добро пожаловать в Rubin AI v2.0!

Я - интеллектуальный помощник с нейросетевой логикой.

Доступные команды:
/start - Начать работу
/help - Помощь
/status - Статус системы
/neural - Нейросетевой анализ

Просто напишите мне вопрос, и я найду лучший ответ!
        """
        await update.message.reply_text(welcome_message)
        
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /help"""
        help_text = """
📚 Справка по Rubin AI v2.0

🔹 Основные команды:
/start - Начать работу с ботом
/help - Показать эту справку
/status - Проверить статус системы
/neural <текст> - Нейросетевой анализ текста

🔹 Как использовать:
Просто напишите ваш вопрос на русском или английском языке. 
Система автоматически определит категорию и направит запрос к нужному модулю.

🔹 Поддерживаемые области:
• Электротехника
• Радиомеханика  
• Контроллеры
• Математика
• Программирование
• Физика
• Общие вопросы

Примеры вопросов:
"Как работает электричество?"
"Реши уравнение x^2 + 5x + 6 = 0"
"Объясни принцип работы контроллера"
        """
        await update.message.reply_text(help_text)
        
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /status"""
        try:
            response = requests.get(f"{self.dispatcher_url}/api/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                status_text = f"""
🟢 Система Rubin AI работает

📊 Статус: {data.get('status', 'unknown')}
🧠 Нейросеть: {data.get('neural_router', 'unknown')}
🔧 Обработчик ошибок: {data.get('error_tracker', 'unknown')}
⏰ Время: {data.get('timestamp', 'unknown')}
                """
            else:
                status_text = "🔴 Система недоступна"
        except Exception as e:
            status_text = f"❌ Ошибка подключения: {str(e)}"
            
        await update.message.reply_text(status_text)
        
    async def neural_analysis(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /neural"""
        if not context.args:
            await update.message.reply_text("Использование: /neural <текст для анализа>")
            return
            
        text = ' '.join(context.args)
        try:
            response = requests.post(
                f"{self.dispatcher_url}/api/neural/analyze",
                json={'message': text},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                analysis_text = f"""
🧠 Нейросетевой анализ:

📝 Текст: {data.get('message', '')}
🏷️ Категория: {data.get('category', 'unknown')}
🎯 Уверенность: {data.get('confidence', 0):.2f}
🔄 Маршрут: {data.get('suggested_server', 'unknown')}
📊 Уверенность маршрута: {data.get('route_confidence', 0):.2f}
⏰ Время: {data.get('timestamp', 'unknown')}
                """
            else:
                analysis_text = "❌ Ошибка нейросетевого анализа"
        except Exception as e:
            analysis_text = f"❌ Ошибка: {str(e)}"
            
        await update.message.reply_text(analysis_text)
        
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик обычных сообщений"""
        user_message = update.message.text
        
        # Показываем, что бот печатает
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
        
        try:
            # Отправляем запрос в диспетчер
            response = requests.post(
                f"{self.dispatcher_url}/api/chat",
                json={'message': user_message},
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Формируем ответ
                if data.get('error'):
                    reply_text = f"❌ {data['error']}\n\n"
                    reply_text += "💡 Попробуйте переформулировать вопрос или обратитесь к администратору."
                else:
                    # Извлекаем реальный ответ от сервера
                    server_response = data.get('response', {})
                    if isinstance(server_response, dict):
                        # Извлекаем content из response
                        content = server_response.get('content', '')
                        if content:
                            reply_text = f"🤖 {content}"
                        else:
                            # Fallback на explanation или title
                            explanation = server_response.get('explanation', '')
                            title = server_response.get('title', '')
                            if explanation:
                                reply_text = f"🤖 {explanation}"
                            elif title:
                                reply_text = f"🤖 {title}"
                            else:
                                # Если нет content, показываем информацию о маршрутизации
                                reply_text = f"✅ Запрос обработан успешно!\n\n"
                                if data.get('neural_analysis'):
                                    reply_text += f"🧠 Категория: {data.get('routed_to', 'unknown')}\n"
                                reply_text += f"📊 Уверенность: {data.get('confidence', 0):.2f}\n"
                                reply_text += f"⏰ Время обработки: {data.get('timestamp', 'unknown')}"
                    else:
                        # Если response не словарь, используем как есть
                        reply_text = f"🤖 {server_response}"
                    
            else:
                reply_text = "❌ Ошибка обработки запроса. Попробуйте позже."
                
        except requests.exceptions.Timeout:
            reply_text = "⏰ Превышено время ожидания. Попробуйте более простой вопрос."
        except Exception as e:
            logger.error(f"Ошибка обработки сообщения: {e}")
            reply_text = "❌ Произошла ошибка. Попробуйте позже."
            
        await update.message.reply_text(reply_text)
        
    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик ошибок"""
        logger.error(f"Ошибка: {context.error}")
        if update and update.effective_message:
            await update.effective_message.reply_text(
                "❌ Произошла ошибка. Попробуйте позже."
            )

def main():
    """Основная функция"""
    if not TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN не установлен!")
        return
        
    # Создаем экземпляр бота
    bot = RubinTelegramBot()
    
    # Создаем приложение
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Добавляем обработчики
    application.add_handler(CommandHandler("start", bot.start))
    application.add_handler(CommandHandler("help", bot.help_command))
    application.add_handler(CommandHandler("status", bot.status_command))
    application.add_handler(CommandHandler("neural", bot.neural_analysis))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_message))
    
    # Добавляем обработчик ошибок
    application.add_error_handler(bot.error_handler)
    
    # Запускаем бота
    logger.info("🤖 Telegram бот Rubin AI запущен")
    application.run_polling()

if __name__ == '__main__':
    main()