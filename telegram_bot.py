#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple Telegram bot for Rubin AI v2 (long polling).
Forwards user messages to Smart Dispatcher at http://localhost:8080/api/chat
and replies back with the dispatcher response.

Usage:
  set TELEGRAM_BOT_TOKEN in environment (Windows PowerShell example):
    $env:TELEGRAM_BOT_TOKEN = "123456:ABC..."
  run:
    python telegram_bot.py
"""

import os
import time
import json
import logging
from typing import Optional

import requests


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("telegram_bot")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
SMART_DISPATCHER_URL = "http://localhost:8080/api/chat"
ELECTRICAL_GRAPH_URL = "http://localhost:8087/api/graph/analyze"
ELECTRICAL_DIGITIZE_URL = "http://localhost:8087/api/graph/digitize"


def get_updates(offset: Optional[int] = None, timeout: int = 25):
    params = {"timeout": timeout}
    if offset:
        params["offset"] = offset
    r = requests.get(f"{TELEGRAM_API}/getUpdates", params=params, timeout=timeout + 5)
    r.raise_for_status()
    return r.json()


def send_message(chat_id: int, text: str):
    data = {"chat_id": chat_id, "text": text}
    r = requests.post(f"{TELEGRAM_API}/sendMessage", json=data, timeout=10)
    r.raise_for_status()
    return r.json()

def send_document(chat_id: int, file_path: str, caption: str = ""):
    """Отправляет документ в Telegram"""
    try:
        with open(file_path, 'rb') as f:
            files = {'document': f}
            data = {'chat_id': chat_id, 'caption': caption}
            r = requests.post(f"{TELEGRAM_API}/sendDocument", files=files, data=data, timeout=30)
            r.raise_for_status()
            return r.json()
    except Exception as e:
        logger.error(f"Ошибка отправки документа: {e}")
        return None


def send_long_message(chat_id: int, text: str):
    """Отправляет длинный текст, разбивая на части <= 4096 символов (безопасно: 3800)."""
    MAX_PART = 3800
    if not text:
        return
    parts = []
    s = str(text)
    while len(s) > MAX_PART:
        cut = s.rfind('\n', 0, MAX_PART)
        if cut == -1 or cut < MAX_PART * 0.5:
            cut = MAX_PART
        parts.append(s[:cut])
        s = s[cut:]
    if s:
        parts.append(s)
    for idx, p in enumerate(parts, 1):
        prefix = "" if len(parts) == 1 else f"[{idx}/{len(parts)}]\n"
        send_message(chat_id, prefix + p)


def get_file_url(file_id: str) -> Optional[str]:
    try:
        r = requests.get(f"{TELEGRAM_API}/getFile", params={"file_id": file_id}, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get("ok") and data.get("result", {}).get("file_path"):
            file_path = data["result"]["file_path"]
            return f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{file_path}"
    except Exception as e:
        logger.error(f"get_file_url error: {e}")
    return None


def analyze_graph_bytes(image_bytes: bytes) -> str:
    try:
        files = {"file": ("graph.jpg", image_bytes, "application/octet-stream")}
        r = requests.post(ELECTRICAL_GRAPH_URL, files=files, timeout=30)
        if r.status_code != 200:
            return f"Ошибка анализа графика: HTTP {r.status_code}"
        data = r.json()
        if not data.get("success"):
            return f"Ошибка анализа графика: {data.get('error','неизвестно')}"
        a = data.get("analysis", {})
        recs = data.get("recommendations", [])
        summary = (
            "Анализ графика:\n"
            f"• Размер: {a.get('image_width','?')}×{a.get('image_height','?')}\n"
            f"• Средняя яркость: {a.get('mean_intensity','?')}\n"
            f"• Разброс: {a.get('std_intensity','?')}\n"
            f"• Контраст контуров: {a.get('edge_strength','?')}\n"
            f"• Тренд (наклон): {a.get('trend_slope','?')}\n\n"
            "Рекомендации:\n- " + "\n- ".join(recs or ["нет рекомендаций"]) 
        )
        return summary
    except Exception as e:
        return f"Ошибка анализа графика: {e}"


def digitize_graph_bytes(image_bytes: bytes) -> str:
    try:
        files = {"file": ("graph.jpg", image_bytes, "application/octet-stream")}
        r = requests.post(ELECTRICAL_DIGITIZE_URL, files=files, timeout=45)
        if r.status_code != 200:
            return f"Ошибка оцифровки графика: HTTP {r.status_code}"
        data = r.json()
        if not data.get("success"):
            return f"Ошибка оцифровки графика: {data.get('error','неизвестно')}"
        pts = data.get("points", [])
        npts = data.get("normalized_points", [])
        summ = data.get("summary", {})
        txt = (
            "Оцифровка графика:\n"
            f"• Точек (px): {len(pts)}\n"
            f"• Точек (norm): {len(npts)}\n"
            f"• Наклон: {summ.get('trend_slope','?')}  σ: {summ.get('std','?')}  среднее: {summ.get('mean','?')}\n"
            f"• Пиков: {len(summ.get('peaks',[]))}, впадин: {len(summ.get('troughs',[]))}"
        )
        return txt
    except Exception as e:
        return f"Ошибка оцифровки графика: {e}"


def import_xml_graph(xml_bytes: bytes) -> str:
    """Импортирует XML файл с данными графика"""
    try:
        files = {"file": ("graph.xml", xml_bytes, "application/xml")}
        r = requests.post("http://localhost:8087/api/graph/import_xml", files=files, timeout=30)
        if r.status_code != 200:
            return f"Ошибка импорта XML: HTTP {r.status_code}"
        data = r.json()
        if not data.get("success"):
            return f"Ошибка импорта XML: {data.get('error','неизвестно')}"
        
        points = data.get("points", [])
        summary = data.get("summary", {})
        
        result = (
            "Импорт XML графика:\n"
            f"• Точек: {len(points)}\n"
            f"• X: мин={summary.get('x_min','?')}, макс={summary.get('x_max','?')}, среднее={summary.get('x_mean','?')}\n"
            f"• Y: мин={summary.get('y_min','?')}, макс={summary.get('y_max','?')}, среднее={summary.get('y_mean','?')}\n"
            f"• Тренд: наклон={summary.get('trend_slope','?')}, R²={summary.get('r_squared','?')}\n"
            f"• Стандартное отклонение: {summary.get('std_dev','?')}"
        )
        return result
    except Exception as e:
        return f"Ошибка импорта XML: {e}"


def ask_dispatcher(message: str) -> str:
    try:
        payload = {"message": message}
        r = requests.post(SMART_DISPATCHER_URL, json=payload, timeout=20)
        if r.status_code != 200:
            return f"Ошибка Smart Dispatcher: HTTP {r.status_code}"
        data = r.json()
        if data.get("success") and data.get("response"):
            return str(data["response"]).strip()
        # fallback
        return json.dumps(data, ensure_ascii=False)
    except Exception as e:
        return f"Ошибка соединения с Smart Dispatcher: {e}"


def main():
    if not TELEGRAM_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN не задан. Установите переменную окружения и запустите снова.")
        return

    logger.info("🚀 Telegram бот запущен (long polling). Нажмите Ctrl+C для выхода.")
    last_update_id = None

    while True:
        try:
            updates = get_updates(offset=last_update_id + 1 if last_update_id else None)
            if not updates.get("ok"):
                time.sleep(2)
                continue

            for upd in updates.get("result", []):
                last_update_id = upd["update_id"]
                msg = upd.get("message") or upd.get("edited_message")
                if not msg:
                    continue
                chat_id = msg["chat"]["id"]
                
                # Если прислали фото — берём самое большое превью
                if "photo" in msg and isinstance(msg["photo"], list) and msg["photo"]:
                    file_id = msg["photo"][-1]["file_id"]
                    url = get_file_url(file_id)
                    if not url:
                        send_message(chat_id, "Не удалось получить файл фото от Telegram")
                        continue
                    try:
                        img_resp = requests.get(url, timeout=30)
                        img_resp.raise_for_status()
                        caption = (msg.get("caption") or "").strip().lower()
                        if caption.startswith("/digitize") or caption.startswith("digitize"):
                            summary = digitize_graph_bytes(img_resp.content)
                        else:
                            summary = analyze_graph_bytes(img_resp.content)
                        send_long_message(chat_id, summary)
                    except Exception as e:
                        send_message(chat_id, f"Ошибка загрузки фото: {e}")
                    continue

                # Документ
                if "document" in msg and msg["document"]:
                    file_id = msg["document"]["file_id"]
                    url = get_file_url(file_id)
                    if not url:
                        send_message(chat_id, "Не удалось получить файл документа от Telegram")
                        continue
                    
                    try:
                        doc_resp = requests.get(url, timeout=30)
                        doc_resp.raise_for_status()
                        caption = (msg.get("caption") or "").strip().lower()
                        
                        # XML файл для импорта графика
                        if caption.startswith("/importxml") or caption.startswith("importxml") or caption.startswith("/import"):
                            summary = import_xml_graph(doc_resp.content)
                            send_long_message(chat_id, summary)
                        # Изображение
                        elif str(msg["document"].get("mime_type",""))[:5] == "image":
                            if caption.startswith("/digitize") or caption.startswith("digitize"):
                                summary = digitize_graph_bytes(doc_resp.content)
                            else:
                                summary = analyze_graph_bytes(doc_resp.content)
                            send_long_message(chat_id, summary)
                        else:
                            send_message(chat_id, "Неподдерживаемый тип файла. Отправьте изображение или XML файл.")
                    except Exception as e:
                        send_message(chat_id, f"Ошибка загрузки документа: {e}")
                    continue

                # Текст
                text = msg.get("text") or ""
                if text.strip():
                    if text.strip().lower() in ("/start", "start", "помощь", "/help"):
                        send_message(chat_id, "Rubin AI готов. Напишите вопрос или пришлите фото графика для анализа.\n\nКоманды директив:\n• прими директиву [текст]\n• список директив\n• статистика директив\n• помощь по директивам")
                        continue
                    
                    # Проверяем команды директив
                    if any(cmd in text.lower() for cmd in [
                        'прими директиву', 'список директив', 'удали директиву', 
                        'статистика директив', 'помощь по директивам'
                    ]):
                        user_id = str(msg.get("from", {}).get("id", "default"))
                        payload = {"message": text, "user_id": user_id}
                        r = requests.post(SMART_DISPATCHER_URL, json=payload, timeout=20)
                        if r.status_code == 200:
                            data = r.json()
                            if data.get("success"):
                                reply = data.get("message", "Команда выполнена")
                            else:
                                reply = data.get("error", "Ошибка выполнения команды")
                        else:
                            reply = f"Ошибка Smart Dispatcher: HTTP {r.status_code}"
                    else:
                        reply = ask_dispatcher(text)
                    send_long_message(chat_id, reply)

        except KeyboardInterrupt:
            logger.info("Остановка бота по запросу пользователя.")
            break
        except Exception as e:
            logger.error(f"Ошибка в основном цикле: {e}")
            time.sleep(2)


if __name__ == "__main__":
    main()


