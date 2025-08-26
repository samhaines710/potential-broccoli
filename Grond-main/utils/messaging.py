# messaging.py

"""
Messaging helpers for Telegram notifications.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Optional

from telegram import Bot, error as tg_error
from utils.logging_utils import write_status

# === Config & env vars ===
TELEGRAM_BOT_TOKEN        = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID          = os.getenv("TELEGRAM_CHAT_ID", "")
ENABLE_TELEGRAM           = os.getenv("ENABLE_TELEGRAM", "False").lower() in ("1", "true", "yes")
TELEGRAM_COOLDOWN_SECONDS = int(os.getenv("TELEGRAM_COOLDOWN_SECONDS", "60"))

# Instantiate Telegram bot if enabled
bot: Optional[Bot]
if ENABLE_TELEGRAM and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
else:
    bot = None

_last_telegram_ts = 0.0


def send_telegram(message: str) -> None:
    """
    Send a message via Telegram respecting a cooldown period.
    If the bot isn't configured or we're still in cooldown, this is a no-op.
    """
    global _last_telegram_ts
    now = time.time()

    if not bot:
        return

    if now - _last_telegram_ts < TELEGRAM_COOLDOWN_SECONDS:
        write_status("Skipping Telegram: cooldown active")
        return

    _last_telegram_ts = now

    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        loop.run_until_complete(
            bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
        )
        write_status(f"Sent Telegram: {message[:100]}")
    except tg_error.TimedOut:
        write_status("Telegram timed out; skipping")
    except tg_error.TelegramError as e:
        write_status(f"Telegram error: {e}")
