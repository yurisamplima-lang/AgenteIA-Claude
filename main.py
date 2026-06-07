import os
import asyncio
import httpx
import logging
import redis.asyncio as aioredis
from fastapi import FastAPI, Request, HTTPException
from dotenv import load_dotenv
import agent
from agent import get_response, clear_history

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="OdontoSorriso WhatsApp Agent")

EVOLUTION_URL = os.getenv("EVOLUTION_URL", "").rstrip("/")
EVOLUTION_TOKEN = os.getenv("EVOLUTION_TOKEN", "")
EVOLUTION_INSTANCE = os.getenv("EVOLUTION_INSTANCE", "")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
DEBOUNCE_SECONDS = 10

redis_client: aioredis.Redis = None

# Tracks phones with an active debounce task in this process
_active_timers: set[str] = set()


@app.on_event("startup")
async def startup():
    global redis_client
    redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)
    agent.init_redis(REDIS_URL)


@app.on_event("shutdown")
async def shutdown():
    await redis_client.aclose()


def send_whatsapp_message(number: str, text: str) -> bool:
    url = f"{EVOLUTION_URL}/message/sendText/{EVOLUTION_INSTANCE}"
    headers = {"apikey": EVOLUTION_TOKEN, "Content-Type": "application/json"}
    payload = {"number": number, "text": text}

    try:
        with httpx.Client(timeout=15) as http:
            resp = http.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            log.info(f"Message sent to {number}")
            return True
    except httpx.HTTPError as e:
        log.error(f"Failed to send message to {number}: {e}")
        return False


def extract_message_data(data: dict) -> tuple[str | None, str | None, str | None]:
    """Returns (phone, message_text, sender_name) or (None, None, None) if not a text message."""
    try:
        key = data.get("key", {})
        if key.get("fromMe"):
            return None, None, None

        remote_jid = key.get("remoteJid", "")
        if "@g.us" in remote_jid:
            return None, None, None

        phone = remote_jid.replace("@s.whatsapp.net", "")
        sender_name = data.get("pushName", "")

        message = data.get("message", {})
        text = (
            message.get("conversation")
            or message.get("extendedTextMessage", {}).get("text")
            or message.get("imageMessage", {}).get("caption")
        )

        return phone, text, sender_name
    except Exception:
        return None, None, None


async def debounce_worker(phone: str):
    """Poll until timer:{phone} expires, then process all queued messages at once."""
    while True:
        await asyncio.sleep(1)
        ttl = await redis_client.ttl(f"timer:{phone}")
        if ttl < 0:  # -2 = key gone (expired), -1 = no TTL (shouldn't happen)
            break

    messages = await redis_client.lrange(f"msgs:{phone}", 0, -1)
    await redis_client.delete(f"msgs:{phone}", f"timer:{phone}")
    _active_timers.discard(phone)

    if not messages:
        return

    combined = "\n".join(messages)
    log.info(f"Processing {len(messages)} queued message(s) from {phone}: {combined[:120]}")

    response_text = await get_response(phone, combined)
    send_whatsapp_message(phone, response_text)


@app.post("/webhook")
async def webhook(request: Request):
    if os.getenv("AGENT_ENABLED", "true").lower() == "false":
        return {"status": "disabled"}

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    event = body.get("event", "")
    if event not in ("messages.upsert", "messages.update"):
        return {"status": "ignored", "event": event}

    message_data = body.get("data", {})
    phone, text, name = extract_message_data(message_data)

    if not phone or not text:
        return {"status": "ignored", "reason": "no text or fromMe"}

    log.info(f"Message from {name} ({phone}): {text[:80]}")

    if text.strip().lower() in ("/reset", "reiniciar", "reset"):
        await redis_client.delete(f"msgs:{phone}", f"timer:{phone}")
        _active_timers.discard(phone)
        await clear_history(phone)
        send_whatsapp_message(phone, "Conversa reiniciada! Como posso te ajudar? 😊")
        return {"status": "ok"}

    await redis_client.rpush(f"msgs:{phone}", text)
    await redis_client.set(f"timer:{phone}", "1", ex=DEBOUNCE_SECONDS)

    if phone not in _active_timers:
        _active_timers.add(phone)
        asyncio.create_task(debounce_worker(phone))

    return {"status": "queued"}


@app.get("/health")
def health():
    return {"status": "online", "agent": "OdontoSorriso"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
