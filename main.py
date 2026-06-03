import os
import httpx
import logging
from fastapi import FastAPI, Request, HTTPException
from dotenv import load_dotenv
from agent import get_response, clear_history

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="OdontoSorriso WhatsApp Agent")

EVOLUTION_URL = os.getenv("EVOLUTION_URL", "").rstrip("/")
EVOLUTION_TOKEN = os.getenv("EVOLUTION_TOKEN", "")
EVOLUTION_INSTANCE = os.getenv("EVOLUTION_INSTANCE", "")


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
        # Skip group messages
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


@app.post("/webhook")
async def webhook(request: Request):
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

    # Reset conversation command
    if text.strip().lower() in ("/reset", "reiniciar", "reset"):
        clear_history(phone)
        send_whatsapp_message(phone, "Conversa reiniciada! Como posso te ajudar? 😊")
        return {"status": "ok"}

    response_text = get_response(phone, text)
    send_whatsapp_message(phone, response_text)

    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "online", "agent": "OdontoSorriso"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
