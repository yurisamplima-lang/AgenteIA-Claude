import os
import json
import redis.asyncio as aioredis
from openai import OpenAI
from services import CLINIC_INFO, SERVICES

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

_redis: aioredis.Redis = None

COMPRESS_AT = 8    # compress when history reaches this size
KEEP_RECENT = 2    # messages kept after compression

SYSTEM_PROMPT = f"""Você é a Sorriso, assistente virtual da {CLINIC_INFO['nome']}.
Seu papel é atender pacientes pelo WhatsApp com simpatia, clareza e agilidade.

INFORMAÇÕES DA CLÍNICA:
- Telefone: {CLINIC_INFO['telefone']}
- Endereço: {CLINIC_INFO['endereco']}
- Horário de atendimento: {CLINIC_INFO['horarios']}
- E-mail: {CLINIC_INFO['email']}

{SERVICES}

SUAS RESPONSABILIDADES:
1. Informar serviços e valores com clareza
2. Ajudar pacientes a agendar consultas (solicite nome, telefone e serviço desejado)
3. Responder dúvidas sobre procedimentos de forma simples e acessível
4. Encaminhar urgências dentárias para contato direto com a clínica
5. Confirmar/reagendar consultas quando solicitado

REGRAS DE COMPORTAMENTO:
- Seja sempre simpática, acolhedora e profissional
- Use linguagem simples, evite jargões técnicos desnecessários
- Respostas curtas e objetivas (máximo 3-4 parágrafos)
- Nunca faça diagnósticos — oriente sempre a agendar uma avaliação
- Se não souber a resposta, diga que vai verificar e peça para aguardar
- Para agendamento: colete nome, telefone e serviço/queixa principal
- Dores ou emergências: oriente a ligar imediatamente para {CLINIC_INFO['telefone']}
- Sempre termine com uma pergunta ou chamada para ação quando pertinente

FORMATO DAS RESPOSTAS:
- Use emojis com moderação para deixar a conversa mais amigável 😊
- Para listas de valores, use formatação clara
- Não use markdown pesado (asteriscos, hashtags) pois aparece feio no WhatsApp

Lembre-se: você representa a clínica. Seja a melhor primeira impressão que o paciente terá!"""

_COMPRESS_PROMPT = """\
Resuma em português a conversa abaixo de forma concisa (máximo 200 palavras), preservando:
- Nome e dados do paciente mencionados
- Serviços ou procedimentos de interesse
- Agendamentos ou compromissos combinados
- Contexto relevante para continuar o atendimento

{previous}
Conversa recente:
{conversation}"""


def init_redis(url: str) -> None:
    global _redis
    _redis = aioredis.from_url(url, decode_responses=True)


async def _load_history(phone: str) -> list[dict]:
    raw = await _redis.get(f"history:{phone}")
    return json.loads(raw) if raw else []


async def _save_history(phone: str, history: list[dict]) -> None:
    await _redis.set(f"history:{phone}", json.dumps(history, ensure_ascii=False))


async def _load_summary(phone: str) -> str:
    return await _redis.get(f"summary:{phone}") or ""


async def _save_summary(phone: str, summary: str) -> None:
    await _redis.set(f"summary:{phone}", summary)


async def _compress(phone: str, history: list[dict]) -> str:
    """Compress history[:-KEEP_RECENT] into a summary, merging with any existing summary."""
    previous_summary = await _load_summary(phone)
    conversation_text = "\n".join(
        f"{'Paciente' if m['role'] == 'user' else 'Sorriso'}: {m['content']}"
        for m in history[:-KEEP_RECENT]
    )
    prompt = _COMPRESS_PROMPT.format(
        previous=f"Resumo anterior:\n{previous_summary}\n\n" if previous_summary else "",
        conversation=conversation_text,
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=300,
    )
    return response.choices[0].message.content


async def get_response(phone: str, user_message: str) -> str:
    history = await _load_history(phone)
    summary = await _load_summary(phone)

    history.append({"role": "user", "content": user_message})

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if summary:
        messages.append({"role": "system", "content": f"Resumo do atendimento anterior:\n{summary}"})
    messages.extend(history)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.7,
        max_tokens=600,
    )

    assistant_message = response.choices[0].message.content
    history.append({"role": "assistant", "content": assistant_message})

    if len(history) >= COMPRESS_AT:
        new_summary = await _compress(phone, history)
        await _save_summary(phone, new_summary)
        history = history[-KEEP_RECENT:]

    await _save_history(phone, history)
    return assistant_message


async def clear_history(phone: str) -> None:
    await _redis.delete(f"history:{phone}", f"summary:{phone}")
