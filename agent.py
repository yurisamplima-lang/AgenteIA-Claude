import os
from openai import OpenAI
from services import CLINIC_INFO, SERVICES

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

# In-memory conversation history per phone number
_conversations: dict[str, list[dict]] = {}

MAX_HISTORY = 20  # max messages per conversation to avoid token overflow


def get_response(phone: str, user_message: str) -> str:
    if phone not in _conversations:
        _conversations[phone] = []

    history = _conversations[phone]
    history.append({"role": "user", "content": user_message})

    # Keep history bounded
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]
        _conversations[phone] = history

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.7,
        max_tokens=600,
    )

    assistant_message = response.choices[0].message.content
    history.append({"role": "assistant", "content": assistant_message})

    return assistant_message


def clear_history(phone: str) -> None:
    _conversations.pop(phone, None)
