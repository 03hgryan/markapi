from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import base64
import anthropic
from openai import OpenAI

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

claude_client = anthropic.Anthropic()
openai_client = OpenAI()

SYSTEM_PROMPTS = {
    "normal": "You are a helpful assistant.",
    "problem_solve": (
        "You are a problem-solving assistant. "
        "When given a problem, always respond in this exact format:\n\n"
        "**Answer:** [Your concise final answer]\n\n"
        "**Reasoning:**\n[Your step-by-step reasoning]\n\n"
        "Always lead with the answer first, then explain the reasoning."
    ),
    "image": (
        "You are a visual analysis assistant. "
        "Analyze the provided image and answer the user's question concisely and accurately."
    ),
}


TRANSLATE_SYSTEM_PROMPT = (
    "You are a translation assistant. "
    "Translate the given text exactly as requested. "
    "Respond with only the translated text, nothing else."
)


def build_system_prompt(mode: str, max_tokens: int) -> str:
    base = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["normal"])
    return f"{base}\n\nYour response must fit within {max_tokens} tokens."


def strip_data_url(data_url: str) -> tuple[str, str]:
    """Return (media_type, base64_data) from a data URL like data:image/png;base64,..."""
    header, data = data_url.split(",", 1)
    media_type = header.split(":")[1].split(";")[0]  # e.g. image/png
    return media_type, data


class TranslateRequest(BaseModel):
    text: str


class ClaudeRequest(BaseModel):
    text: str
    model: str = "claude-opus-4-6"
    mode: str = "normal"
    max_tokens: int = 150
    image: Optional[str] = None  # base64 data URL


class GPTRequest(BaseModel):
    text: str
    model: str = "gpt-4o"
    mode: str = "normal"
    max_tokens: int = 150
    image: Optional[str] = None  # base64 data URL


@app.get("/")
async def root():
    return "hi"


@app.post("/translate")
async def translate(body: TranslateRequest):
    print(f"Translate request: {body}")
    response = openai_client.responses.create(
        model="gpt-4o-mini",
        instructions=TRANSLATE_SYSTEM_PROMPT,
        input=body.text,
    )
    print("translation:", response.output[0].content[0].text)
    return {"answer": response.output[0].content[0].text}


@app.post("/claude")
async def ask_claude(body: ClaudeRequest):
    print(f"Received request: {body}")

    if body.image:
        media_type, b64_data = strip_data_url(body.image)
        content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": b64_data,
                },
            },
            {"type": "text", "text": body.text},
        ]
    else:
        content = body.text

    message = claude_client.messages.create(
        model=body.model,
        max_tokens=body.max_tokens,
        system=build_system_prompt(body.mode, body.max_tokens),
        messages=[{"role": "user", "content": content}],
    )
    print("answer:", message.content[0].text)
    return {"answer": message.content[0].text}


@app.post("/gpt")
async def ask_gpt(body: GPTRequest):
    print(f"Received request: {body}")

    if body.image:
        input_payload = [
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": body.image},
                    {"type": "input_text", "text": body.text},
                ],
            }
        ]
    else:
        input_payload = body.text

    response = openai_client.responses.create(
        model=body.model,
        instructions=build_system_prompt(body.mode, body.max_tokens),
        input=input_payload,
        max_output_tokens=body.max_tokens,
    )
    print("answer:", response.output[0].content[0].text)
    return {"answer": response.output[0].content[0].text}
