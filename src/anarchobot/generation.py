from typing import List

import torch

from .model import TransformerLM
from .tokenizer import SentencePieceTokenizer
from .utils import get_device


def generate_text(
    prompt: str,
    model: TransformerLM,
    tokenizer: SentencePieceTokenizer,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 40,
) -> str:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        tokens = torch.tensor([tokenizer.encode(prompt)], device=device)
        out = model.generate(tokens, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
        return tokenizer.decode(out[0].tolist())


def chat_loop(model: TransformerLM, tokenizer: SentencePieceTokenizer, system_prompt: str = "You are AnarchoBot."):
    device = next(model.parameters()).device
    model.eval()
    history: List[str] = [system_prompt]
    print("Entering chat. Ctrl+C to exit.")
    while True:
        try:
            user_in = input("user> ")
        except KeyboardInterrupt:
            print("\nbye")
            break
        history.append(f"USER: {user_in}")
        prompt = "\n".join(history) + "\nASSISTANT:"
        tokens = torch.tensor([tokenizer.encode(prompt)], device=device)
        with torch.no_grad():
            out = model.generate(tokens, max_new_tokens=256, temperature=0.8, top_k=40)
        text = tokenizer.decode(out[0].tolist())
        assistant_reply = text[len(prompt) :]
        history.append(f"ASSISTANT: {assistant_reply.strip()}")
        print(f"bot> {assistant_reply.strip()}")
