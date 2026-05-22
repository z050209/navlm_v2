"""VLM backend dispatch.

The synth pipeline calls a teacher VLM in two places:
  1. The big generation step (image + system + user → CoT + answer)
  2. The small verifier step (image + question → YES/NO/UNSURE)

This module abstracts the backend so swapping Gemma → GPT-5 / Claude is one
line in `call_teacher`. Local vLLM uses the OpenAI-compatible REST API.
"""

import json
import os
import sys
from pathlib import Path

import requests

# Allow `from synth_utils import ...` whether we're imported as a package
# (`toolbox.synth.backends`) or directly via sys.path manipulation.
_PARENT = str(Path(__file__).resolve().parent.parent)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
from synth_utils import img_to_data_url  # noqa: E402


# ---------- local vLLM (Gemma / Qwen / etc.) ----------

def call_local_vlm(image_path, sys_prompt, user_msg, vllm_url, model,
                   max_tokens=900, temperature=0.3, timeout=180):
    """Single-image, system-prompt + user-message call to a vLLM server."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": user_msg},
                {"type": "image_url",
                 "image_url": {"url": img_to_data_url(Path(image_path))}},
            ]},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    r = requests.post(f"{vllm_url}/chat/completions", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


# ---------- placeholders for future cloud backends ----------

def call_openai(image_path, sys_prompt, user_msg, model="gpt-5", **kw):
    """Placeholder for OpenAI Responses API. Implement when needed."""
    raise NotImplementedError(
        "OpenAI backend not yet wired. Set OPENAI_API_KEY and implement "
        "the Responses-API call here."
    )


def call_anthropic(image_path, sys_prompt, user_msg,
                    model="claude-4-6-sonnet-latest", **kw):
    """Placeholder for Anthropic Messages API. Implement when needed."""
    raise NotImplementedError(
        "Anthropic backend not yet wired. Set ANTHROPIC_API_KEY and "
        "implement the Messages-API call here."
    )


# ---------- Google Gemini (generativelanguage REST API) ----------

# gemini-2.5-flash list price, USD per 1M tokens (edit if Google changes it)
GEMINI_PRICE = {"gemini-2.5-flash": (0.30, 2.50),
                "gemini-2.5-pro": (1.25, 10.0)}
_COST_LEDGER = (Path(__file__).resolve().parents[2] / "costs" / "gemini_calls.jsonl")


def _log_gemini_cost(model, usage):
    """Append one call's token usage + USD cost to the spend ledger."""
    import datetime
    pin, pout = GEMINI_PRICE.get(model, (0.30, 2.50))
    p = usage.get("promptTokenCount", 0) or 0
    o = usage.get("candidatesTokenCount", 0) or 0
    cost = p / 1e6 * pin + o / 1e6 * pout
    _COST_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(_COST_LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "model": model, "prompt_tokens": p, "output_tokens": o,
            "cost_usd": round(cost, 6),
        }) + "\n")
    return cost


def call_gemini(image_path, sys_prompt, user_msg,
                model="gemini-2.5-flash", max_tokens=2048,
                temperature=0.3, timeout=180):
    """Single-image call to the Google Gemini API. Logs token cost to
    costs/gemini_calls.jsonl on every call.

    Key: GEMINI_API_KEY env > GOOGLE_API_KEY env > baked-in project key.
    """
    import base64

    key = (os.environ.get("GEMINI_API_KEY")
           or os.environ.get("GOOGLE_API_KEY") or "")
    if not key:
        raise RuntimeError("set GEMINI_API_KEY (see .env.example)")
    p = Path(image_path)
    mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    img_b64 = base64.b64encode(p.read_bytes()).decode()
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent?key={key}")
    payload = {
        "system_instruction": {"parts": [{"text": sys_prompt}]},
        "contents": [{"role": "user", "parts": [
            {"text": user_msg},
            {"inline_data": {"mime_type": mime, "data": img_b64}},
        ]}],
        "generationConfig": {"temperature": temperature,
                             "maxOutputTokens": max_tokens},
    }
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    _log_gemini_cost(model, data.get("usageMetadata", {}))
    cands = data.get("candidates", [])
    if not cands:
        raise RuntimeError(f"Gemini: no candidates ({data.get('promptFeedback')})")
    parts = cands[0].get("content", {}).get("parts", [])
    text = "".join(pt.get("text", "") for pt in parts).strip()
    if not text:
        raise RuntimeError("Gemini: empty response "
                           f"(finishReason={cands[0].get('finishReason')})")
    return text


# ---------- dispatch ----------

def call_teacher(backend, image_path, sys_prompt, user_msg, **kw):
    """Dispatch to the chosen backend. Same signature for all callers."""
    if backend == "gemma":
        return call_local_vlm(image_path, sys_prompt, user_msg, **kw)
    if backend == "openai":
        return call_openai(image_path, sys_prompt, user_msg, **kw)
    if backend == "anthropic":
        return call_anthropic(image_path, sys_prompt, user_msg, **kw)
    if backend == "gemini":
        return call_gemini(image_path, sys_prompt, user_msg, **kw)
    raise ValueError(f"unknown backend: {backend!r}")
