"""LLM abstraction — Ollama (local), Anthropic, or OpenAI-compatible APIs.

The orchestrator sends ONE pre-digested prompt and expects ONE JSON decision.
No tool calling, no multi-turn — keep it simple for 8B models.
"""

import json
import logging
import re
import time
from abc import ABC, abstractmethod

import requests

logger = logging.getLogger(__name__)


# ── JSON Parsing ────────────────────────────────────────────

def _extract_json(text: str) -> dict | None:
    """Extract a JSON decision object from LLM output.

    Handles: clean JSON, markdown fences, JSON mixed with commentary.
    """
    if not text or not text.strip():
        return None

    # Try 1: markdown code fence
    fence_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try 2: find JSON object with "decision" key
    for match in re.finditer(r'\{[^{}]*"decision"[^{}]*\}', text, re.DOTALL):
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            continue

    # Try 3: find any JSON object (outermost braces)
    brace_match = re.search(r'\{.*\}', text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    # Try 4: the whole text is JSON
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    return None


def _safe_decision(raw: dict | None, raw_text: str = "") -> dict:
    """Normalize a parsed decision or return a safe fallback."""
    if raw is None:
        return {
            "decision": "NO_TRADE",
            "reasoning": f"Could not parse JSON from LLM output: {raw_text[:200]}",
            "confidence": 0,
            "parse_error": True,
        }

    # Normalize decision field
    decision = str(raw.get("decision", "NO_TRADE")).upper().replace(" ", "_")
    if decision not in ("TRADE", "NO_TRADE"):
        decision = "NO_TRADE"
    raw["decision"] = decision

    # Ensure confidence is numeric
    try:
        raw["confidence"] = int(raw.get("confidence", 0))
    except (ValueError, TypeError):
        raw["confidence"] = 0

    return raw


# ── Base Class ──────────────────────────────────────────────

class LLMClient(ABC):
    """Send a digest prompt, get a parsed JSON decision."""

    @abstractmethod
    def _call(self, prompt: str) -> str:
        """Send prompt, return raw text response."""

    def decide(self, digest: str) -> dict:
        """Send digest, return normalized decision dict."""
        start = time.time()
        try:
            raw_text = self._call(digest)
            elapsed = time.time() - start
            logger.info("LLM responded in %.1fs (%d chars)", elapsed, len(raw_text))
        except Exception as e:
            elapsed = time.time() - start
            logger.error("LLM call failed after %.1fs: %s", elapsed, e)
            return {
                "decision": "NO_TRADE",
                "reasoning": f"LLM unavailable: {e}",
                "confidence": 0,
                "error": str(e),
            }

        parsed = _extract_json(raw_text)
        result = _safe_decision(parsed, raw_text)
        result["_raw_output"] = raw_text[:500]
        result["_elapsed_s"] = round(elapsed, 1)
        return result


# ── Ollama (local) ──────────────────────────────────────────

class OllamaClient(LLMClient):
    """Local Ollama inference — zero cost, runs on Orange Pi."""

    def __init__(self, url: str = "http://localhost:11434",
                 model: str = "0xroyce/plutus"):
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = 300  # 5 min — generous for slow hardware

    def _call(self, prompt: str) -> str:
        resp = requests.post(
            f"{self.url}/api/generate",
            json={
                "model": self.model,
                "system": (
                    "You are a quantitative 0DTE options trading analyst. "
                    "Analyze the market data provided and make a trading decision. "
                    "Respond with ONLY a valid JSON object, no commentary."
                ),
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,   # Low temp for consistent decisions
                    "num_predict": 512,    # Cap output tokens
                },
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "")

    def __repr__(self) -> str:
        return f"OllamaClient(model={self.model}, url={self.url})"


# ── Anthropic ───────────────────────────────────────────────

class AnthropicClient(LLMClient):
    """Claude API — best quality, ~$0.25-2/day."""

    def __init__(self, api_key: str, model: str = "claude-haiku-4-20250514"):
        self.api_key = api_key
        self.model = model

    def _call(self, prompt: str) -> str:
        try:
            import anthropic
        except ImportError:
            raise RuntimeError("pip install anthropic")

        client = anthropic.Anthropic(api_key=self.api_key)
        message = client.messages.create(
            model=self.model,
            max_tokens=512,
            temperature=0.3,
            system=(
                "You are a quantitative 0DTE options trading analyst. "
                "Analyze the market data provided and make a trading decision. "
                "Respond with ONLY a valid JSON object, no commentary."
            ),
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    def __repr__(self) -> str:
        return f"AnthropicClient(model={self.model})"


# ── OpenAI-compatible ───────────────────────────────────────

class OpenAICompatibleClient(LLMClient):
    """Works with OpenAI, Groq, Together, any compatible endpoint."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini",
                 base_url: str = "https://api.openai.com/v1"):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")

    def _call(self, prompt: str) -> str:
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 512,
                "response_format": {"type": "json_object"},
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def __repr__(self) -> str:
        return f"OpenAICompatibleClient(model={self.model}, url={self.base_url})"


# ── Factory ─────────────────────────────────────────────────

def get_llm_client(provider: str | None = None) -> LLMClient:
    """Create an LLM client from config."""
    from hybrid import config

    provider = provider or getattr(config, "LLM_PROVIDER", "ollama")

    if provider == "ollama":
        return OllamaClient(
            url=getattr(config, "OLLAMA_URL", "http://localhost:11434"),
            model=getattr(config, "OLLAMA_MODEL", "0xroyce/plutus"),
        )
    elif provider == "anthropic":
        key = getattr(config, "ANTHROPIC_API_KEY", "")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        return AnthropicClient(
            api_key=key,
            model=getattr(config, "CLAUDE_MODEL", "claude-haiku-4-20250514"),
        )
    elif provider == "openai":
        key = getattr(config, "OPENAI_API_KEY", "")
        if not key:
            raise ValueError("OPENAI_API_KEY not set")
        return OpenAICompatibleClient(
            api_key=key,
            model=getattr(config, "OPENAI_MODEL", "gpt-4o-mini"),
            base_url=getattr(config, "OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
