"""LLM API abstraction — OpenAI-compatible (MindRouter) and Anthropic providers.

Supports a two-pass approach for reasoning models:
  Pass 1: Let the model reason freely (unstructured)
  Pass 2: Feed reasoning back and extract structured JSON output
"""

import base64
import json
import os
import time

import anthropic
import httpx


def call_llm(model_config, messages):
    """Call an LLM and return (raw_response_json, response_text, latency_ms).

    Messages follow OpenAI format: [{"role": ..., "content": ...}]
    For vision models, content can be a list of content blocks.

    Returns both content and reasoning_content in the raw response.
    The response_text is the content field (or reasoning_content as fallback).
    """
    provider = model_config["provider"]
    if provider == "openai_compat":
        return _call_openai_compat(model_config, messages)
    elif provider == "anthropic":
        return _call_anthropic(model_config, messages)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def call_llm_two_pass(model_config, messages, extraction_prompt_fn,
                      extraction_model_config=None):
    """Two-pass LLM call: reasoning then extraction.

    Pass 1: Send messages, get unstructured reasoning (uses model_config).
    Pass 2: Send extraction prompt with the reasoning, get structured output.
            Uses extraction_model_config if provided, otherwise model_config.

    Args:
        model_config: Model configuration dict for pass 1 (reasoning)
        messages: Original prompt messages
        extraction_prompt_fn: Callable(reasoning_text) -> messages for pass 2
        extraction_model_config: Optional separate model config for pass 2.
            Use a stronger model here to reduce parse errors.

    Returns: (raw_pass1, reasoning_text, raw_pass2, extracted_text, total_latency_ms)
    """
    # Pass 1: reasoning
    raw1, text1, latency1 = call_llm(model_config, messages)

    # Get full reasoning (may be in reasoning_content for reasoning models)
    raw1_data = json.loads(raw1) if isinstance(raw1, str) else raw1
    msg = raw1_data.get("choices", [{}])[0].get("message", {})
    reasoning = msg.get("reasoning_content") or text1 or ""

    # Pass 2: extraction (optionally with a different, stronger model)
    pass2_config = extraction_model_config or model_config
    extraction_messages = extraction_prompt_fn(reasoning)
    raw2, text2, latency2 = call_llm(pass2_config, extraction_messages)

    return raw1, reasoning, raw2, text2, latency1 + latency2


def _get_api_key(config):
    env_var = config.get("api_key_env")
    if not env_var:
        return None
    key = os.environ.get(env_var)
    if not key:
        raise RuntimeError(f"Environment variable {env_var} not set")
    return key


def _call_openai_compat(config, messages):
    api_key = _get_api_key(config)
    endpoint = config["endpoint"].rstrip("/")
    url = f"{endpoint}/chat/completions"

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    body = {
        "model": config["model_id"],
        "messages": messages,
        "temperature": config.get("temperature", 0.0),
    }
    if "max_tokens" in config:
        body["max_tokens"] = config["max_tokens"]

    start = time.monotonic()
    with httpx.Client(timeout=config.get("timeout", 300.0)) as client:
        resp = client.post(url, json=body, headers=headers)
        resp.raise_for_status()
    latency_ms = int((time.monotonic() - start) * 1000)

    raw = resp.text
    data = resp.json()
    msg = data["choices"][0]["message"]
    # Some models (e.g. reasoning models) put output in reasoning_content with null content
    text = msg.get("content") or msg.get("reasoning_content") or ""
    return raw, text, latency_ms


def _call_anthropic(config, messages):
    api_key = _get_api_key(config)
    client = anthropic.Anthropic(api_key=api_key)

    # Convert OpenAI-style messages to Anthropic format
    system_text = None
    anthropic_messages = []
    for msg in messages:
        if msg["role"] == "system":
            system_text = msg["content"]
        else:
            anthropic_messages.append(msg)

    kwargs = {
        "model": config["model_id"],
        "messages": anthropic_messages,
        "max_tokens": config.get("max_tokens", 4096),
        "temperature": config.get("temperature", 0.0),
    }
    if system_text:
        kwargs["system"] = system_text

    start = time.monotonic()
    response = client.messages.create(**kwargs)
    latency_ms = int((time.monotonic() - start) * 1000)

    raw = response.model_dump_json()
    text = response.content[0].text
    return raw, text, latency_ms


def make_image_content_block(image_bytes, media_type="image/png"):
    """Create an OpenAI-compatible image_url content block from raw bytes."""
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{media_type};base64,{b64}"},
    }
