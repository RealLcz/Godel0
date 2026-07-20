# This file is adapted from https://github.com/jennyzzt/dgm.
# Extended for Godel0 with multi-model support (Qwen, Minimax, DeepSeek, OpenAI, Anthropic).

import json
import os
import re

import anthropic
import backoff
import openai

MAX_OUTPUT_TOKENS = 4096

AVAILABLE_LLMS = [
    "gpt-5",
    "o4-mini",
    "o3",
    "deepseek/deepseek-chat-v3.1",
    "deepseek/deepseek-chat",
    "deepseek-chat",
    "deepseek-reasoner",
    "anthropic/claude-sonnet-4",
    "qwen/qwen-coder-32b",
    "qwen/qwen2.5-coder-32b",
    "minimax/abab6.5s-chat",
    "minimax/MiniMax-Text-01",
]

VLLM_HOST = os.getenv("VLLM_HOST", "127.0.0.1")
VLLM_PORT = os.getenv("VLLM_PORT", "8000")

DEEPSEEK_API_BASE_URL = os.getenv("DEEPSEEK_API_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MAX_OUTPUT_TOKENS = int(os.environ.get("DEEPSEEK_MAX_OUTPUT_TOKENS", "16384"))
DEEPSEEK_API_TIMEOUT = float(os.environ.get("DEEPSEEK_API_TIMEOUT", "240"))

MINIMAX_API_BASE_URL = os.getenv("MINIMAX_API_BASE_URL", "https://api.minimax.chat/v1")
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY", "")

QWEN_API_BASE_URL = os.getenv("QWEN_API_BASE_URL", "")
QWEN_API_KEY = os.getenv("QWEN_API_KEY", os.getenv("DASHSCOPE_API_KEY", ""))


def is_deepseek_api_model(model: str) -> bool:
    return model in {"deepseek-chat", "deepseek-reasoner"} or model.startswith("deepseek/")


def is_minimax_model(model: str) -> bool:
    return model.startswith("minimax/")


def is_qwen_model(model: str) -> bool:
    return model.startswith("qwen/") or model.startswith("Qwen/")


def is_vllm_model(model: str) -> bool:
    return "vllm" in model.lower()


def is_openai_model(model: str) -> bool:
    return "gpt" in model.lower() or model.startswith("o")


def create_client(model: str):
    """Create an LLM client for the given model.

    Supports:
      - OpenAI (gpt-*, o*)
      - DeepSeek (deepseek/*, deepseek-chat, deepseek-reasoner)
      - Minimax (minimax/*)
      - Qwen via vLLM (qwen/*, Qwen/*) or DashScope API
      - Anthropic via OpenRouter (anthropic/*)
      - Custom vLLM (vllm-<host>)
    """
    if is_openai_model(model):
        print(f"Using OpenAI API with model {model}.")
        return openai.OpenAI(), model

    if is_deepseek_api_model(model):
        actual_model = model.split("/")[-1] if "/" in model else model
        print(f"Using DeepSeek API with model {actual_model}.")
        api_key = DEEPSEEK_API_KEY or os.getenv("OPENAI_API_KEY", "")
        client = openai.OpenAI(
            base_url=DEEPSEEK_API_BASE_URL,
            api_key=api_key,
            timeout=DEEPSEEK_API_TIMEOUT,
            max_retries=1,
        )
        return client, actual_model

    if is_minimax_model(model):
        actual_model = model.split("/", 1)[-1] if "/" in model else model
        print(f"Using Minimax API with model {actual_model}.")
        client = openai.OpenAI(
            base_url=MINIMAX_API_BASE_URL,
            api_key=MINIMAX_API_KEY,
        )
        return client, actual_model

    if is_qwen_model(model):
        actual_model = model.split("/", 1)[-1] if "/" in model else model
        if QWEN_API_BASE_URL:
            print(f"Using Qwen DashScope API with model {actual_model}.")
            client = openai.OpenAI(
                base_url=QWEN_API_BASE_URL,
                api_key=QWEN_API_KEY,
            )
            return client, actual_model
        else:
            print(f"Using vLLM for Qwen model {model}.")
            client = openai.OpenAI(
                base_url=f"http://{VLLM_HOST}:{VLLM_PORT}/v1",
                api_key="dummy",
            )
            # vLLM validates the exact served model name. Preserve the
            # organization prefix instead of using the DashScope short name.
            return client, model

    if is_vllm_model(model):
        host = model[model.index("-") + 1:] if "-" in model else VLLM_HOST
        print(f"Using vLLM API at {host}.")
        return (
            openai.OpenAI(base_url=f"http://{host}:8000/v1", api_key="dummy"),
            model,
        )

    # Default: OpenRouter (covers Anthropic and other models)
    print(f"Using OpenRouter API with model {model}.")
    return (
        openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OpenRouter_API_KEY", ""),
        ),
        model,
    )


def llm_container_env():
    """Return environment variables needed inside containers."""
    return {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", ""),
        "OpenRouter_API_KEY": os.getenv("OpenRouter_API_KEY", ""),
        "DEEPSEEK_API_KEY": DEEPSEEK_API_KEY,
        "DEEPSEEK_API_BASE_URL": DEEPSEEK_API_BASE_URL,
        "DEEPSEEK_MAX_OUTPUT_TOKENS": str(DEEPSEEK_MAX_OUTPUT_TOKENS),
        "DEEPSEEK_API_TIMEOUT": str(DEEPSEEK_API_TIMEOUT),
        "MINIMAX_API_KEY": MINIMAX_API_KEY,
        "MINIMAX_API_BASE_URL": MINIMAX_API_BASE_URL,
        "QWEN_API_KEY": QWEN_API_KEY,
        "QWEN_API_BASE_URL": QWEN_API_BASE_URL,
        "VLLM_HOST": VLLM_HOST,
        "VLLM_PORT": VLLM_PORT,
    }


@backoff.on_exception(
    backoff.expo,
    (
        openai.RateLimitError,
        openai.APITimeoutError,
        anthropic.RateLimitError,
        anthropic.APIStatusError,
    ),
    max_time=120,
)
def get_json_response_from_llm(
    msg,
    client,
    model,
    system_message,
):
    new_msg_history = [{"role": "user", "content": msg}]
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_message},
            *new_msg_history,
        ],
        "n": 1,
        "stop": None,
        "seed": 0,
        "response_format": {"type": "json_object"},
    }
    response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content
    content_json = json.loads(content)
    new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    return content_json, new_msg_history


def get_response_from_llm(
    msg,
    client,
    model,
    system_message,
    print_debug=False,
    msg_history=None,
    temperature=0.7,
):
    if msg_history is None:
        msg_history = []

    if model.startswith("o"):
        new_msg_history = msg_history + [
            {"role": "user", "content": system_message + msg}
        ]
        response = client.chat.completions.create(
            model=model,
            messages=new_msg_history,
            temperature=1,
            n=1,
            seed=0,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif "gpt" in model.lower():
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            n=1,
            stop=None,
            seed=0,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    else:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        actual_model = model
        try:
            models_list = client.models.list()
            if models_list.data:
                actual_model = models_list.data[0].id
        except Exception:
            pass
        response = client.chat.completions.create(
            model=actual_model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_OUTPUT_TOKENS,
            n=1,
            stop=None,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    if print_debug:
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        print(f'User: {new_msg_history[-2]["content"]}')
        print(f'Assistant: {new_msg_history[-1]["content"]}')
        print("*" * 21 + " LLM END " + "*" * 21)
        print()
    return content, new_msg_history


def extract_json_between_markers(llm_output):
    inside_json_block = False
    json_lines = []

    for line in llm_output.split("\n"):
        striped_line = line.strip()

        if striped_line.startswith("```json"):
            inside_json_block = True
            continue

        if inside_json_block and striped_line.startswith("```"):
            inside_json_block = False
            break

        if inside_json_block:
            json_lines.append(line)

    if not json_lines:
        fallback_pattern = r"\{.*?\}"
        matches = re.findall(fallback_pattern, llm_output, re.DOTALL)
        for candidate in matches:
            candidate = candidate.strip()
            if candidate:
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    candidate_clean = re.sub(r"[\x00-\x1F\x7F]", "", candidate)
                    try:
                        return json.loads(candidate_clean)
                    except json.JSONDecodeError:
                        continue
        return None

    json_string = "\n".join(json_lines).strip()

    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
        try:
            return json.loads(json_string_clean)
        except json.JSONDecodeError:
            return None
