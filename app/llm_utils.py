from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Sequence, Tuple

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None  # type: ignore


def _load_env_file() -> None:
    """Ensure .env values are available even without python-dotenv."""
    project_root = Path(__file__).resolve().parents[1]
    candidate_paths = [
        project_root / ".streamlit" / "secrets.toml",
        project_root / "secrets.toml",
    ]
    env_path = next((path for path in candidate_paths if path.exists()), None)
    if env_path is None:
        return
    if load_dotenv:
        load_dotenv(env_path, override=False)
        return
    for line in env_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


_load_env_file()


def _format_prompt(sentences: Sequence[str]) -> str:
    lines = "\n".join(f"- {s.strip()}" for s in sentences if s and s.strip())
    return (
        "以下の代表文をよく読み、このクラスタが表すテーマを日本語で30文字以内のタイトルと"
        "500文字程度の説明にまとめてください。その後、そのテーマに対する具体的な対策を200文字程度で提案してください。"
        "ただし、それぞれ別の文書から抽出したものであるため、文書に固有した要約ではなく、なるべく各文章に共通するエッセンスのみを書いてください。\n"
        f"{lines}\n"
        "出力形式: 「タイトル: ...\\n説明: ...\\n対策: ...」"
    )


def _call_openai(sentences: Sequence[str], model: str = "gpt-5-mini") -> str:
    if OpenAI is None:
        return "openai パッケージがインストールされていません。requirements を確認してください。"

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "OPENAI_API_KEY が設定されていません。環境変数を設定してください。"

    client = OpenAI(api_key=api_key)
    prompt = _format_prompt(sentences)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "あなたは日本語で要約とラベル付けを行うアナリストです。",
                },
                {"role": "user", "content": prompt},
            ],
        )
    except Exception as exc:  # pragma: no cover - network errors
        return f"LLM呼び出しに失敗しました: {exc}"

    return response.choices[0].message.content.strip()


@lru_cache(maxsize=128)
def _cached_label(model: str, sentences_key: Tuple[str, ...]) -> str:
    return _call_openai(sentences_key, model=model)


def label_cluster(sentences: Sequence[str], model: str = "gpt-5-mini") -> str:
    """
    Return a human-readable label for a cluster using representative sentences.
    Results are cached per (model, sentences) combination to avoid repeated calls.
    """
    sanitized = tuple(s.strip() for s in sentences if s and s.strip())
    if not sanitized:
        return "代表文が不足しているためラベルを生成できません。"
    return _cached_label(model, sanitized)
