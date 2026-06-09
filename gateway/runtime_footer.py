"""Gateway runtime-metadata footer.

Renders a compact footer showing runtime state and appends it to the FINAL
message of an agent turn when enabled. Per-platform overrides live under
``display.platforms.<platform>.runtime_footer``.
"""

from __future__ import annotations

import os
from typing import Any, Iterable, Optional

_DEFAULT_FIELDS: tuple[str, ...] = ("model", "context_pct", "cwd")
_DEFAULT_SEPARATOR = " · "


def _home_relative_cwd(cwd: str) -> str:
    if not cwd:
        return ""
    try:
        home = os.path.expanduser("~")
        p = os.path.abspath(cwd)
        if home and (p == home or p.startswith(home + os.sep)):
            return "~" + p[len(home):]
        return p
    except Exception:
        return cwd


def _model_short(model: Optional[str]) -> str:
    if not model:
        return ""
    return str(model).rsplit("/", 1)[-1]


def _clean_text(value: Any) -> str:
    text = str(value or "").strip()
    return text


def _normalize_separator(separator: Any) -> str:
    text = str(separator) if separator is not None else ""
    return text if text else _DEFAULT_SEPARATOR


def _rate_limit_bucket_left_pct(bucket: Any) -> Optional[int]:
    if not bucket:
        return None
    limit = getattr(bucket, "limit", 0) or 0
    remaining = getattr(bucket, "remaining", 0) or 0
    if limit <= 0:
        return None
    try:
        return max(0, min(100, round((float(remaining) / float(limit)) * 100)))
    except Exception:
        return None


def _format_account_budget(provider: Optional[str]) -> str:
    normalized = _clean_text(provider).lower()
    if normalized != "openai-codex":
        return ""
    try:
        from agent.account_usage import fetch_account_usage

        snapshot = fetch_account_usage(normalized)
    except Exception:
        return ""

    windows = getattr(snapshot, "windows", ()) or ()
    parts: list[str] = []
    for window in windows:
        used_percent = getattr(window, "used_percent", None)
        if used_percent is None:
            continue
        label = str(getattr(window, "label", "") or "").strip().lower()
        if label.startswith("session") or label.startswith("primary") or "current session" in label:
            prefix = "s"
        elif label.startswith("week") or label.startswith("secondary") or "current week" in label:
            prefix = "w"
        else:
            continue
        remaining = max(0, min(100, round(100 - float(used_percent))))
        parts.append(f"{prefix}:{remaining}% left")
    return " | ".join(parts)


def _format_rate_limit_budget(rate_limit_state: Any, provider: Optional[str] = None) -> str:
    """Render compact short/window budget as ``s:4% left | w:9% left``.

    Prefer provider rate-limit headers when present; fall back to account usage
    for OAuth-backed providers that expose subscription windows (OpenAI Codex).
    """
    normalized = _clean_text(provider).lower()
    if normalized == "openai-codex":
        account_budget = _format_account_budget(normalized)
        if account_budget:
            return account_budget

    if not rate_limit_state:
        return ""

    short_bucket = getattr(rate_limit_state, "tokens_min", None) or getattr(rate_limit_state, "requests_min", None)
    wide_bucket = getattr(rate_limit_state, "tokens_hour", None) or getattr(rate_limit_state, "requests_hour", None)
    short_pct = _rate_limit_bucket_left_pct(short_bucket)
    wide_pct = _rate_limit_bucket_left_pct(wide_bucket)
    parts: list[str] = []
    if short_pct is not None:
        parts.append(f"s:{short_pct}% left")
    if wide_pct is not None:
        parts.append(f"w:{wide_pct}% left")
    return " | ".join(parts)


def resolve_footer_config(
    user_config: dict[str, Any] | None,
    platform_key: str | None = None,
) -> dict[str, Any]:
    resolved = {"enabled": False, "fields": list(_DEFAULT_FIELDS), "separator": _DEFAULT_SEPARATOR}
    cfg = (user_config or {}).get("display") or {}

    global_cfg = cfg.get("runtime_footer")
    if isinstance(global_cfg, dict):
        if "enabled" in global_cfg:
            resolved["enabled"] = bool(global_cfg.get("enabled"))
        if isinstance(global_cfg.get("fields"), list) and global_cfg["fields"]:
            resolved["fields"] = [str(f) for f in global_cfg["fields"]]
        if "separator" in global_cfg:
            resolved["separator"] = _normalize_separator(global_cfg.get("separator"))

    if platform_key:
        platforms = cfg.get("platforms") or {}
        plat_cfg = platforms.get(platform_key)
        if isinstance(plat_cfg, dict):
            plat_footer = plat_cfg.get("runtime_footer")
            if isinstance(plat_footer, dict):
                if "enabled" in plat_footer:
                    resolved["enabled"] = bool(plat_footer.get("enabled"))
                if isinstance(plat_footer.get("fields"), list) and plat_footer["fields"]:
                    resolved["fields"] = [str(f) for f in plat_footer["fields"]]
                if "separator" in plat_footer:
                    resolved["separator"] = _normalize_separator(plat_footer.get("separator"))

    return resolved


def _compact_token_count(value: int) -> str:
    try:
        n = int(value)
    except Exception:
        return str(value)
    if abs(n) >= 1_000_000:
        whole = round(n / 1_000_000)
        return f"{whole}m"
    if abs(n) >= 1_000:
        whole = round(n / 1_000)
        return f"{whole}k"
    return str(n)


def _format_context(context_tokens: int, context_length: Optional[int]) -> str:
    if not context_length or context_length <= 0 or context_tokens < 0:
        return ""
    pct = max(0, min(100, round((context_tokens / context_length) * 100)))
    return f"ctx:{_compact_token_count(context_tokens)}/{_compact_token_count(context_length)} ({pct}%)"


def _format_context_pct(context_tokens: int, context_length: Optional[int]) -> str:
    if not context_length or context_length <= 0 or context_tokens < 0:
        return ""
    pct = max(0, min(100, round((context_tokens / context_length) * 100)))
    return f"ctx {pct}%"


def format_runtime_footer(
    *,
    model: Optional[str],
    context_tokens: int,
    context_length: Optional[int],
    cwd: Optional[str] = None,
    provider: Optional[str] = None,
    profile: Optional[str] = None,
    reasoning: Optional[str] = None,
    fields: Iterable[str] = _DEFAULT_FIELDS,
    separator: str = _DEFAULT_SEPARATOR,
    compression_count: Optional[int] = None,
    rate_limit_budget: Optional[str] = None,
) -> str:
    lines: list[list[str]] = [[]]

    def push(value: str) -> None:
        value = _clean_text(value)
        if value:
            lines[-1].append(value)

    def newline() -> None:
        if lines[-1]:
            lines.append([])

    rel_cwd = _home_relative_cwd(cwd or os.environ.get("TERMINAL_CWD", ""))
    profile_text = _clean_text(profile or os.environ.get("HERMES_PROFILE", "default"))
    provider_text = _clean_text(provider)
    reasoning_text = _clean_text(reasoning)
    if reasoning_text.lower() == "none":
        reasoning_text = ""
    budget_text = _clean_text(rate_limit_budget)

    for field in fields:
        if field == "line_break":
            newline()
        elif field == "model":
            push(_model_short(model))
        elif field == "provider":
            push(provider_text)
        elif field in {"thinking", "reasoning"}:
            push(reasoning_text)
        elif field == "profile":
            push(profile_text)
        elif field == "context":
            push(_format_context(context_tokens, context_length))
        elif field == "context_pct":
            push(_format_context_pct(context_tokens, context_length))
        elif field == "cwd":
            push(rel_cwd)
        elif field == "compaction":
            push(f"c:{int(compression_count or 0)}")
        elif field == "rate_limit_budget":
            push(budget_text or "budget:n/a")
        else:
            # Unknown fields are ignored for forwards/backwards compatibility.
            continue

    rendered_lines = [separator.join(parts).strip() for parts in lines if parts]
    return "\n".join(line for line in rendered_lines if line).strip()


def build_footer_line(
    *,
    user_config: dict[str, Any] | None,
    platform_key: str | None,
    model: Optional[str],
    context_tokens: int,
    context_length: Optional[int],
    cwd: Optional[str] = None,
    provider: Optional[str] = None,
    profile: Optional[str] = None,
    reasoning: Optional[str] = None,
    rate_limit_state: Any = None,
    compression_count: Optional[int] = None,
    rate_limit_budget: Optional[str] = None,
) -> str:
    cfg = resolve_footer_config(user_config, platform_key)
    if not cfg.get("enabled"):
        return ""

    if not provider and isinstance((user_config or {}).get("model"), dict):
        provider = (user_config or {}).get("model", {}).get("provider")
    if not reasoning and isinstance((user_config or {}).get("agent"), dict):
        reasoning = (user_config or {}).get("agent", {}).get("reasoning_effort")

    if not rate_limit_budget or provider == "openai-codex":
        if rate_limit_state and not isinstance(rate_limit_state, str):
            rate_limit_budget = _format_rate_limit_budget(rate_limit_state, provider)
        elif provider == "openai-codex":
            rate_limit_budget = _format_account_budget(provider) or rate_limit_budget
        elif isinstance(rate_limit_state, str):
            rate_limit_budget = rate_limit_state

    return format_runtime_footer(
        model=model,
        provider=provider,
        profile=profile,
        reasoning=reasoning,
        context_tokens=context_tokens,
        context_length=context_length,
        cwd=cwd,
        fields=cfg.get("fields") or _DEFAULT_FIELDS,
        separator=_normalize_separator(cfg.get("separator")),
        compression_count=compression_count,
        rate_limit_budget=rate_limit_budget,
    )
