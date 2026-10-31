"""Prints the OpenRouter credit balance and this key's usage. Uses the same key as the translation check
(env OPENROUTER_API_KEY, else the "openrouter:<key>" line in bodhan.txt).

    python backend/openrouter_credits.py"""

import sys

import requests

try:
    from . import review
except ImportError:
    import review

BASE = "https://openrouter.ai/api/v1"


def get(path: str, key: str) -> dict | None:
    """The response's "data", or None (with the error printed) if the request fails."""
    try:
        res = requests.get(f"{BASE}/{path}", headers={"Authorization": f"Bearer {key}"}, timeout=30)
    except requests.RequestException as exc:
        print(f"{path}: request failed: {exc}")
        return None
    if res.status_code != 200:
        print(f"{path}: OpenRouter {res.status_code}: {res.text[:300]}")
        return None
    return res.json().get("data") or {}


def main() -> int:
    try:
        key = review.api_key()
    except RuntimeError as exc:
        print(exc)
        return 1

    credits = get("credits", key)  # account: credits bought and used
    if credits is not None:
        total, used = credits.get("total_credits") or 0, credits.get("total_usage") or 0
        print(f"Account credits: ${total:.4f} bought, ${used:.4f} used, ${total - used:.4f} left")

    info = get("key", key)  # this key: its own usage and limit
    if info is not None:
        limit = info.get("limit")
        print(f"This key ({info.get('label') or 'no label'}): ${info.get('usage') or 0:.4f} used, "
              + (f"limit ${limit:.4f}, ${info.get('limit_remaining') or 0:.4f} remaining" if limit is not None else "no limit")
              + (", free tier" if info.get("is_free_tier") else ""))

    return 0 if credits is not None or info is not None else 1


if __name__ == "__main__":
    sys.exit(main())
