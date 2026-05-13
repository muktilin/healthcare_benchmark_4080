import argparse
import html
import json
import re
from typing import List, Tuple

ON_CHIP_MODEL_NAME = "nvidia/DLER-R1-1.5B-Research"
ON_CHIP_SERVER_URL_DEFAULT = "http://192.168.115.190:8080"

_CLIENT_CACHE = {}


def is_valid_on_chip_response(text: str) -> bool:
    if not text or not text.strip():
        return False
    lowered = text.lower()
    invalid_markers = (
        "no activity log found",
        "did not return",
        "could not connect",
        "error:",
        "no response from model",
        "unexpected response format",
    )
    return not any(marker in lowered for marker in invalid_markers)


def _clean_json_text(text: str) -> str:
    stripped = (text or "").strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped)
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    return match.group(0) if match else stripped


def _stringify_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(item) for item in value if item is not None)
    if isinstance(value, dict):
        return "; ".join(f"{key}: {val}" for key, val in value.items())
    return str(value).strip()


def parse_on_chip_summary(text: str) -> dict:
    fields = {}
    label_map = {
        "summary": "summary",
        "key actions": "key_actions",
        "key_actions": "key_actions",
        "risk": "risk",
        "anomaly": "anomaly",
        "advice": "advice",
        "advise": "advice",
    }

    try:
        payload = json.loads(_clean_json_text(text))
        fields.update(
            {
                "summary": _stringify_value(payload.get("summary") or payload.get("Summary")),
                "key_actions": _stringify_value(
                    payload.get("key_actions")
                    or payload.get("key actions")
                    or payload.get("Key actions")
                    or payload.get("Key Actions")
                ),
                "risk": _stringify_value(payload.get("risk") or payload.get("Risk")),
                "anomaly": _stringify_value(payload.get("anomaly") or payload.get("Anomaly")),
                "advice": _stringify_value(payload.get("advice") or payload.get("advise") or payload.get("Advice")),
            }
        )
    except Exception:
        labels = "|".join(re.escape(label) for label in label_map)
        pattern = re.compile(
            rf"(?P<label>{labels})\s*:\s*(?P<value>.*?)(?=\n\s*(?:{labels})\s*:|\Z)",
            flags=re.IGNORECASE | re.DOTALL,
        )
        for match in pattern.finditer(text or ""):
            key = label_map[match.group("label").lower()]
            fields[key] = match.group("value").strip()

    defaults = {
        "summary": "No clear summary was returned.",
        "key_actions": "Limited evidence.",
        "risk": "Limited evidence.",
        "anomaly": "None reported.",
        "advice": "Review the activity timeline and continue routine observation.",
    }
    return {key: fields.get(key) or value for key, value in defaults.items()}


def render_summary_cards(report: dict) -> str:
    items = [
        ("summary", "📝", "Summary", report["summary"]),
        ("actions", "✅", "Key actions", report["key_actions"]),
        ("risk", "⚠️", "Risk", report["risk"]),
        ("anomaly", "🔎", "Anomaly", report["anomaly"]),
        ("advice", "💡", "Advice", report["advice"]),
    ]
    cards = []
    for css_name, icon, name, value in items:
        safe_value = html.escape(value).replace("\n", "<br>")
        cards.append(
            f"""
            <div class="summary-item {css_name}">
              <div class="summary-icon">{html.escape(icon)}</div>
              <div>
                <div class="summary-name">{html.escape(name)}</div>
                <div class="summary-value">{safe_value}</div>
              </div>
            </div>
            """
        )
    return f"<div class='summary-panel'><div class='summary-grid'>{''.join(cards)}</div></div>"


def render_on_chip_summary(text: str) -> str:
    return render_summary_cards(parse_on_chip_summary(text))


def empty_summary_cards(message: str = "Generate a summary to review structured care items.") -> str:
    return f"<div class='summary-panel'><div class='summary-empty'>{html.escape(message)}</div></div>"


def _parse_log_line(line: str) -> Tuple[str, str]:
    match = re.match(r"^\[(.*?)\]\s*(.*)$", line.strip())
    if not match:
        return "", line.strip()
    return match.group(1).strip(), match.group(2).strip()


def _compress_log_for_small_model(log_text: str, max_items: int = 14, max_chars: int = 900) -> str:
    lines = [line.strip() for line in log_text.splitlines() if line.strip()]
    if not lines:
        return ""

    merged: List[Tuple[str, str, str]] = []
    for line in lines:
        label, action = _parse_log_line(line)
        if not action:
            continue
        if merged and merged[-1][2] == action:
            start_label, _, last_action = merged[-1]
            merged[-1] = (start_label, label or start_label, last_action)
        else:
            merged.append((label, label, action))

    if len(merged) > max_items:
        if max_items <= 4:
            keep_indices = list(range(min(len(merged), max_items)))
        else:
            head = 4
            tail = 3
            middle_slots = max_items - head - tail
            middle_start = head
            middle_end = max(head, len(merged) - tail)
            step = max(1, (middle_end - middle_start) // max(1, middle_slots))
            middle_indices = list(range(middle_start, middle_end, step))[:middle_slots]
            keep_indices = list(range(head)) + middle_indices + list(range(max(head, len(merged) - tail), len(merged)))
        merged = [merged[i] for i in keep_indices if 0 <= i < len(merged)]

    compact_lines = []
    for start_label, end_label, action in merged:
        if start_label and end_label and start_label != end_label:
            compact_lines.append(f"{start_label} -> {end_label}: {action}")
        elif start_label:
            compact_lines.append(f"{start_label}: {action}")
        else:
            compact_lines.append(action)

    compact_text = "\n".join(compact_lines)
    while len(compact_text) > max_chars and len(compact_lines) > 4:
        compact_lines = compact_lines[:-1]
        compact_text = "\n".join(compact_lines)
    return compact_text[:max_chars]


def estimate_token_count(text: str) -> int:
    if not text:
        return 0
    # Lightweight approximation for quick budgeting when the on-chip tokenizer
    # is not available locally. English text is often around 3-4 chars/token.
    return max(1, (len(text) + 3) // 4)


def _get_client(model_name: str = ON_CHIP_MODEL_NAME, server_url: str = ON_CHIP_SERVER_URL_DEFAULT):
    cache_key = (server_url, model_name)
    client = _CLIENT_CACHE.get(cache_key)
    if client is None:
        from llm_api_client import LLMClient

        client = LLMClient(
            server_url=server_url,
            model=model_name,
            temperature=0.6,
            no_think=False,
        )
        _CLIENT_CACHE[cache_key] = client
    return client


def generate_on_chip_summary(
    person_id: int,
    log_text: str,
    model_name: str = ON_CHIP_MODEL_NAME,
    server_url: str = ON_CHIP_SERVER_URL_DEFAULT,
) -> str:
    if not log_text:
        return "No activity log found."

    compact_log = _compress_log_for_small_model(log_text)
    example_block = (
        "Output Example:\n"
        "Summary: The elder rested first, then changed posture and walked briefly before sitting again.\n"
        "Key actions: lying, sitting, standing, walking.\n"
        "Risk: Low to mild mobility risk during posture changes.\n"
        "Anomaly: None.\n"
    )

    prompt = (
        f"Summarize activity for person {person_id} from one video.\n"
        f"{example_block}"
        "Use only the records below.\n"
        "Keep the total answer under 300 words.\n"
        "Use exactly 4 short lines:\n"
        "Summary: ...\n"
        "Key actions: ...\n"
        "Risk: ...\n"
        "Anomaly: ...\n"
        "If evidence is weak, say limited evidence.\n"
        f"Records:\n{compact_log}"
    )
    prompt_tokens = estimate_token_count(prompt)
    print(f"[OnChip] Estimated input tokens: {prompt_tokens}")

    client = _get_client(model_name=model_name, server_url=server_url)
    response = client.send_prompt(prompt, reset_conversation=True)
    client.reset_conversation()
    return response.strip() if response else "On-chip model did not return a valid summary."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--person_id", type=int, default=1)
    parser.add_argument("--server_url", default=ON_CHIP_SERVER_URL_DEFAULT)
    parser.add_argument("--model_name", default=ON_CHIP_MODEL_NAME)
    parser.add_argument("--log_text", default=None)
    args = parser.parse_args()

    demo_log = """
[2019-06-22 09:00:02] lying
[2019-06-22 09:12:11] lying
[2019-06-22 09:18:24] sitting
[2019-06-22 09:21:48] sitting
[2019-06-22 09:36:03] standing
[2019-06-22 09:37:15] walking
[2019-06-22 09:39:40] walking
[2019-06-22 09:42:01] sitting
[2019-06-22 10:03:22] eating
[2019-06-22 10:16:40] sitting
[2019-06-22 10:45:05] walking
[2019-06-22 10:46:11] standing
""".strip()
    
    log_text = args.log_text if args.log_text else demo_log
    compact_log = _compress_log_for_small_model(log_text)
    prompt_preview = (
        f"Please generate summarization for the elderly.\n"
        "Use only the records below.\n"
        "Summary: ...\n"
        "Key actions: ...\n"
        "Risk: ...\n"
        "Anomaly: ...\n"
        "If evidence is weak, say limited evidence.\n"
        f"Records:\n{compact_log}"
    )

    print(f"Server: {args.server_url}")
    print(f"Model: {args.model_name}")
    print(f"Person ID: {args.person_id}")
    print(f"Estimated input tokens: {estimate_token_count(prompt_preview)}")
    print("\nCompressed log:")
    print("-" * 60)
    print(compact_log)
    print("-" * 60)
    
    summary = generate_on_chip_summary(
        person_id=args.person_id,
        log_text=log_text,
        model_name=args.model_name,
        server_url=args.server_url,
    )

    print("\nModel response:")
    print("=" * 60)
    print(summary)
    print("=" * 60)


if __name__ == "__main__":
    main()
