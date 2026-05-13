import argparse
import html
import json
import re
from typing import List, Tuple

ON_CHIP_MODEL_NAME = "nvidia/DLER-R1-1.5B-Research"
ON_CHIP_SERVER_URL_DEFAULT = "http://192.168.115.190:8080"
QWEN_POSTPROCESS_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

_CLIENT_CACHE = {}
_QWEN_POSTPROCESS_CACHE = {}


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
            rf"(?P<label>{labels})\s*:\s*(?P<value>.*?)(?=\s*(?:{labels})\s*:|\Z)",
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


def _extract_generated_text(result) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for key in ("generated_text", "text", "output_text"):
            if key in result:
                return _extract_generated_text(result[key])
        return ""
    if isinstance(result, list):
        if not result:
            return ""
        if all(isinstance(item, dict) and "role" in item for item in result):
            for message in reversed(result):
                if message.get("role") == "assistant":
                    content = message.get("content", "")
                    if isinstance(content, list):
                        return "\n".join(
                            part.get("text", "")
                            for part in content
                            if isinstance(part, dict) and part.get("type") == "text"
                        ).strip()
                    return str(content).strip()
        return _extract_generated_text(result[0])
    return str(result).strip()


def _get_qwen_postprocess_model(model_name: str = QWEN_POSTPROCESS_MODEL_NAME):
    cached = _QWEN_POSTPROCESS_CACHE.get(model_name)
    if cached is not None:
        return cached

    from transformers import AutoTokenizer

    try:
        from transformers import AutoModelForImageTextToText

        model_cls = AutoModelForImageTextToText
    except ImportError:
        from transformers import Qwen2_5_VLForConditionalGeneration

        model_cls = Qwen2_5_VLForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = model_cls.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    cached = (tokenizer, model)
    _QWEN_POSTPROCESS_CACHE[model_name] = cached
    return cached


def _run_qwen_postprocess(prompt: str, model_name: str = QWEN_POSTPROCESS_MODEL_NAME) -> str:
    tokenizer, model = _get_qwen_postprocess_model(model_name)
    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = prompt
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=512, do_sample=False)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()


def _format_timeline_for_postprocess(log_text: str, max_items: int = 80, max_chars: int = 6000) -> str:
    lines = [line.strip() for line in (log_text or "").splitlines() if line.strip()]
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
        head = min(24, max_items // 2)
        tail = min(16, max_items - head)
        middle_slots = max_items - head - tail
        middle_start = head
        middle_end = max(head, len(merged) - tail)
        if middle_slots > 0 and middle_end > middle_start:
            step = max(1, (middle_end - middle_start) // middle_slots)
            middle_indices = list(range(middle_start, middle_end, step))[:middle_slots]
        else:
            middle_indices = []
        keep_indices = list(range(head)) + middle_indices + list(range(max(head, len(merged) - tail), len(merged)))
        merged = [merged[i] for i in keep_indices if 0 <= i < len(merged)]

    formatted = []
    for start_label, end_label, action in merged:
        if start_label and end_label and start_label != end_label:
            formatted.append(f"{start_label} -> {end_label}: {action}")
        elif start_label:
            formatted.append(f"{start_label}: {action}")
        else:
            formatted.append(action)

    evidence = "\n".join(formatted)
    while len(evidence) > max_chars and len(formatted) > 8:
        formatted = formatted[:-4]
        evidence = "\n".join(formatted)
    return evidence[:max_chars]


def _has_time_reference(text: str) -> bool:
    return bool(
        re.search(
            r"(\d{1,2}:\d{2}(?::\d{2})?|T\+\d+(?:\.\d+)?s?|\d{4}[-/]\d{2}[-/]\d{2}|->)",
            text or "",
        )
    )


def _sentence_count(text: str) -> int:
    sentences = [part.strip() for part in re.split(r"[.!?。！？]+", text or "") if part.strip()]
    return len(sentences)


def _report_has_timeline_detail(report: dict) -> bool:
    summary = report.get("summary", "")
    leaked_labels = re.search(r"\b(key actions|risk|anomaly|advice|advise)\s*:", summary, flags=re.IGNORECASE)
    return _has_time_reference(summary) and _sentence_count(summary) >= 3 and not leaked_labels


def _line_time_and_action(line: str) -> Tuple[str, str]:
    if ":" not in line:
        return "", line.strip()
    time_part, action = line.rsplit(":", 1)
    return time_part.strip(), action.strip()


def _time_bounds(time_part: str) -> Tuple[str, str]:
    if "->" not in (time_part or ""):
        stripped = (time_part or "").strip()
        return stripped, stripped
    start, end = time_part.split("->", 1)
    return start.strip(), end.strip()


def _compact_action_label(action: str) -> str:
    action = re.sub(r"\s+", " ", action or "").strip(" .;")
    lowered = action.lower()
    if "sitting" in lowered and "looking down" in lowered:
        return "sitting and looking down"
    if "holding" in lowered and "examining" in lowered and "tissue" in lowered:
        return "examining a tissue"
    if "holding" in lowered and "examining" in lowered:
        return "examining a small object"
    if "phone" in lowered or "conversation" in lowered:
        return "phone conversation"
    if "interacting" in lowered and ("individual" in lowered or "person" in lowered):
        return "interacting with another person"
    if "resting" in lowered or "waiting" in lowered:
        return "resting or waiting"
    if "walking" in lowered:
        return "walking"
    if "standing" in lowered:
        return "standing"
    if "sitting" in lowered:
        return "sitting"
    if "lying" in lowered:
        return "lying"
    prefixes = (
        "the person is ",
        "person is ",
        "the person appears to be ",
        "person appears to be ",
        "appears to be ",
    )
    for prefix in prefixes:
        if lowered.startswith(prefix):
            action = action[len(prefix):]
            break
    action = re.sub(r"\bthe person\b", "person", action, flags=re.IGNORECASE)
    if len(action) > 70:
        action = action[:67].rsplit(" ", 1)[0] + "..."
    return action.strip()


def _unique_actions(segments: List[Tuple[str, str]], limit: int = 5) -> List[str]:
    actions = []
    seen = set()
    for _, action in segments:
        label = _compact_action_label(action)
        key = label.lower()
        if not label or key in seen:
            continue
        seen.add(key)
        actions.append(label)
        if len(actions) >= limit:
            break
    return actions


def _strip_time_references(text: str) -> str:
    text = re.sub(r"\(?\b\d{4}[-/]\d{2}[-/]\d{2}\s+\d{1,2}:\d{2}(?::\d{2})?(?:\s*->\s*\d{4}[-/]\d{2}[-/]\d{2}\s+\d{1,2}:\d{2}(?::\d{2})?)?\)?", "", text or "")
    text = re.sub(r"\(?\b\d{1,2}:\d{2}(?::\d{2})?(?:\s*->\s*\d{1,2}:\d{2}(?::\d{2})?)?\)?", "", text)
    text = re.sub(r"\(?\bT\+\d+(?:\.\d+)?s?\)?", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([,.;])", r"\1", text)
    text = re.sub(r"\bfrom\s+to\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bto\s+window\b", "window", text, flags=re.IGNORECASE)
    return text.strip(" ,;")


def _dedupe_list_text(text: str, max_items: int = 5) -> str:
    parts = [part.strip(" ,.;") for part in re.split(r";|\n|,(?=\s*(?:the person|person|[A-Za-z]+ing\b))", text or "")]
    deduped = []
    seen = set()
    for part in parts:
        cleaned = _compact_action_label(_strip_time_references(part))
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        deduped.append(cleaned)
        if len(deduped) >= max_items:
            break
    return "; ".join(deduped)


def _clean_refined_report(report: dict) -> dict:
    cleaned = dict(report)
    cleaned["key_actions"] = _dedupe_list_text(cleaned.get("key_actions", ""), max_items=5) or "Limited evidence."

    risk = _strip_time_references(cleaned.get("risk", ""))
    risk = re.sub(r"\baround\s*[,;]?\s*", "", risk, flags=re.IGNORECASE).strip(" ,;")
    cleaned["risk"] = risk or "Limited risk evidence."

    anomaly = _strip_time_references(cleaned.get("anomaly", ""))
    if not anomaly or "none observed" in anomaly.lower():
        anomaly = "None observed in the DB timeline."
    cleaned["anomaly"] = anomaly

    advice = _strip_time_references(cleaned.get("advice", ""))
    if re.search(r"\buse\s+the\s+window\b", advice, flags=re.IGNORECASE):
        advice = ""
    cleaned["advice"] = advice or "Continue routine observation and review comfort or posture changes."
    return cleaned


def _build_timeline_report(log_text: str, person_id: int = None, base_report: dict = None) -> dict:
    base_report = base_report or {}
    timeline = _format_timeline_for_postprocess(log_text, max_items=24, max_chars=3000)
    lines = [line for line in timeline.splitlines() if line.strip()]
    if not lines:
        return {
            "summary": base_report.get("summary") or "No DB timeline records were available for time-based summarization.",
            "key_actions": base_report.get("key_actions") or "Limited evidence.",
            "risk": base_report.get("risk") or "Limited evidence.",
            "anomaly": base_report.get("anomaly") or "None reported.",
            "advice": base_report.get("advice") or "Review the activity timeline and continue routine observation.",
        }

    segments = [_line_time_and_action(line) for line in lines]
    segments = [(time_part, action) for time_part, action in segments if action]
    first_time = _time_bounds(segments[0][0])[0] if segments else ""
    last_time = _time_bounds(segments[-1][0])[1] if segments else ""
    observed_window = f"{first_time} to {last_time}" if first_time and last_time and first_time != last_time else first_time or last_time

    actions = _unique_actions(segments, limit=5)
    action_flow = ", ".join(actions) or "recorded activity"

    detail_segments = []
    seen_detail = set()
    for time_part, action in segments:
        label = _compact_action_label(action)
        if not label or label.lower() in seen_detail:
            continue
        seen_detail.add(label.lower())
        detail_segments.append((time_part, label))
        if len(detail_segments) >= 1:
            break
    detail_sentences = []
    for time_part, action in detail_segments:
        detail_sentences.append(f"Around {time_part}, the timeline shows {action}." if time_part else f"The timeline shows {action}.")
    summary_sentences = [
        f"For person {person_id if person_id is not None else 'the selected person'}, the DB timeline covers {observed_window}."
        if observed_window
        else f"For person {person_id if person_id is not None else 'the selected person'}, the DB timeline records the observed activity flow.",
        f"Across that period, the main activity flow is {action_flow}.",
    ] + detail_sentences
    summary = " ".join(summary_sentences[:4])

    key_actions = "; ".join(actions)
    mobility_actions = [
        action
        for action in actions
        if any(keyword in action.lower() for keyword in ("walking", "standing", "fall", "lying", "sitting"))
    ]
    if mobility_actions:
        risk = "Low to mild mobility risk; review posture changes and any walking or standing intervals."
    else:
        risk = "Limited mobility risk evidence in the observed DB window."

    anomaly_source = base_report.get("anomaly", "")
    if anomaly_source and anomaly_source.lower() not in {"none reported.", "none observed in the db timeline.", "none"}:
        anomaly = _strip_time_references(anomaly_source)
        if not anomaly or "none observed" in anomaly.lower():
            anomaly = "None observed in the DB timeline."
    else:
        anomaly = "None observed in the DB timeline."

    advice = "Check comfort and posture stability, and review whether long resting or transition periods need follow-up."

    return {
        "summary": summary,
        "key_actions": key_actions or base_report.get("key_actions") or "Limited evidence.",
        "risk": risk,
        "anomaly": anomaly,
        "advice": advice,
    }


def _build_refine_prompt(
    on_chip_text: str,
    timeline_evidence: str,
    person_id: int = None,
    previous_answer: str = "",
) -> str:
    retry_block = (
        f"\nPrevious answer that failed validation:\n{previous_answer}\n"
        "Rewrite it now with explicit DB times in summary.\n"
        if previous_answer
        else ""
    )
    return (
        "You are refining an elder-care activity summary.\n"
        "Return only one JSON object with exactly these keys:\n"
        "summary, key_actions, risk, anomaly, advice.\n"
        "Hard requirements:\n"
        "- summary must be 3 to 5 complete sentences.\n"
        "- summary must cite the overall DB time window and may cite 1 to 2 important time ranges.\n"
        "- If the on-chip summary has no times, ignore its timing and use the DB timeline times directly.\n"
        "- Combine the on-chip semantic description with the DB timeline actions and times.\n"
        "- key_actions must list 3 to 5 unique action phrases only; do not include timestamps.\n"
        "- risk must be one concise sentence without timestamps.\n"
        "- anomaly must be one concise sentence without timestamps; say None observed in the DB timeline if appropriate.\n"
        "- advice must be one concise caregiver instruction without timestamps.\n"
        "- Do not invent actions, risks, or anomalies that are not supported by the summary or timeline.\n"
        "- Avoid repeating the same action phrase across fields.\n"
        "- If evidence is insufficient, say limited evidence in the affected field.\n"
        f"Person ID: {person_id if person_id is not None else 'unknown'}\n"
        f"{retry_block}\n"
        f"On-chip result:\n{on_chip_text or 'None'}\n\n"
        f"DB timeline records with times:\n{timeline_evidence or 'None'}"
    )


def refine_on_chip_summary(
    on_chip_text: str,
    log_text: str = "",
    person_id: int = None,
    model_name: str = QWEN_POSTPROCESS_MODEL_NAME,
) -> dict:
    timeline_evidence = _format_timeline_for_postprocess(log_text) if log_text else ""
    prompt = _build_refine_prompt(on_chip_text, timeline_evidence, person_id=person_id)

    try:
        qwen_text = _run_qwen_postprocess(prompt, model_name=model_name)
        if qwen_text and is_valid_on_chip_response(qwen_text):
            report = parse_on_chip_summary(qwen_text)
            if _report_has_timeline_detail(report):
                return _clean_refined_report(report)
            retry_prompt = _build_refine_prompt(
                on_chip_text,
                timeline_evidence,
                person_id=person_id,
                previous_answer=qwen_text,
            )
            retry_text = _run_qwen_postprocess(retry_prompt, model_name=model_name)
            if retry_text and is_valid_on_chip_response(retry_text):
                retry_report = parse_on_chip_summary(retry_text)
                if _report_has_timeline_detail(retry_report):
                    return _clean_refined_report(retry_report)
                return _build_timeline_report(log_text, person_id=person_id, base_report=retry_report)
            return _build_timeline_report(log_text, person_id=person_id, base_report=report)
    except Exception as exc:
        print(f"[QwenPostprocess] Falling back to rule parser: {exc}")

    fallback_text = on_chip_text if is_valid_on_chip_response(on_chip_text) else ""
    fallback_report = parse_on_chip_summary(fallback_text)
    if _report_has_timeline_detail(fallback_report):
        return _clean_refined_report(fallback_report)
    return _build_timeline_report(log_text, person_id=person_id, base_report=fallback_report)


def render_refined_on_chip_summary(
    on_chip_text: str,
    log_text: str = "",
    person_id: int = None,
    model_name: str = QWEN_POSTPROCESS_MODEL_NAME,
) -> str:
    return render_summary_cards(
        refine_on_chip_summary(
            on_chip_text=on_chip_text,
            log_text=log_text,
            person_id=person_id,
            model_name=model_name,
        )
    )


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
