import sqlite3
import re
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

# model_name = "Qwen/Qwen3-32B"
model_name = "Qwen/Qwen1.5-0.5B"

print(f"Loading model: {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)


def _normalize_date(text):
    if not text:
        return None
    m = re.search(r'(\d{4})[/-](\d{2})[/-](\d{2})', text)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    m = re.search(r'(\d{2})[/-](\d{2})[/-](\d{4})', text)
    if m:
        return f"{m.group(3)}-{m.group(1)}-{m.group(2)}"
    return None


def get_person_activity_log(
    person_id,
    db_path="video_data.db",
    date_filter=None,
    include_bbox=False,
    video_names=None,
):
    """
    Extract data from the 'frames' table for a specific person.
    """
    if not Path(db_path).exists():
        print(f"Error: Database file not found at {db_path}")
        return None
        
    conn = sqlite3.connect(db_path)
    
    query = """
    SELECT ocr_time, timestamp, action, bbox, video_name
    FROM frames
    WHERE person_id = ?
    ORDER BY video_name, frame_idx ASC
    """
    
    try:
        # Use pandas to read the query results
        df = pd.read_sql_query(query, conn, params=(person_id,))
    except Exception as e:
        print(f"SQL Execution failed: {e}")
        conn.close()
        return None
    finally:
        conn.close()

    if df.empty:
        print(f"Reminder: No records found for ID {person_id} in the database.")
        return None

    # 2. Data Cleaning and Compression
    # Filter out buffering or invalid status labels
    invalid_labels = ["buffering", "None", "Unknown", "Buffering", "Uncertain", ""]
    df = df[~df['action'].isin(invalid_labels)]

    if df.empty:
        print(f"Reminder: All records for ID {person_id} contain invalid action labels.")
        return None

    if date_filter:
        normalized_filter = _normalize_date(date_filter) or date_filter
        def _match_date(ocr_text):
            normalized = _normalize_date(ocr_text)
            return normalized == normalized_filter
        df = df[df['ocr_time'].apply(_match_date)]
        if df.empty:
            print(f"Reminder: No records for ID {person_id} match date {date_filter}.")
            return None
    if video_names:
        name_set = set(video_names)
        stem_set = set([Path(v).stem for v in video_names])
        df = df[df["video_name"].isin(name_set) | df["video_name"].isin(stem_set)]
        if df.empty:
            print(f"Reminder: No records for ID {person_id} match specified videos.")
            return None

    # Action deduplication and merging (State Compression)
    compressed_log = []
    last_action = None
    
    for _, row in df.iterrows():
        # Prioritize OCR time; if not identified, fallback to relative timestamp
        # Check if raw_ocr is a non-empty string as per dbmanager defaults
        raw_ocr = row['ocr_time']
        current_time = raw_ocr if (raw_ocr and raw_ocr.strip()) else f"T+{row['timestamp']:.1f}s"
        
        if row['action'] != last_action:
            if include_bbox:
                compressed_log.append(f"[{current_time}] {row['action']} {row['bbox']}")
            else:
                compressed_log.append(f"[{current_time}] {row['action']}")
            last_action = row['action']
            
    return "\n".join(compressed_log)


def get_person_action_timeline(person_id, db_path="video_data.db", date_filter=None, video_names=None):
    """
    Return per-second action timeline based on ocr_time and timestamp.
    """
    if not Path(db_path).exists():
        return None
    conn = sqlite3.connect(db_path)
    query = """
    SELECT ocr_time, timestamp, action, video_name
    FROM frames
    WHERE person_id = ?
    ORDER BY video_name, frame_idx ASC
    """
    try:
        df = pd.read_sql_query(query, conn, params=(person_id,))
    finally:
        conn.close()
    if df.empty:
        return None

    invalid_labels = ["buffering", "None", "Unknown", "Buffering", "Uncertain", ""]
    df = df[~df['action'].isin(invalid_labels)]
    if df.empty:
        return None

    if date_filter:
        normalized_filter = _normalize_date(date_filter) or date_filter
        def _match_date(ocr_text):
            normalized = _normalize_date(ocr_text)
            return normalized == normalized_filter
        df = df[df['ocr_time'].apply(_match_date)]
        if df.empty:
            return None

    if video_names:
        name_set = set(video_names)
        stem_set = set([Path(v).stem for v in video_names])
        df = df[df["video_name"].isin(name_set) | df["video_name"].isin(stem_set)]
        if df.empty:
            return None

    # floor timestamp to second
    df["sec"] = df["timestamp"].apply(lambda x: int(x))
    # take most frequent action per second
    timeline = []
    for sec, group in df.groupby("sec"):
        action = group["action"].mode().iloc[0]
        timeline.append((sec, action))
    timeline.sort(key=lambda x: x[0])

    # format with ocr_time if available
    ocr_time = df["ocr_time"].iloc[0] if df["ocr_time"].iloc[0] else ""
    lines = []
    for sec, action in timeline:
        label = f"{ocr_time} T+{sec:04d}s" if ocr_time else f"T+{sec:04d}s"
        lines.append(f"[{label}] {action}")
    return "\n".join(lines)


def query_qwen_summary(person_id, log_text):
    if not log_text:
        return "No activity log found."
    
    print(log_text)
    # Prompt with specific requirements for time periods and evidence
    prompt = f"""
    CONTEXT: You are a geriatric care forensic analyst. Generate a daily report for Person ID: {person_id}.
    
    RAW DATA (Time-sequenced):
    {log_text}
    
    STRICT INSTRUCTIONS:
    1. **NO HALLUCINATION**: Every activity MUST have a corresponding timestamp from the RAW DATA.
    2. **PRECISE TIME RANGES**: In the 'daily_routine_highlights', you must specify the exact start and end times (e.g., "09:15 - 10:45") for major activities.
    3. **TEMPORAL CONTINUITY**: If a person is 'sitting' for 40 minutes with a 2-second 'walking' glitch in the middle, ignore the 'walking' as sensor noise.
    4. **EVIDENCE TAGGING**: Every 'care_note' must cite the specific key actions seen in the data.
    
    OUTPUT FORMAT: Respond ONLY with a JSON object.
    {{
        "person_id": {person_id},
        "caregiver_report": {{
            "daily_routine_highlights": [
                {{
                    "period": "Morning",
                    "time_range": "e.g., 08:00 - 11:30",
                    "event": "Detailed description of activity flow",
                    "care_note": "Evidence-based observation (e.g., 'Stable gait observed during 09:00 walk')"
                }},
                {{
                    "period": "Afternoon",
                    "time_range": "e.g., 12:00 - 17:00",
                    "event": "...",
                    "care_note": "..."
                }},
                {{
                    "period": "Evening",
                    "time_range": "...",
                    "event": "...",
                    "care_note": "..."
                }}
            ],
            "safety_assessment": {{
                "fall_risk": "Low/Medium/High (based on data)",
                "sedentary_risk": "Warning if immobility > 2hrs (citing timestamps)"
            }},
            "anomalies": ["List specific timestamps and suspicious actions, or 'None'"],
            "handoff_instructions": ["Actionable advice for the next shift caregiver"]
        }}
    }}
    """

    messages = [
        {"role": "system", "content": "You are a data-driven care assistant. You provide precise time-stamped reports and ignore sensor noise."},
        {"role": "user", "content": prompt}
    ]

    # --- Inference Logic ---
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=32768)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    # Parse Chain-of-Thought (thinking) process and the JSON output
    try:
        # Looking for special token to separate thinking from response
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip()
    final_json = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()

    return thinking_content, final_json


if __name__ == "__main__":
    TARGET_ID = 7  # ID of the elderly person to analyze
    
    print(f"Extracting activity records for ID {TARGET_ID}...")
    activity_log = get_person_activity_log(TARGET_ID, db_path='/data/lllidy/Projects/healthcare_benchmark/tool/original_person.db')

    if activity_log:
        print(f"Valid records found. Calling Qwen3 to generate health daily report...\n")
        thinking, summary = query_qwen_summary(TARGET_ID, activity_log)
        
        print("-" * 20 + " Model Thinking Process " + "-" * 20)
        print(thinking)
        print("\n" + "=" * 20 + f" Activity Summary Report for ID {TARGET_ID} " + "=" * 20)
        print(summary)
    else:
        print(f"No valid data found for ID {TARGET_ID} in the database.")
