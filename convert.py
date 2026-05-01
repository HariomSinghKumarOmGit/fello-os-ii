#!/usr/bin/env python3
"""
WhatsApp Chat → JSONL Training-Pair Converter
Reads a WhatsApp exported _chat.txt and produces prompt/completion
pairs where the OTHER person's consecutive messages become the prompt
and YOUR consecutive replies become the completion.
"""

import re, json, sys

# ── CONFIG ──────────────────────────────────────────────────────────
INPUT_FILE  = "/Users/hariomsingh/Desktop/felloos-ii/c-export/_chat.txt"
OUTPUT_FILE = "train.jsonl"
YOUR_NAME   = "Hariom"
# ────────────────────────────────────────────────────────────────────

# WhatsApp line pattern (handles both 12h and 24h timestamps)
LINE_RE = re.compile(
    r"^\[(\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?)\]\s+"
    r"(.+?):\s(.*)",
    re.IGNORECASE,
)

SKIP_PATTERNS = [
    "image omitted",
    "audio omitted",
    "video omitted",
    "document omitted",
    "sticker omitted",
    "GIF omitted",
    "Contact card omitted",
    "location:",
    "You deleted this message",
    "This message was deleted",
    "Messages and calls are end-to-end encrypted",
    "<Media omitted>",
]


def is_junk(text: str) -> bool:
    """Return True if the message is a media placeholder or system line."""
    stripped = text.strip()
    if not stripped:
        return True
    for pat in SKIP_PATTERNS:
        if pat.lower() in stripped.lower():
            return True
    # skip bare URLs with no real text
    if re.match(r"^https?://\S+$", stripped):
        return True
    return False


def parse_chat(path: str):
    """Yield (sender, text) tuples from the exported chat file."""
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip("\n\r\ufeff")
            m = LINE_RE.match(line)
            if m:
                sender = m.group(2).strip()
                text   = m.group(3).strip()
                # Remove the "‎" (U+200E left-to-right mark) WhatsApp sprinkles
                text = text.replace("\u200e", "").strip()
                if not is_junk(text):
                    yield sender, text


def build_pairs(messages, your_name: str):
    """
    Walk through messages and create prompt/completion pairs.
    Consecutive messages from the OTHER person → prompt.
    Consecutive messages from YOU after that   → completion.
    """
    pairs = []
    prompt_buf  = []
    reply_buf   = []
    state = "collecting_prompt"  # or "collecting_reply"

    for sender, text in messages:
        is_you = (sender.strip().lower() == your_name.lower())

        if state == "collecting_prompt":
            if not is_you:
                prompt_buf.append(text)
            else:
                # transition: start collecting your reply
                if prompt_buf:
                    reply_buf = [text]
                    state = "collecting_reply"
                # else: you spoke first with no prompt, skip
        else:  # collecting_reply
            if is_you:
                reply_buf.append(text)
            else:
                # transition: flush the completed pair, start new prompt
                if prompt_buf and reply_buf:
                    pairs.append((
                        "\n".join(prompt_buf),
                        "\n".join(reply_buf),
                    ))
                prompt_buf = [text]
                reply_buf  = []
                state = "collecting_prompt"

    # flush final pair
    if prompt_buf and reply_buf:
        pairs.append(("\n".join(prompt_buf), "\n".join(reply_buf)))

    return pairs


def main():
    messages = list(parse_chat(INPUT_FILE))
    print(f"📨  Parsed {len(messages)} usable messages from chat.")

    pairs = build_pairs(messages, YOUR_NAME)
    print(f"✅  Built {len(pairs)} prompt → completion pairs.")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for prompt, completion in pairs:
            obj = {"prompt": prompt, "completion": " " + completion}
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"💾  Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
