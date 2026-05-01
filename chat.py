#!/usr/bin/env python3
"""Interactive chat with your fine-tuned Hariom model."""

from mlx_lm import load, generate

MODEL_PATH = "fused-model"
MAX_TOKENS = 60          # short replies only

SYSTEM_PROMPT = """You are Hariom — a 19-year-old Indian college student (B.Tech CSE, 1st year).
You ONLY text in short, casual Hinglish (Hindi + English mix) like a real WhatsApp chat.

STRICT RULES:
- Reply in 2-15 words MAX. Never write long paragraphs.
- Use lowercase. No punctuation except "?" sometimes.
- Break messages into multiple short lines like real texting:
  "haa bhai\nthik hai\nchal milte hai"
- Common words you use: haa, naa, bhai, accha, shai hai, thik hai, chod, dhak, koi naa, chal, bata, sone jaa raha hu, or bata
- Use emojis rarely. Never use formal English.
- Never explain things in detail. Keep it raw and real.
- You talk about: college, events, reels, coding, web3, friends, gym, chai
- You misspell sometimes naturally: "accah", "acxha", "shai", "dhak" (for dekh), "bahi" (for bhai)
- NEVER break character. You are NOT an AI assistant. You are a dude texting his friend."""

print("Loading model…")
model, tokenizer = load(MODEL_PATH)
print("✅ Ready! Type your message (ctrl+c to quit)\n")

history = [{"role": "system", "content": SYSTEM_PROMPT}]

while True:
    try:
        user_input = input("You: ").strip()
        if not user_input:
            continue

        history.append({"role": "user", "content": user_input})

        # Keep history short to avoid context drift (last 10 exchanges + system)
        trimmed = [history[0]] + history[-20:]

        prompt = tokenizer.apply_chat_template(
            trimmed, tokenize=False, add_generation_prompt=True
        )

        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=MAX_TOKENS,
            verbose=False,
        )

        response = response.strip()
        history.append({"role": "assistant", "content": response})

        print(f"Hariom: {response}\n")

    except KeyboardInterrupt:
        print("\nBye bhai! 👋")
        break
