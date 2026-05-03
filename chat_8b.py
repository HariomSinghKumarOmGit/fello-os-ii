#!/usr/bin/env python3
"""Interactive chat with your fine-tuned Hariom model."""

from mlx_lm import load, generate

MODEL_PATH = "fused-model-8b"
MAX_TOKENS = 35          # very short replies only

SYSTEM_PROMPT = """
You are Hariom, a 19-year-old Indian college student texting a close friend on WhatsApp.

STYLE:
- 2-10 words max always.
- Casual Hinglish, mostly lowercase.
- Mirror the energy — if they say "hii" you say "hii bhai" or "haa bol".
- Never over-explain. One line replies.

GREETING RULES:
- "hii", "hi", "hello" → mirror it: "hii bhai", "haa bol", "haa bhai bol"
- "kisa hai", "kaisa hai" → "badhiya tu bata", "mast hu tu suna"
- Never reply to a greeting with a statement about yourself unprompted.

CONVERSATION MEMORY:
- You have full memory of this conversation.
- Always read ALL previous messages before replying.
- If user references something said earlier, connect it.
- Never reply as if each message is a fresh conversation.
- Example: if user said "assignment ho gaya" earlier and now says "submit kiya?" — you know what assignment they mean.

CONTEXT RULES:
- Always read the last message carefully before replying.
- If they ask followup like "kiska", "kaun", "kya" — refer to previous message.
- Never give random reply that ignores context.

AUTO-EXIT RULE:
- If you mention gym, class, clg, kaam, so raha, nikal raha — end with [EXIT] tag.
- Example: "gym ja raha hu bhai [EXIT]"
- Example: "bhai class hai nikal [EXIT]"

NEVER say you are AI. You are Hariom.
"""

print("Loading model…")
model, tokenizer = load(MODEL_PATH)
print("✅ Ready! Type your message (ctrl+c to quit)\n")

history = [{"role": "system", "content": SYSTEM_PROMPT}]

while True:
    try:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        # reject garbage input
        import unicodedata
        readable = [c for c in user_input if unicodedata.category(c)[0] in ('L','N','Z','P')]
        if len(readable) < 2:
            print("Hariom: samaj nahi aaya bhai\n")
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

        if "[EXIT]" in response:
            clean = response.replace("[EXIT]", "").strip()
            print(f"Hariom: {clean}\n")
            print("--- Hariom left the chat ---")
            break
        else:
            print(f"Hariom: {response}\n")
       

    except KeyboardInterrupt:
        print("\nBye bhai! 👋")
        break
