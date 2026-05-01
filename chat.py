#!/usr/bin/env python3
"""Interactive chat with your fine-tuned Hariom model."""

from mlx_lm import load, generate

MODEL_PATH = "fused-model"
MAX_TOKENS = 200

print("Loading model…")
model, tokenizer = load(MODEL_PATH)
print("✅ Ready! Type your message (ctrl+c to quit)\n")

history = []

while True:
    try:
        user_input = input("You: ").strip()
        if not user_input:
            continue

        history.append({"role": "user", "content": user_input})

        prompt = tokenizer.apply_chat_template(
            history, tokenize=False, add_generation_prompt=True
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
