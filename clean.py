import json
import random

# Filter step
with open('train.jsonl', 'r') as f:
    lines = f.readlines()

filtered = []
for line in lines:
    d = json.loads(line)
    # The data uses "prompt" and "completion"
    reply = d.get('completion', '')
    word_count = len(reply.split())
    if 2 <= word_count <= 20:  # keep only short replies
        filtered.append(line)

print(f'kept {len(filtered)} short pairs out of {len(lines)} total')

# Shuffle and split step
random.seed(42)
random.shuffle(filtered)

split = int(len(filtered) * 0.9)
train_lines = filtered[:split]
valid_lines = filtered[split:]
test_lines = valid_lines[:20]  # Just grab first 20 of valid for test

with open('train.jsonl', 'w') as f:
    f.writelines(train_lines)

with open('valid.jsonl', 'w') as f:
    f.writelines(valid_lines)

with open('test.jsonl', 'w') as f:
    f.writelines(test_lines)

print(f'train: {len(train_lines)}, valid: {len(valid_lines)}')
