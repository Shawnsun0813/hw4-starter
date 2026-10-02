tiny_lines = [
    """In 2026, we teach “intro-to-AI” with hands-on labs—no hype. Students ask: “Why tokens?” 
Because models read pieces, not words. E.g., ‘ChatGPT-5’ ≠ ‘Chat’, ‘GPT’, ‘5’ in all schemes.
We track loss/accuracy, compare char/word/BPE, and test a URL: https://example.org/a/b?c=42.
Café prices rose 3.7%—blame supply-chain weirdness (and ☕ demand).
"""
]
text = "\n".join(tiny_lines)

# Alternative corpora (uncomment one):
# DNA: text = "TATAAA\nCGCGCG\nATG...TAA\nACGTACGTACGT\n"
# Emoji: text = "☀️🌤️⛅🌧️⛈️🌈\n🍞🧈🍯\n🥚🍳🍞\n🙂➡️😊\n"
# Nursery: text = "Twinkle twinkle little star,\nHow I wonder what you are.\n"

print("Corpus length:", len(text))
chars = sorted(list(set(text)))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
vocab_size = len(chars)

import re
# Phase 1: lowercase...
lower_text = text.lower()
clean_text = re.sub(r"[^\w\s]", " ", lower_text)
words = clean_text.split()

print(words)

# Phase 2: create unique vocabulary
vocab = sorted(set(words))

print(vocab)
print("Vocabulary size:", len(vocab))

# Phase 3: word -> integer
stoi = {word: i for i, word in enumerate(vocab)}

print(stoi)

# Phase 4: integer -> word
itos = {i: word for word, i in stoi.items()}

print(itos)

# Phase 5: build a new sentence word by word
new_sentence = []

for item in vocab:
    new_sentence.append(item)

print(new_sentence)

chars1 = sorted(list(set(new_sentence)))
vocab_size = len(chars1)
print(vocab_size)
