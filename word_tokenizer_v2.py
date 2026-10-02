import re

tiny_lines = [
    """In 2026, we teach “intro-to-AI” with hands-on labs—no hype. Students ask: “Why tokens?” 
Because models read pieces, not words. E.g., ‘ChatGPT-5’ ≠ ‘Chat’, ‘GPT’, ‘5’ in all schemes.
We track loss/accuracy, compare char/word/BPE, and test a URL: https://example.org/a/b?c=42. 
Café prices rose 3.7%—blame supply-chain weirdness (and ☕ demand). 
"""
]

text = "\n".join(tiny_lines)


# ---------------- Character-level tokenizer ----------------

char_vocab = sorted(set(text))

char_stoi = {char: i for i, char in enumerate(char_vocab)}
char_itos = {i: char for char, i in char_stoi.items()}

print("Character tokenized length:", len(text))
print("Character vocabulary size:", len(char_vocab))


# ---------------- Word-level tokenizer ----------------

# Phase 1: lowercase, remove punctuation, split by whitespace
lower_text = text.lower()
clean_text = re.sub(r"[^\w\s]", " ", lower_text)
words = clean_text.split()

print("\nTokens:")
print(words)

print("\nWord tokenized length:", len(words))


# Phase 2: build vocabulary of unique words
word_vocab = sorted(set(words))

print("\nWord vocabulary:")
print(word_vocab)
print("Word vocabulary size:", len(word_vocab))


# Phase 3: word -> integer
word_stoi = {word: i for i, word in enumerate(word_vocab)}

print("\nWord to integer:")
print(word_stoi)


# Phase 4: integer -> word
word_itos = {i: word for word, i in word_stoi.items()}

print("\nInteger to word:")
print(word_itos)


# Phase 5: encode original tokens, then decode them
encoded = [word_stoi[word] for word in words]

print("\nEncoded tokens:")
print(encoded)

decoded = [word_itos[i] for i in encoded]

new_sentence = " ".join(decoded)

print("\nDecoded sentence:")
print(new_sentence)
