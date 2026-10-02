import numpy as np
import tensorflow as tf


SENTENCES = [
    "I like cats.",
    "I like dogs.",
    "I like noodles.",
]
PROMPT = "I like"
INPUT_LENGTH = 40
MAX_OUTPUT_LENGTH = 39
EPOCHS = 20
TEMPERATURES = [0.1, 0.5, 0.7, 1.0, 1.2]
NUM_GENERATIONS = 15


tf.keras.utils.set_random_seed(42)
np.random.seed(42)

corpus = "\n".join(SENTENCES)
characters = sorted(set(corpus))
char_to_id = {character: index + 1 for index, character in enumerate(characters)}
id_to_char = {index: character for character, index in char_to_id.items()}
padding_id = 0
vocabulary_size = len(char_to_id) + 1


def encode_context(text):
    context = [char_to_id[character] for character in text[-INPUT_LENGTH:]]
    return [padding_id] * (INPUT_LENGTH - len(context)) + context


training_inputs = []
training_targets = []
for next_index, next_character in enumerate(corpus):
    training_inputs.append(encode_context(corpus[:next_index]))
    training_targets.append(char_to_id[next_character])

training_inputs = np.asarray(training_inputs, dtype=np.int32)
training_targets = np.asarray(training_targets, dtype=np.int32)

model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(INPUT_LENGTH,), dtype=tf.int32),
        tf.keras.layers.Embedding(
            input_dim=vocabulary_size,
            output_dim=32,
            mask_zero=True,
        ),
        tf.keras.layers.LSTM(128),
        tf.keras.layers.Dense(vocabulary_size, activation="softmax"),
    ]
)
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
history = model.fit(training_inputs, training_targets, epochs=EPOCHS, verbose=0)


def temperature_probabilities(probabilities, temperature):
    if temperature <= 0:
        raise ValueError("Temperature must be greater than zero.")

    logits = np.log(probabilities[1:] + 1e-8) / temperature
    if "\n" in char_to_id:
        logits[char_to_id["\n"] - 1] = -np.inf
    logits -= np.max(logits)
    scaled_probabilities = np.exp(logits)
    return scaled_probabilities / np.sum(scaled_probabilities)


def sample_character(probabilities, temperature):
    scaled_probabilities = temperature_probabilities(probabilities, temperature)
    sampled_id = np.random.choice(len(scaled_probabilities), p=scaled_probabilities) + 1
    return id_to_char[sampled_id]


def generate_sentence(temperature):
    output = PROMPT
    while len(output) < MAX_OUTPUT_LENGTH and not output.endswith("."):
        model_input = np.asarray([encode_context(output)], dtype=np.int32)
        probabilities = model.predict(model_input, verbose=0)[0]
        output += sample_character(probabilities, temperature)

    if not output.endswith("."):
        output = output[: MAX_OUTPUT_LENGTH - 1].rstrip() + "."
    return output


print(f"Corpus length: {len(corpus)}")
print(f"Vocab: {characters}")
print(f"Num training samples: {len(training_inputs)}")
print(f"Final loss: {history.history['loss'][-1]}")

for temperature in TEMPERATURES:
    print(f"\n=== Temperature {temperature:.1f} ===")
    prompt_input = np.asarray([encode_context(PROMPT)], dtype=np.int32)
    first_probabilities = model.predict(prompt_input, verbose=0)[0]
    adjusted_probabilities = temperature_probabilities(first_probabilities, temperature)
    top_indices = np.argsort(adjusted_probabilities)[-5:][::-1]
    print("Top probabilities for the first generated character:")
    for index in top_indices:
        character = id_to_char[index + 1]
        print(f"  {character!r}: {adjusted_probabilities[index] * 100:.1f}%")
    for _ in range(NUM_GENERATIONS):
        print(f"  {generate_sentence(temperature)}")
