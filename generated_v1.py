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
TEMPERATURES = [0.2, 0.5, 0.8, 1.0, 1.2]


tf.keras.utils.set_random_seed(42)

characters = sorted(set("".join(SENTENCES)))
char_to_id = {character: index + 1 for index, character in enumerate(characters)}
id_to_char = {index: character for character, index in char_to_id.items()}
padding_id = 0
vocabulary_size = len(char_to_id) + 1


def encode_context(text):
    context = [char_to_id[character] for character in text[-INPUT_LENGTH:]]
    return [padding_id] * (INPUT_LENGTH - len(context)) + context


training_inputs = []
training_targets = []
for sentence in SENTENCES:
    for next_index, next_character in enumerate(sentence):
        prefix = sentence[:next_index]
        training_inputs.append(encode_context(prefix))
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
model.fit(training_inputs, training_targets, epochs=EPOCHS, verbose=0)


def sample_character(probabilities, temperature):
    if temperature <= 0:
        raise ValueError("Temperature must be greater than zero.")

    probabilities = probabilities[1:]
    logits = np.log(probabilities + 1e-8) / temperature
    logits -= np.max(logits)
    scaled_probabilities = np.exp(logits)
    scaled_probabilities /= np.sum(scaled_probabilities)
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


for temperature in TEMPERATURES:
    print(f"Temperature {temperature:.1f}: {generate_sentence(temperature)}")
