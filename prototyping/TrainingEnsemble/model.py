import os
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# ===================== CONFIG =====================

DATA_ROOT = "../Spectrogram/DatasetGenerated"

CLASS_MAP = {
    "Not_Heartbeat": 0,
    "Heartbeat": 1
}

FNAME_PREFIX = "HB"

BATCH_SIZE = 16
EPOCHS = 20
VAL_SPLIT = 0.2
SEED = 42

# ==================================================

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ------------------ scan dataset ------------------

samples = []  # (spec_path, env_path, label)

spec_re = re.compile(rf"{FNAME_PREFIX}_spec_(\d+)\.npy")
env_re  = re.compile(rf"{FNAME_PREFIX}_env_(\d+)\.npy")

for class_name, label in CLASS_MAP.items():
    class_dir = os.path.join(DATA_ROOT, class_name)
    if not os.path.isdir(class_dir):
        continue

    spec_files = {}
    env_files = {}

    for f in os.listdir(class_dir):
        m = spec_re.match(f)
        if m:
            spec_files[m.group(1)] = f
        m = env_re.match(f)
        if m:
            env_files[m.group(1)] = f

    common_ids = sorted(set(spec_files) & set(env_files))

    for idx in common_ids:
        samples.append((
            os.path.join(class_dir, spec_files[idx]),
            os.path.join(class_dir, env_files[idx]),
            label
        ))

if not samples:
    raise RuntimeError("No valid samples found")

print(f"Total samples: {len(samples)}")

# ------------------ shuffle + split ------------------

np.random.shuffle(samples)

split = int(len(samples) * (1 - VAL_SPLIT))
train_samples = samples[:split]
val_samples   = samples[split:]

# ------------------ infer shapes ------------------

spec0 = np.load(train_samples[0][0])
env0  = np.load(train_samples[0][1])

F, T = spec0.shape
ENV_T = env0.shape[0]

print("Spectrogram shape:", spec0.shape)
print("Envelope length:", env0.shape)

# ------------------ data generator ------------------

def data_generator(samples, batch_size=BATCH_SIZE):
    while True:
        np.random.shuffle(samples)
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]

            X_spec = np.zeros((len(batch), F, T, 1), dtype=np.float32)
            X_env  = np.zeros((len(batch), ENV_T, 1), dtype=np.float32)
            y      = np.zeros((len(batch),), dtype=np.int32)

            for j, (spec_path, env_path, label) in enumerate(batch):
                X_spec[j, ..., 0] = np.load(spec_path)
                X_env[j, :, 0] = np.load(env_path)
                y[j] = label

            yield (X_spec, X_env), y

# ===================== MODEL =====================

# -------- CNN branch --------
spec_input = layers.Input(shape=(F, T, 1), name="spec_input")

x = layers.Conv2D(16, 3, activation="relu", padding="same")(spec_input)
x = layers.MaxPooling2D(2)(x)

x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
x = layers.MaxPooling2D(2)(x)

x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
x = layers.GlobalAveragePooling2D()(x)

spec_feat = layers.Dense(64, activation="relu")(x)

# -------- LSTM branch --------
env_input = layers.Input(shape=(ENV_T, 1), name="env_input")

y = layers.LSTM(64, return_sequences=True)(env_input)
y = layers.LSTM(32)(y)

env_feat = layers.Dense(64, activation="relu")(y)

# -------- Fusion --------
combined = layers.Concatenate()([spec_feat, env_feat])

z = layers.Dense(64, activation="relu")(combined)
z = layers.Dropout(0.3)(z)

output = layers.Dense(1, activation="sigmoid")(z)

model = models.Model(
    inputs=[spec_input, env_input],
    outputs=output
)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ===================== TRAIN =====================

model.fit(
    data_generator(train_samples),
    steps_per_epoch=len(train_samples) // BATCH_SIZE,
    validation_data=data_generator(val_samples),
    validation_steps=len(val_samples) // BATCH_SIZE,
    epochs=EPOCHS
)

# ------------------ save ------------------

model.save("cnn_lstm_heartbeat_classifier.keras")
print("Model saved.")
