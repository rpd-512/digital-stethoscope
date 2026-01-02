import os
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# ===================== CONFIG =====================

DATA_ROOT = "../../Spectrogram/DatasetGenerated/NewData/"

CLASS_MAP = {
    "Not_Heartbeat": 0,
    "Heartbeat": 1
}

def weighted_bce(pos_weight=1.0, neg_weight=1.0):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        weight = y_true * pos_weight + (1.0 - y_true) * neg_weight
        return tf.reduce_mean(bce * weight)
    return loss

FNAME_PREFIX = "HB"

BATCH_SIZE = 16
EPOCHS = 10
VAL_SPLIT = 0.2
SEED = 137

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

# ===================== SMALLER MODEL =====================

# ===================== INPUTS =====================

# Spectrogram: (F, T, 1) with T = ANY
spec_input = layers.Input(shape=(F, None, 1), name="spec_input")

# Envelope: (T, 1) with T = SAME time axis
env_input = layers.Input(shape=(None, 1), name="env_input")

# ===================== SPECTROGRAM BRANCH =====================

x = layers.Conv2D(16, (5, 5), padding="same", activation="relu")(spec_input)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2))(x)

x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2))(x)

x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
x = layers.BatchNormalization()(x)

# Collapse frequency, keep time
x = layers.GlobalAveragePooling2D(data_format="channels_last")(x)
spec_feat = layers.Dense(64, activation="relu")(x)

# ===================== ENVELOPE BRANCH (TCN-style) =====================

y = layers.Conv1D(16, 9, padding="same", activation="relu")(env_input)
y = layers.BatchNormalization()(y)

y = layers.Conv1D(32, 7, padding="same", activation="relu")(y)
y = layers.BatchNormalization()(y)

y = layers.Conv1D(64, 5, padding="same", activation="relu")(y)
y = layers.BatchNormalization()(y)

y = layers.GlobalAveragePooling1D()(y)
env_feat = layers.Dense(64, activation="relu")(y)

# ===================== FUSION =====================

combined = layers.Concatenate()([spec_feat, env_feat])

z = layers.Dense(64, activation="relu")(combined)
z = layers.Dropout(0.3)(z)

z = layers.Dense(32, activation="relu")(z)
z = layers.Dropout(0.2)(z)

output = layers.Dense(1, activation="sigmoid")(z)

model = models.Model(
    inputs=[spec_input, env_input],
    outputs=output
)

model.compile(
    optimizer=tf.keras.optimizers.AdamW(
        learning_rate=2e-4,
        weight_decay=1e-4
    ),
    loss=weighted_bce(pos_weight=1.0, neg_weight=1.23),
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

model.save("model_006.keras")
print("Model saved.")


from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

def evaluate_confusion_matrix(model, samples, batch_size=16, threshold=0.5):
    y_true = []
    y_pred = []

    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]

        X_spec = np.zeros((len(batch), F, T, 1), dtype=np.float32)
        X_env  = np.zeros((len(batch), ENV_T, 1), dtype=np.float32)
        y      = np.zeros((len(batch),), dtype=np.int32)

        for j, (spec_path, env_path, label) in enumerate(batch):
            X_spec[j, ..., 0] = np.load(spec_path)
            X_env[j, :, 0] = np.load(env_path)
            y[j] = label

        probs = model.predict((X_spec, X_env), verbose=0).ravel()
        preds = (probs >= threshold).astype(int)

        y_true.extend(y)
        y_pred.extend(preds)

    return np.array(y_true), np.array(y_pred)

THRESHOLD = 0.65  # <-- tune this to reduce false positives

y_true, y_pred = evaluate_confusion_matrix(
    model,
    val_samples,
    batch_size=BATCH_SIZE,
    threshold=THRESHOLD
)


cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix (threshold =", THRESHOLD, "):")
print(cm)

tn, fp, fn, tp = cm.ravel()

print(f"""
TN (Correct Not_Heartbeat): {tn}
FP (False Heartbeat):       {fp}
FN (Missed Heartbeat):      {fn}
TP (Correct Heartbeat):     {tp}
""")

print("Classification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=["Not_Heartbeat", "Heartbeat"]
))

import matplotlib.pyplot as plt

probs = []

for i in range(0, len(val_samples), BATCH_SIZE):
    batch = val_samples[i:i + BATCH_SIZE]
    X_spec = np.zeros((len(batch), F, T, 1))
    X_env  = np.zeros((len(batch), ENV_T, 1))

    for j, (spec_path, env_path, _) in enumerate(batch):
        X_spec[j, ..., 0] = np.load(spec_path)
        X_env[j, :, 0] = np.load(env_path)

    probs.extend(model.predict((X_spec, X_env), verbose=0).ravel())

plt.hist(probs, bins=50)
plt.axvline(THRESHOLD, color="r")
plt.title("Prediction Confidence Distribution")
plt.xlabel("Heartbeat probability")
plt.ylabel("Count")
plt.show()
