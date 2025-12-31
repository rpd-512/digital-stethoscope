import os
import re
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, stft, resample_poly
from sys import argv
from tqdm import tqdm

# ===================== parameters =====================

INPUT_DIR = argv[1]
OUTPUT_DIR = "DatasetGenerated"
FNAME_PREFIX = "HB"

SR = 4000
BLOCK_SEC = 3.0
BLOCK = int(SR * BLOCK_SEC)

LOW = 30
HIGH = 500
ENV_CUTOFF = 8

NFFT = 512
OVERLAP = 448

AMP_LIMIT = 0.1

# =====================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- index handling ----------------
pattern = re.compile(rf"{FNAME_PREFIX}_k(\d+)_spec\.npy")
existing = [
    int(m.group(1)) for f in os.listdir(OUTPUT_DIR)
    if (m := pattern.match(f))
]
k_global = max(existing) + 1 if existing else 0
print(f"[INFO] Starting from k{k_global:03d}")

# ---------------- filters (SAME AS LIVE) ----------------
bp_sos = butter(4, [LOW, HIGH], btype="band", fs=SR, output="sos")
env_sos = butter(2, ENV_CUTOFF, btype="low", fs=SR, output="sos")

# ---------------- WAV files ----------------
wav_files = sorted(f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".wav"))
if not wav_files:
    raise RuntimeError("No WAV files found")

# ===================== processing =====================
for wav in tqdm(wav_files, desc="WAV files"):
    audio, fs = sf.read(os.path.join(INPUT_DIR, wav))

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if fs != SR:
        audio = resample_poly(audio, SR, fs)

    num_blocks = len(audio) // BLOCK

    for _ in tqdm(range(num_blocks), desc="Blocks", leave=False):
        x = audio[:BLOCK]
        audio = audio[BLOCK:]

        # -------- bandpass --------
        x = sosfiltfilt(bp_sos, x)

        # -------- HARD CLIP (critical) --------
        x = np.clip(x, -AMP_LIMIT, AMP_LIMIT)

        if np.max(np.abs(x)) < 0.02 * AMP_LIMIT:
            continue

        # -------- envelope (RECTIFY + LPF) --------
        env = np.abs(x)
        env = sosfiltfilt(env_sos, env)
        env = np.clip(env, 0, AMP_LIMIT)

        # -------- STFT --------
        f, t, Z = stft(
            x,
            fs=SR,
            nperseg=NFFT,
            noverlap=OVERLAP,
            window="hann",
            boundary=None,
            padded=False
        )

        P = np.abs(Z) ** 2
        S = 10 * np.log10(P + 1e-12)

        # KEEP ONLY 0–500 Hz
        freq_mask = f <= HIGH
        f = f[freq_mask]
        S = S[freq_mask, :]


        # -------- POWER spectrogram (MATCH specgram) --------
        P = np.abs(Z) ** 2
        S = 10 * np.log10(P + 1e-12)

        # -------- save ML --------
        np.save(f"{OUTPUT_DIR}/{FNAME_PREFIX}_k{k_global:03d}_spec.npy", S.astype(np.float32))
        np.save(f"{OUTPUT_DIR}/{FNAME_PREFIX}_k{k_global:03d}_env.npy", env.astype(np.float32))

        # -------- visualization --------
        plt.figure(figsize=(6, 4))
        plt.imshow(
            S,
            origin="lower",
            aspect="auto",
            extent=[t[0], t[-1], f[0], f[-1]],
            cmap="inferno"
        )
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.colorbar(label="Power (dB)")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/{FNAME_PREFIX}_k{k_global:03d}_spec.png", dpi=150)
        plt.close()

        k_global += 1

print(f"[DONE] Dataset contains samples up to k{k_global-1:03d}")
