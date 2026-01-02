import os
import sys
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, hilbert, resample_poly, stft
from math import gcd
import re
from tqdm import tqdm

# ===================== USER PARAMETERS =====================

FNAME_PREFIX = "HB"

SR = 4000
BLOCK_SEC = 3.0

LOW = 30
HIGH = 500

ENV_CUTOFF = 8

NFFT = 512
OVERLAP = 448

AMP_LIMIT = 1

OUT_DIR = "DatasetGenerated/NewData/Heartbeat"

# ===========================================================

BLOCK = int(SR * BLOCK_SEC)

# ------------------ sanity check ------------------
if len(sys.argv) < 2:
    print("Usage: python generate_dataset.py <audiofile.wav | directory>")
    sys.exit(1)

input_path = sys.argv[1]
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------ collect WAV files ------------------
if os.path.isdir(input_path):
    wav_files = sorted(
        os.path.join(input_path, f)
        for f in os.listdir(input_path)
        if f.lower().endswith(".wav")
    )
elif os.path.isfile(input_path):
    wav_files = [input_path]
else:
    raise ValueError("Input must be a WAV file or a directory")

if not wav_files:
    raise ValueError("No WAV files found")

# ------------------ find last used index ------------------
pattern = re.compile(rf"{FNAME_PREFIX}_spec_(\d+)\.npy")

existing = [
    int(pattern.search(f).group(1))
    for f in os.listdir(OUT_DIR)
    if pattern.search(f)
]

start_index = max(existing) + 1 if existing else 0
global_index = start_index

print(f"Starting index: {global_index}")

# ------------------ filters ------------------
bp_sos = butter(4, [LOW, HIGH], btype="band", fs=SR, output="sos")
env_sos = butter(2, ENV_CUTOFF, btype="low", fs=SR, output="sos")

# ================== PROCESS FILES ==================
for audio_path in tqdm(wav_files, desc="Processing files"):

    print(f"Processing: {audio_path}")

    sig, file_sr = sf.read(audio_path)

    if sig.ndim > 1:
        sig = sig[:, 0]

    # -------- resample if needed --------
    if file_sr != SR:
        g = gcd(file_sr, SR)
        sig = resample_poly(sig, SR // g, file_sr // g)
        print(f"  Resampled {file_sr} Hz → {SR} Hz")

    num_blocks = len(sig) // BLOCK

    for b in tqdm(range(1,num_blocks), desc="Processing blocks"):
        block = sig[b * BLOCK:(b + 1) * BLOCK]

        # -------- bandpass --------
        filt = sosfiltfilt(bp_sos, block)

        # -------- envelope --------
        envelope = np.abs(hilbert(filt))
        envelope = sosfiltfilt(env_sos, envelope)
        envelope = np.clip(envelope, 0, AMP_LIMIT)

        # -------- STFT --------
        freqs, bins, Zxx = stft(
            filt,
            fs=SR,
            nperseg=NFFT,
            noverlap=OVERLAP,
            window="hann",
            boundary=None,
            padded=False
        )

        Pxx = np.abs(Zxx) ** 2
        Pxx_db = 10 * np.log10(Pxx + 1e-12)
        Pxx_norm = (Pxx_db - Pxx_db.min()) / (Pxx_db.max() - Pxx_db.min() + 1e-12)

        # -------- save tensors --------
        np.save(
            os.path.join(OUT_DIR, f"{FNAME_PREFIX}_spec_{global_index:04d}.npy"),
            Pxx_norm.astype(np.float32)
        )

        np.save(
            os.path.join(OUT_DIR, f"{FNAME_PREFIX}_env_{global_index:04d}.npy"),
            envelope.astype(np.float32)
        )

        # -------- save spectrogram image --------
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.imshow(
            Pxx_norm,
            origin="lower",
            aspect="auto",
            extent=[0, BLOCK_SEC, freqs[0], freqs[-1]],
            cmap="inferno"
        )
        ax.set_ylim(0, 200)
        ax.axis("off")
        fig.savefig(
            os.path.join(OUT_DIR, f"{FNAME_PREFIX}_spec_{global_index:04d}.png"),
            dpi=200,
            bbox_inches="tight",
            pad_inches=0
        )
        plt.close(fig)

        # -------- save envelope image --------
        t = np.linspace(0, BLOCK_SEC, len(envelope))
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.plot(t, envelope, color="black", linewidth=1)
        ax.set_ylim(0, AMP_LIMIT)
        fig.savefig(
            os.path.join(OUT_DIR, f"{FNAME_PREFIX}_env_{global_index:04d}.png"),
            dpi=200,
            bbox_inches="tight",
            pad_inches=0
        )
        plt.close(fig)

        global_index += 1

print(f"Done. Generated {global_index - start_index} new blocks.")
