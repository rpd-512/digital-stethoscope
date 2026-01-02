import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, stft
from matplotlib.animation import FuncAnimation

# ------------------ parameters ------------------
SR = 4000
BLOCK_SEC = 2.0

LOW = 30
HIGH = 500

ENV_CUTOFF = 8        # envelope smoothing cutoff (Hz)

NFFT = 512
OVERLAP = 448

AMP_LIMIT = 1

BLOCK = int(SR * BLOCK_SEC)

# ------------------ filters ------------------
bp_sos = butter(4, [LOW, HIGH], btype='band', fs=SR, output='sos')
env_sos = butter(2, ENV_CUTOFF, btype='low', fs=SR, output='sos')

# ------------------ buffer ------------------
buffer = np.zeros(BLOCK)

def audio_callback(indata, frames, time, status):
    global buffer
    sig = indata[:, 0]
    buffer = np.roll(buffer, -len(sig))
    buffer[-len(sig):] = sig

# ------------------ plot ------------------
fig, (ax_spec, ax_env) = plt.subplots(
    2, 1, figsize=(10, 6),
    gridspec_kw={"height_ratios": [3, 1]},
    sharex=True
)

# ------------------ update ------------------

def update(frame):
    ax_spec.clear()
    ax_env.clear()

    # -------- bandpass --------
    filt = sosfiltfilt(bp_sos, buffer)

    # -------- hard amplitude limit --------
    #filt = np.clip(filt, -AMP_LIMIT, AMP_LIMIT)

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

    # power spectrogram (same meaning as specgram's Pxx)
    Pxx = np.abs(Zxx) ** 2


    # -------- plot (manual, specgram-style) --------
    ax_spec.imshow(
        10 * np.log10(Pxx + 1e-12),
        origin="lower",
        aspect="auto",
        extent=[-BLOCK_SEC, 0, freqs[0], freqs[-1]],
        cmap="inferno"
    )

    ax_spec.set_ylim(0, 500)
    ax_spec.set_xlim(-BLOCK_SEC, 0)
    ax_spec.set_ylabel("Frequency (Hz)")
    ax_spec.set_title("Live Heart Sound Spectrogram")
    
    rectified = np.abs(filt)
    envelope = sosfiltfilt(env_sos, rectified)

    # optional: clip envelope too
    envelope = np.clip(envelope, 0, AMP_LIMIT)

    t = np.linspace(-BLOCK_SEC, 0, BLOCK)
    ax_env.plot(t, envelope, color="cyan", linewidth=1.5)
    ax_env.set_xlim(-BLOCK_SEC, 0)
    ax_env.set_ylim(0, AMP_LIMIT)
    ax_env.set_ylabel("Amplitude")
    ax_env.set_xlabel("Time (s)")
    ax_env.set_title("Amplitude Envelope")


# ------------------ stream ------------------
stream = sd.InputStream(
    samplerate=SR,
    channels=1,
    callback=audio_callback,
    blocksize=256
)

with stream:
    ani = FuncAnimation(fig, update, interval=100)
    plt.tight_layout()
    plt.show()
