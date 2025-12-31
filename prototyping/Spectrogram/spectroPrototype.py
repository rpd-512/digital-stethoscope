import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
from matplotlib.animation import FuncAnimation

# ------------------ parameters ------------------
SR = 4000
BLOCK_SEC = 3.0

LOW = 30
HIGH = 120

NFFT = 512
OVERLAP = 448

BLOCK = int(SR * BLOCK_SEC)


# ------------------ filter ------------------
sos = butter(4, [LOW, HIGH], btype='band', fs=SR, output='sos')

# ------------------ buffer ------------------
buffer = np.zeros(BLOCK)

def audio_callback(indata, frames, time, status):
    global buffer
    sig = indata[:, 0]
    buffer = np.roll(buffer, -len(sig))
    buffer[-len(sig):] = sig

# ------------------ plot ------------------
fig, ax = plt.subplots()

spec = ax.specgram(
    buffer,
    NFFT=NFFT,
    Fs=SR,
    noverlap=OVERLAP,
    cmap="inferno"
)[3]

ax.set_ylim(0, 200)
ax.set_xlabel("Time")
ax.set_ylabel("Frequency (Hz)")
ax.set_title("Live Heart Sound Spectrogram")

# ------------------ update ------------------
def update(frame):
    global spec
    ax.clear()
    filt = sosfiltfilt(sos, buffer)
    ax.specgram(
        filt,
        NFFT=NFFT,
        Fs=SR,
        noverlap=OVERLAP,
        cmap="inferno"
    )
    ax.set_ylim(0, 200)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Live Heart Sound Spectrogram")

# ------------------ stream ------------------
stream = sd.InputStream(
    samplerate=SR,
    channels=1,
    callback=audio_callback,
    blocksize=256
)

with stream:
    ani = FuncAnimation(fig, update, interval=100)
    plt.show()
