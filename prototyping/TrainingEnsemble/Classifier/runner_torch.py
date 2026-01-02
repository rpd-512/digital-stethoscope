import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, stft, hilbert, find_peaks
from matplotlib.animation import FuncAnimation
import torch
import torch.nn as nn
from statistics import median
from collections import deque

# ===================== MODEL =====================

class CNNEnvFusion(nn.Module):
    def __init__(self):
        super().__init__()

        self.spec_net = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.spec_fc = nn.Linear(64, 64)

        self.env_net = nn.Sequential(
            nn.Conv1d(1, 16, 9, padding=4),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Conv1d(16, 32, 7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )
        self.env_fc = nn.Linear(64, 64)

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, spec, env):
        x = self.spec_net(spec)
        x = torch.relu(self.spec_fc(x.view(x.size(0), -1)))

        env = env.permute(0, 2, 1)
        y = self.env_net(env)
        y = torch.relu(self.env_fc(y.view(y.size(0), -1)))

        return torch.sigmoid(self.classifier(torch.cat([x, y], dim=1))).squeeze(1)


DEVICE = torch.device("cpu")
model = CNNEnvFusion().to(DEVICE)
model.load_state_dict(torch.load("model_006.pt", map_location=DEVICE))
model.eval()

# ===================== CONF / BPM =====================

CONF_HISTORY = 10
CONF_THRESH  = 0.60
MEAN_THRESH  = 0.60

conf_buffer = deque(maxlen=CONF_HISTORY)

BPM_HISTORY = 10
bpm_buffer = deque(maxlen=BPM_HISTORY)

HB_HISTORY = []

# ===================== AUDIO PARAMS =====================

SR = 4000
BLOCK_SEC = 3.0
BLOCK = int(SR * BLOCK_SEC)

LOW = 30
HIGH = 500
ENV_CUTOFF = 8

NFFT = 512
OVERLAP = 448
AMP_LIMIT = 1.5

BPM_DISPLAY_POINTS = 60
bpm_time_buffer = deque(maxlen=BPM_DISPLAY_POINTS)

bp_sos = butter(4, [LOW, HIGH], btype="band", fs=SR, output="sos")
env_sos = butter(2, ENV_CUTOFF, btype="low", fs=SR, output="sos")

buffer = np.zeros(BLOCK)

def audio_callback(indata, frames, time, status):
    global buffer
    buffer = np.roll(buffer, -len(indata))
    buffer[-len(indata):] = indata[:, 0]

# ===================== BPM DETECTION =====================

def detect_bpm(envelope, sr):
    min_distance = int(0.15 * sr)
    if len(envelope) < min_distance:
        return None, [], []

    height = np.percentile(envelope, 60)
    prominence = np.percentile(envelope, 40)

    peaks, _ = find_peaks(
        envelope,
        distance=min_distance,
        height=height,
        prominence=prominence
    )

    if len(peaks) < 2:
        return None, [], []

    intervals = np.diff(peaks) / sr
    bpm = 60.0 / np.mean(intervals) / 2

    s1, s2 = [], []
    for i in range(1, len(peaks) - 1):
        if (peaks[i+1] - peaks[i]) < (peaks[i] - peaks[i-1]):
            s1.append(peaks[i])
        else:
            s2.append(peaks[i])

    return bpm, s1, s2

# ===================== PLOT =====================

fig, (ax_spec, ax_env, ax_bpm) = plt.subplots(
    3, 1, figsize=(12, 9),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 1, 0.8]},
    constrained_layout=True
)

# ===================== UPDATE =====================

def update(_):
    ax_spec.clear()
    ax_env.clear()

    filt = sosfiltfilt(bp_sos, buffer)
    rect = np.abs(hilbert(filt))
    envelope = sosfiltfilt(env_sos, rect)
    envelope = np.clip(envelope, 0, AMP_LIMIT)

    freqs, bins, Zxx = stft(
        filt, fs=SR, nperseg=NFFT, noverlap=OVERLAP,
        window="hann", boundary=None, padded=False
    )

    Pxx = np.abs(Zxx) ** 2
    Pxx_db = 10 * np.log10(Pxx + 1e-12)
    Pxx_norm = (Pxx_db - Pxx_db.min()) / (np.ptp(Pxx_db) + 1e-12)

    spec_t = torch.tensor(Pxx_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    env_t  = torch.tensor(envelope, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

    with torch.no_grad():
        prob = model(spec_t, env_t).item()

    conf_buffer.append(prob)
    mean_conf = np.mean(conf_buffer)

    heartbeat = (
        len(conf_buffer) == CONF_HISTORY and
        mean_conf > MEAN_THRESH and
        sum(v > CONF_THRESH for v in conf_buffer) >= 3
    )

    ax_spec.imshow(
        Pxx_norm, origin="lower", aspect="auto",
        extent=[-BLOCK_SEC, 0, freqs[0], freqs[-1]],
        cmap="inferno"
    )
    ax_spec.set_ylim(0, 500)
    ax_spec.set_title("Live Heart Sound Spectrogram")

    t = np.linspace(-BLOCK_SEC, 0, BLOCK)
    ax_env.plot(t, envelope, color="cyan", lw=1.5)
    ax_env.set_ylim(0, AMP_LIMIT)

    if heartbeat:
        bpm, s1, s2 = detect_bpm(envelope, SR)
        if bpm:
            bpm_buffer.append(bpm)
            bpm = median(bpm_buffer)
            HB_HISTORY.append(bpm)
            bpm_time_buffer.append(bpm)

            ax_env.scatter(np.array(s1)/SR-BLOCK_SEC, envelope[s1], c="red")
            ax_env.scatter(np.array(s2)/SR-BLOCK_SEC, envelope[s2], c="blue")
            ax_env.set_title(f"Heartbeat Detected | BPM: {bpm:.1f}", color="green")
        else:
            ax_env.set_title("Heartbeat Detected (BPM unstable)", color="orange")
    else:
        ax_env.set_title("No Heartbeat", color="red")
        if bpm_time_buffer:
            bpm_time_buffer.append(bpm_time_buffer[-1])

    ax_bpm.clear()
    ax_bpm.plot(np.linspace(-len(bpm_time_buffer)*0.1, 0, len(bpm_time_buffer)),
                bpm_time_buffer, color="orange")
    ax_bpm.set_ylim(50, 200)
    ax_bpm.set_title("Live BPM Tracking")
    ax_bpm.grid(alpha=0.3)

# ===================== STREAM =====================

stream = sd.InputStream(
    samplerate=SR,
    channels=1,
    callback=audio_callback,
    blocksize=256
)

with stream:
    ani = FuncAnimation(fig, update, interval=100)
    plt.show()

# ===================== HISTORY =====================

plt.figure(figsize=(10, 4))
plt.plot(HB_HISTORY)
plt.title("Heartbeat BPM History")
plt.xlabel("Detections")
plt.ylabel("BPM")
plt.grid()
plt.show()
