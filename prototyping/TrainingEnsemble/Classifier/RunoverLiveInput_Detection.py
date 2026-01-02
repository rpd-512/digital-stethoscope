import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, stft, hilbert, find_peaks
from matplotlib.animation import FuncAnimation
import tensorflow as tf
from statistics import median
from collections import deque

CONF_HISTORY = 10        # number of recent windows
CONF_THRESH  = 0.55     # per-window confidence
MEAN_THRESH  = 0.60     # mean confidence threshold

conf_buffer = deque(maxlen=CONF_HISTORY)

BPM_HISTORY = 10        # number of recent BPMs to average
bpm_buffer = deque(maxlen=BPM_HISTORY)

HB_HISTORY = []


def weighted_bce(pos_weight=1.0, neg_weight=1.23):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        weight = y_true * pos_weight + (1.0 - y_true) * neg_weight
        return tf.reduce_mean(bce * weight)
    return loss


PHASE_LEVELS = {
    "S1": 0.02,
    "Systole": 0.04,
    "S2": 0.6,
    "Diastole": 0.8,
}

model = tf.keras.models.load_model("model_006.keras", custom_objects={"loss": weighted_bce()})

# ------------------ parameters ------------------
SR = 4000
BLOCK_SEC = 3.0

LOW = 30
HIGH = 500
ENV_CUTOFF = 8        # envelope smoothing cutoff (Hz)

NFFT = 512
OVERLAP = 448

AMP_LIMIT = 1.5

BLOCK = int(SR * BLOCK_SEC)

BPM_DISPLAY_POINTS = 60   # ~last 6 seconds if update ≈10 Hz
bpm_time_buffer = deque(maxlen=BPM_DISPLAY_POINTS)

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
fig, (ax_spec, ax_env, ax_bpm) = plt.subplots(
    3, 1,
    figsize=(12, 9),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 1, 0.8]},
    constrained_layout=True
)



# ------------------ update ------------------
def detect_bpm(envelope, sr):
    # Step 0: safe preprocess + bandpass

    # Step 2: Peak detection (tunable)
    min_distance = int(0.15 * sr)   # allow up to 240 BPM
    if len(envelope) < min_distance:
        return None, np.array([], dtype=int), envelope

    # choose a conservative threshold; if signal is very quiet, lower percentile
    height_thresh = np.percentile(envelope, 60)  # try 60th percentile
    prominence_thresh = np.percentile(envelope, 40)

    peaks, props = find_peaks(
        envelope,
        distance=min_distance,
        height=height_thresh,
        prominence=prominence_thresh
    )
    avg_peak_height = np.mean(envelope[peaks]) if len(peaks) > 0 else 0
    
    if len(peaks) > 0:
        avg_peak_height = np.mean(envelope[peaks])
        peaks = peaks[envelope[peaks] > 0.3 * avg_peak_height]


    if len(peaks) < 2:
        return None, peaks

    intervals = np.diff(peaks) / sr  # seconds between beats
    bpm = 60.0 / np.mean(intervals) / 2
    s1 = []
    s2 = []
    #if the next one is closer to the previous one than to the one after, it's s1
    for i in range(1, len(peaks)-1):
        prev_gap = peaks[i] - peaks[i-1]
        next_gap = peaks[i+1] - peaks[i]

        if next_gap < prev_gap:
            s1.append(peaks[i])
        else:
            s2.append(peaks[i])

    return bpm, s1, s2
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
    Pxx_db = 10 * np.log10(Pxx + 1e-12)
    Pxx_norm = (Pxx_db - Pxx_db.min()) / (Pxx_db.max() - Pxx_db.min() + 1e-12)

    rectified = np.abs(hilbert(filt))
    envelope = sosfiltfilt(env_sos, rectified)

    # optional: clip envelope too
    envelope = np.clip(envelope, 0, AMP_LIMIT)

   # -------- prepare model inputs --------
    spec_input = Pxx_norm[np.newaxis, ..., np.newaxis]   # (1, F, T, 1)
    env_input  = envelope[np.newaxis, ..., np.newaxis]  # (1, T, 1)

    pred = model.predict((spec_input, env_input), verbose=0)

    val = float(pred[0][0])
    conf_buffer.append(val)
    mean_conf = sum(conf_buffer) / len(conf_buffer)
    heartbeat = (
        len(conf_buffer) == CONF_HISTORY and
        mean_conf > MEAN_THRESH and
        sum(v > CONF_THRESH for v in conf_buffer) >= 3
    )
    #heartbeat = val > 0.5
    #mean_conf = val
    print(f"Heartbeat probability: {mean_conf:.3f}", end=" → ")


    # -------- plot (manual, specgram-style) --------
    ax_spec.imshow(
        Pxx_norm,
        origin="lower",
        aspect="auto",
        extent=[-BLOCK_SEC, 0, freqs[0], freqs[-1]],
        cmap="inferno"
    )

    ax_spec.set_ylim(0, 500)
    ax_spec.set_xlim(-BLOCK_SEC, 0)
    ax_spec.set_ylabel("Frequency (Hz)")
    ax_spec.set_title("Live Heart Sound Spectrogram")
    
    t = np.linspace(-BLOCK_SEC, 0, BLOCK)
    ax_env.plot(t, envelope, color="cyan", linewidth=1.5)
    ax_env.set_xlim(-BLOCK_SEC, 0)
    ax_env.set_ylim(0, AMP_LIMIT)
    ax_env.set_ylabel("Amplitude")
    ax_env.set_xlabel("Time (s)")
    if(heartbeat):
        print("Heartbeat detected!")
        bpm, s1, s2 = detect_bpm(envelope, SR)
        if bpm is not None:
            bpm_buffer.append(bpm)
            avg_bpm = median(bpm_buffer)
            print(f"Estimated BPM: {avg_bpm:.2f}")
            bpm = avg_bpm

        else:
            print("Not enough heartbeats detected for BPM.")
            bpm = 0.0
        HB_HISTORY.append(bpm)
        ax_env.set_title(f"Amplitude Envelope - Heartbeat Detected, BPM: {bpm:.2f}", color="green")
    else:
        print("No heartbeat detected.")
        ax_env.set_title("Amplitude Envelope - No Heartbeat", color="red")

    if heartbeat and bpm > 0:
        bpm_time_buffer.append(bpm)
    else:
        if len(bpm_time_buffer) > 0:
            bpm_time_buffer.append(bpm_time_buffer[-1])


    ax_bpm.clear()

    t_bpm = np.linspace(
        -len(bpm_time_buffer) * 0.1,  # 0.1 s per frame (interval=100 ms)
        0,
        len(bpm_time_buffer)
    )

    ax_bpm.plot(t_bpm, bpm_time_buffer, color="orange", linewidth=2)

    ax_bpm.set_ylim(50, 200)
    ax_bpm.set_ylabel("BPM")
    ax_bpm.set_xlabel("Time (s)")
    ax_bpm.set_title("Live BPM Tracking")
    ax_bpm.grid(alpha=0.3)


    #plot the peaks if heartbeat and bpm > 0:
    if heartbeat and bpm > 0:
        peak_times_s1 = np.array(s1) / SR - BLOCK_SEC
        peak_times_s2 = np.array(s2) / SR - BLOCK_SEC
        ax_env.scatter(peak_times_s1, envelope[s1], color="red", label="Detected Beats S1")
        ax_env.scatter(peak_times_s2, envelope[s2], color="blue", label="Detected Beats S2")
        #draw 4 horisontal lines, with their edges connecting in each phase, showing s1, systol, s2, diastole, use 
        #ax_env.legend()
    
 
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


plt.figure(figsize=(10, 4))
plt.plot(HB_HISTORY, linestyle='-')
plt.title("Heartbeat BPM History")
plt.xlabel("Detections")
plt.ylabel("BPM")
plt.grid()
plt.show()