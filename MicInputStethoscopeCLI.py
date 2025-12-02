import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, find_peaks, filtfilt
import numpy as np
import sounddevice as sd
import threading
import time


BUFFER_SECONDS = 10
SAMPLERATE = 44100
CHANNELS = 1

buffer_size = BUFFER_SECONDS * SAMPLERATE
audio_buffer = np.zeros(buffer_size, dtype=np.float64)
write_pos = 0
lock = threading.Lock()

def audio_callback(indata, frames, time, status):
    global audio_buffer, write_pos
    if status:
        print(status)
        
    samples = indata[:, 0].astype(np.float64)

    with lock:
        end_pos = write_pos + frames

        if end_pos < buffer_size:
            audio_buffer[write_pos:end_pos] = samples
        else:
            # wrap-around write
            first = buffer_size - write_pos
            audio_buffer[write_pos:] = samples[:first]
            audio_buffer[:frames-first] = samples[first:]

        write_pos = (write_pos + frames) % buffer_size


def preproc_signal(sig):
    # ensure float64 for numerical stability in filtfilt/sosfiltfilt
    sig = np.asarray(sig, dtype=np.float64)

    # replace any tiny NaNs/Infs with zeros
    sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)

    # remove DC offset
    sig = sig - np.mean(sig)

    # normalize amplitude (avoid dividing by zero)
    peak = np.max(np.abs(sig)) + 1e-12
    sig = sig / peak

    return sig

def bandpass_filter(sig, sr, low=20, high=150, order=4):
    """
    Uses SOS + sosfiltfilt for numerical stability.
    Returns filtered signal (float64). If signal is too short, returns the original signal.
    """
    sig = preproc_signal(sig)

    # if signal too short relative to filter order, skip filtering
    # padlen for sosfiltfilt is handled internally but very short signals still fail
    min_len = max(3 * (2 * order + 1), 100)  # heuristic min length (100 samples fallback)
    if len(sig) < min_len:
        # signal too short for reliable filtfilt; return preprocessed signal
        return sig

    nyq = 0.5 * sr
    lowc = max(low / nyq, 1e-6)
    highc = min(high / nyq, 0.999999)

    # use sos for better numerical behaviour
    sos = butter(order, [lowc, highc], btype='band', output='sos')
    filtered = sosfiltfilt(sos, sig)
    return filtered

def envelope(sig, sr, cutoff=7):
    """
    Rectify + lowpass envelope using SOS filtfilt.
    """
    rect = np.abs(sig)
    rect = rect.astype(np.float64)

    # lowpass
    nyq = 0.5 * sr
    cutoffc = max(min(cutoff / nyq, 0.999999), 1e-6)
    sos = butter(2, cutoffc, btype='low', output='sos')

    # guard for very short signals
    if len(rect) < 50:
        return rect

    env = sosfiltfilt(sos, rect)
    # small smoothing / baseline removal (optional)
    env = env - np.min(env)
    return env

def detect_bpm(audio, sr):
    # Step 0: safe preprocess + bandpass
    filtered = bandpass_filter(audio, sr)
    if np.isnan(filtered).any() or np.isinf(filtered).any():
        raise RuntimeError("Filtering still produced NaN/Inf — aborting (unexpected).")

    # Step 1: Envelope
    env = envelope(filtered, sr)

    # Step 2: Peak detection (tunable)
    min_distance = int(0.25 * sr)   # allow up to 240 BPM
    if len(env) < min_distance:
        return None, np.array([], dtype=int), env

    # choose a conservative threshold; if signal is very quiet, lower percentile
    height_thresh = np.percentile(env, 60)  # try 60th percentile
    prominence_thresh = np.percentile(env, 40)

    peaks, props = find_peaks(
        env,
        distance=min_distance,
        height=height_thresh,
        prominence=prominence_thresh
    )

    if len(peaks) < 2:
        return None, peaks, env

    intervals = np.diff(peaks) / sr  # seconds between beats
    bpm = 60.0 / np.mean(intervals)
    return bpm, peaks, env

def get_last_audio():
    with lock:
        if write_pos == 0:
            return audio_buffer.copy()
        
        return np.concatenate((audio_buffer[write_pos:], audio_buffer[:write_pos]))


stream = sd.InputStream(
    samplerate=SAMPLERATE,
    channels=CHANNELS,
    callback=audio_callback,
    blocksize=1024
)
stream.start()

bpm_history = []

segment_history = []

while True:
    try:
        segment = get_last_audio()
        segment_history.append(segment)
        bpm, peaks, env = detect_bpm(segment, SAMPLERATE)
        if bpm is None:
            print("BPM: ---")
        else:
            print(f"BPM: {bpm/2:.2f}")
            bpm_history.append(bpm/2)
        time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")
        break

#plot bpm history
if bpm_history:
    plt.plot(bpm_history)
    plt.xlabel("Time (s)")
    plt.ylabel("BPM")
    plt.title("Detected BPM Over Time")
    plt.show()
#plot last segment with peaks
if segment_history:
    last_segment = segment_history[-1]
    filtered = bandpass_filter(last_segment, SAMPLERATE)
    env = envelope(filtered, SAMPLERATE)
    bpm, peaks, env = detect_bpm(last_segment, SAMPLERATE)

    t = np.linspace(0, len(last_segment) / SAMPLERATE, num=len(last_segment))

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, filtered, label='Filtered Signal')
    plt.scatter(t[peaks], filtered[peaks], color='red', label='Detected Peaks')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Filtered Audio Signal with Detected Peaks")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(t, env, color='orange', label='Envelope')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Envelope of Filtered Signal")
    plt.legend()

    plt.tight_layout()
    plt.show()