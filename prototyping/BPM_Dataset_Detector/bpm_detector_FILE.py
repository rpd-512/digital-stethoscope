import os
import sys
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, find_peaks

# -----------------------
# AUDIO I/O
# -----------------------
def trim_edges(sig, frac=0.1):
    n = len(sig)
    k = int(frac * n)
    if n <= 2 * k:
        return sig
    return sig[k:-k]

def read_audio_file(filename):
    audio, samplerate = sf.read(filename)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio, samplerate

# -----------------------
# SAFER HEART SOUND DSP
# -----------------------

def preproc_signal(sig):
    sig = np.asarray(sig, dtype=np.float64)
    sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
    sig = sig - np.mean(sig)
    peak = np.max(np.abs(sig)) + 1e-12
    return sig / peak

def bandpass_filter_safe(sig, sr, low=20, high=150, order=4):
    sig = preproc_signal(sig)
    min_len = max(3 * (2 * order + 1), 100)
    if len(sig) < min_len:
        return sig

    nyq = 0.5 * sr
    lowc = max(low / nyq, 1e-6)
    highc = min(high / nyq, 0.999999)

    sos = butter(order, [lowc, highc], btype="band", output="sos")
    return sosfiltfilt(sos, sig)

def envelope(sig, sr, cutoff=7):
    rect = np.abs(sig).astype(np.float64)
    nyq = 0.5 * sr
    cutoffc = max(min(cutoff / nyq, 0.999999), 1e-6)

    sos = butter(2, cutoffc, btype="low", output="sos")
    if len(rect) < 50:
        return rect

    env = sosfiltfilt(sos, rect)
    return env - np.min(env)

def detect_bpm(audio, sr):
    filtered = bandpass_filter_safe(audio, sr)
    env = envelope(filtered, sr)

    min_distance = int(0.25 * sr)
    if len(env) < min_distance:
        return None, np.array([], dtype=int), env

    height_thresh = np.percentile(env, 60)
    prominence_thresh = np.percentile(env, 40)

    peaks, _ = find_peaks(
        env,
        distance=min_distance,
        height=height_thresh,
        prominence=prominence_thresh
    )

    if len(peaks) < 2:
        return None, peaks, env

    intervals = np.diff(peaks) / sr
    bpm = 60.0 / np.mean(intervals)
    return bpm, peaks, env

# -----------------------
# PLOTTING
# -----------------------

def plot_results(env, peaks, sr, title):
    t = np.arange(len(env)) / sr
    plt.figure(figsize=(12, 4))
    plt.plot(t, env, label="Envelope")
    if len(peaks) > 0:
        plt.scatter(peaks / sr, env[peaks], color="red", label="Detected Beats")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

# -----------------------
# MAIN DRIVER
# -----------------------

def process_file(filepath):
    audio, sr = read_audio_file(filepath)
    audio = trim_edges(audio, frac=0.1)
    bpm, peaks, env = detect_bpm(audio, sr)

    fname = os.path.basename(filepath)

    if bpm is None:
        print(f"{fname} → BPM: Not detected")
    else:
        print(f"{fname} → BPM: {bpm/2:.2f}")

    plot_results(env, peaks, sr, title=fname)

def collect_wav_files(path):
    if os.path.isfile(path):
        return [path]
    elif os.path.isdir(path):
        return sorted(
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.lower().endswith(".wav")
        )
    else:
        raise ValueError("Input must be a WAV file or directory")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python bpm_detect.py <wav file | directory>")
        sys.exit(1)

    input_path = sys.argv[1]
    wav_files = collect_wav_files(input_path)

    if not wav_files:
        print("No WAV files found.")
        sys.exit(1)

    for wav in wav_files:
        process_file(wav)
