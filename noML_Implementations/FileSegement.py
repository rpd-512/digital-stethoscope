import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, find_peaks, filtfilt
from sys import argv

def read_audio_file(filename):
    audio, samplerate = sf.read(filename)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio, samplerate

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

def bandpass_filter_safe(sig, sr, low=20, high=150, order=4):
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
    filtered = bandpass_filter_safe(audio, sr)
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


def classify_s1_s2(peaks):
    s1_list = []
    s2_list = []
    for p in range(1, len(peaks)-1):
        prev_int = peaks[p]   - peaks[p-1]
        next_int = peaks[p+1] - peaks[p]
        if next_int < prev_int:
            s1_list.append(peaks[p])
        else:
            s2_list.append(peaks[p])
    return s1_list, s2_list

# -----------------------
# PLOTTING
# -----------------------

def plot_results(audio, env, peaks, sr, s1_list=None, s2_list=None):
    t = np.arange(len(env)) / sr

    plt.figure(figsize=(14, 5))

    # Plot the envelope
    plt.plot(t, env, color="darkgreen", linewidth=1.2, label="Envelope")

    # Plot all detected peaks
    if len(peaks) > 0:
        plt.scatter(peaks/sr,
                    env[peaks],
                    color="red",
                    s=40,
                    label="Detected Peaks")

    # Plot S1
    if s1_list is not None and len(s1_list) > 0:
        plt.scatter(np.array(s1_list)/sr,
                    env[s1_list],
                    color="blue",
                    s=70,
                    marker="o",
                    label="S1")

    # Plot S2
    if s2_list is not None and len(s2_list) > 0:
        plt.scatter(np.array(s2_list)/sr,
                    env[s2_list],
                    color="orange",
                    s=70,
                    marker="o",
                    label="S2")

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Heart Sound Envelope with S1 / S2 Markers")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_lub_dub(audio, peaks, sr, s1_list, s2_list):
    t = np.arange(len(audio)) / sr

    plt.figure(figsize=(14, 5))

    # Plot the raw waveform
    plt.plot(t, audio, color="navy", linewidth=0.8, label="PCG")

    # Mark S1
    if len(s1_list) > 0:
        plt.scatter(np.array(s1_list)/sr,
                    audio[s1_list],
                    color="blue",
                    s=50,
                    label="S1")

    # Mark S2
    if len(s2_list) > 0:
        plt.scatter(np.array(s2_list)/sr,
                    audio[s2_list],
                    color="orange",
                    s=50,
                    label="S2")

    # Detected peaks (optional)
    if len(peaks) > 0:
        plt.scatter(peaks/sr,
                    audio[peaks],
                    color="red",
                    s=20,
                    alpha=0.4,
                    label="All detected peaks")

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("S1 / S2 Heart Sound Visualizer")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    try:
        filename = argv[1]
    except IndexError:
        print("Usage: python FileBPM.py <audio_filename>")
        exit(1)
    audio, sr = read_audio_file(filename)

    bpm, peaks, env = detect_bpm(audio, sr)
    s1_list, s2_list = classify_s1_s2(peaks)
    if bpm is None:
        print("Not enough heartbeats detected.")
    else:
        print(f"Detected BPM: {bpm/2:.2f}")

    plot_lub_dub(audio, peaks, sr, s1_list, s2_list)
    plot_results(audio, env, peaks, sr, s1_list, s2_list)
