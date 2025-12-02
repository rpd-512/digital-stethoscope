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

# -----------------------
# SAFER HEART SOUND DSP
# -----------------------

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
    print(f"Envelope cutoff (normalized): {cutoffc:.6f}")
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

# -----------------------
# PLOTTING
# -----------------------

def plot_results(audio, env, peaks, sr):
    t = np.arange(len(audio)) / sr
    plt.figure(figsize=(12, 5))
    plt.plot(t, env, label="Envelope")
    if len(peaks) > 0:
        plt.scatter(peaks / sr, env[peaks], color="red", label="Detected Beats")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Heart Sound Envelope + Detected Beats")
    plt.legend()
    plt.show()

def plot_audio(audio, samplerate):
    t = np.linspace(0, len(audio) / samplerate, num=len(audio))
    plt.plot(t, audio)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Audio Waveform")
    plt.show()

if __name__ == "__main__":
    try:
        filename = argv[1]
    except IndexError:
        print("Usage: python FileBPM.py <audio_filename>")
        exit(1)
    audio, sr = read_audio_file(filename)

    #plot_audio(audio, sr)

    bpm, peaks, env = detect_bpm(audio, sr)
    if bpm is None:
        print("Not enough heartbeats detected.")
    else:
        print(f"Detected BPM: {bpm/2:.2f}")

    plot_results(audio, env, peaks, sr)
