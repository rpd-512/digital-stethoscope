import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt

def record_audio(duration=10, samplerate=44100):
    print("Recording...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    print("Done.")
    return audio.flatten()

def plot_audio(audio, samplerate):
    t = np.linspace(0, len(audio) / samplerate, num=len(audio))
    plt.plot(t, audio)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Audio Waveform")
    plt.show()

if __name__ == "__main__":
    sr = 1024
    duration = 3
    data = record_audio(duration, sr)
    plot_audio(data, sr)
