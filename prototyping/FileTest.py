import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

def read_audio_file(filename):
    audio, samplerate = sf.read(filename)
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    print(f"Loaded '{filename}'")
    print(f"Sample rate: {samplerate} Hz")
    print(f"Total samples: {len(audio)}")
    return audio, samplerate

def plot_audio(audio, samplerate):
    t = np.linspace(0, len(audio) / samplerate, num=len(audio))
    plt.plot(t, audio)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Audio Waveform")
    plt.show()

if __name__ == "__main__":
    filename = "input.wav"   # change this
    audio, sr = read_audio_file(filename)
    plot_audio(audio, sr)
