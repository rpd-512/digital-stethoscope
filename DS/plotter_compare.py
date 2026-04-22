import json
import sys
import matplotlib.pyplot as plt

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def extract_data(data):
    # Amplitude
    amp_t = [p["t"] for p in data["amplitude_timeline"]]
    amp = [p["amplitude"] for p in data["amplitude_timeline"]]

    # BPM + confidence
    bpm_t = [p["t"] for p in data["bpm_timeline"]]
    bpm = [p["bpm"] for p in data["bpm_timeline"]]

    confidence = [
        (p["env_stability"] + p["spec_stability"]) / 2
        for p in data["bpm_timeline"]
    ]

    return amp_t, amp, bpm_t, bpm, confidence


def plot_file(data, title="Session"):
    amp_t, amp, bpm_t, bpm, conf = extract_data(data)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(title)

    # BPM
    axs[0].plot(bpm_t, bpm)
    axs[0].set_title("BPM")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("BPM")

    # Amplitude
    axs[1].plot(amp_t, amp)
    axs[1].set_title("Amplitude")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Amplitude")

    # Confidence
    axs[2].plot(bpm_t, conf)
    axs[2].set_title("Confidence")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Confidence")

    plt.tight_layout()
    plt.show()


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 plot.py file1.json file2.json ...")
        return

    for path in sys.argv[1:]:
        data = load_json(path)
        title = f"{data['patient']['name']} | Mean BPM: {data['session']['mean_bpm']}"
        plot_file(data, title)


if __name__ == "__main__":
    main()