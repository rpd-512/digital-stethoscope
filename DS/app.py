"""
Digital Stethoscope — Flask backend
Streams audio data to a web frontend via Server-Sent Events.
"""

import sys
import os
import json
import threading
import time
import queue
import numpy as np
import webbrowser
from collections import deque
from statistics import median
from flask import Flask, render_template, Response, request, jsonify
from scipy.signal import butter, sosfiltfilt, stft, hilbert, find_peaks

# ── Try importing optional heavy deps ──────────────────────────────────────
try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except Exception:
    AUDIO_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# ── Check all deps and emit install status ─────────────────────────────────
DEPS = {
    "numpy":       True,
    "scipy":       True,
    "flask":       True,
    "sounddevice": AUDIO_AVAILABLE,
    "torch":       TORCH_AVAILABLE,
}

# ── Model ──────────────────────────────────────────────────────────────────
if TORCH_AVAILABLE:
    class CNNEnvFusion(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.spec_net = torch.nn.Sequential(
                torch.nn.Conv2d(1, 16, 5, padding=2), torch.nn.BatchNorm2d(16), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(16, 32, 3, padding=1), torch.nn.BatchNorm2d(32), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.BatchNorm2d(64), torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d((1, 1))
            )
            self.spec_fc = torch.nn.Linear(64, 64)
            self.env_net = torch.nn.Sequential(
                torch.nn.Conv1d(1, 16, 9, padding=4), torch.nn.BatchNorm1d(16), torch.nn.ReLU(),
                torch.nn.Conv1d(16, 32, 7, padding=3), torch.nn.BatchNorm1d(32), torch.nn.ReLU(),
                torch.nn.Conv1d(32, 64, 5, padding=2), torch.nn.BatchNorm1d(64), torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool1d(1)
            )
            self.env_fc = torch.nn.Linear(64, 64)
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Dropout(0.3),
                torch.nn.Linear(64, 32), torch.nn.ReLU(), torch.nn.Dropout(0.2),
                torch.nn.Linear(32, 1)
            )

        def forward(self, spec, env):
            x = self.spec_net(spec)
            x = torch.relu(self.spec_fc(x.view(x.size(0), -1)))
            env = env.permute(0, 2, 1)
            y = self.env_net(env)
            y = torch.relu(self.env_fc(y.view(y.size(0), -1)))
            return torch.sigmoid(self.classifier(torch.cat([x, y], dim=1))).squeeze(1)

    DEVICE = torch.device("cpu")
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_006.pt")
    if os.path.exists(MODEL_PATH):
        model = CNNEnvFusion().to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        MODEL_LOADED = True
    else:
        model = None
        MODEL_LOADED = False
else:
    model = None
    MODEL_LOADED = False

# ── Audio params (mutable via settings) ───────────────────────────────────
class Config:
    SR         = 4000
    BLOCK_SEC  = 3.0
    LOW        = 30
    HIGH       = 500         # ← user-settable upper bound
    ENV_CUTOFF = 8
    AMP_LIMIT  = 1.5         # ← user-settable
    NFFT       = 512
    OVERLAP    = 448
    CONF_HISTORY = 10
    CONF_THRESH  = 0.60
    MEAN_THRESH  = 0.60
    BPM_HISTORY  = 10

cfg = Config()

def make_filters():
    bp  = butter(4, [cfg.LOW, cfg.HIGH], btype="band", fs=cfg.SR, output="sos")
    env = butter(2, cfg.ENV_CUTOFF,      btype="low",  fs=cfg.SR, output="sos")
    return bp, env

bp_sos, env_sos = make_filters()

# ── Shared state ───────────────────────────────────────────────────────────
BLOCK    = int(cfg.SR * cfg.BLOCK_SEC)
buffer   = np.zeros(BLOCK)
buf_lock = threading.Lock()

conf_buffer = deque(maxlen=cfg.CONF_HISTORY)
bpm_buffer  = deque(maxlen=cfg.BPM_HISTORY)
bpm_time_buffer = deque(maxlen=60)
HB_HISTORY = []

data_queue  = queue.Queue(maxsize=5)   # latest processed frames for SSE
audio_active = threading.Event()

# ── Audio callback ─────────────────────────────────────────────────────────
def audio_callback(indata, frames, time_info, status):
    global buffer
    with buf_lock:
        buffer = np.roll(buffer, -len(indata))
        buffer[-len(indata):] = indata[:, 0]

# ── BPM detection ──────────────────────────────────────────────────────────
def detect_bpm(envelope, sr):
    min_distance = int(0.15 * sr)
    if len(envelope) < min_distance:
        return None, [], []
    h   = np.percentile(envelope, 60)
    pro = np.percentile(envelope, 40)
    peaks, _ = find_peaks(envelope, distance=min_distance, height=h, prominence=pro)
    if len(peaks) < 2:
        return None, [], []
    intervals = np.diff(peaks) / sr
    bpm = 60.0 / np.mean(intervals) / 2
    s1, s2 = [], []
    for i in range(1, len(peaks) - 1):
        if (peaks[i+1] - peaks[i]) < (peaks[i] - peaks[i-1]):
            s1.append(int(peaks[i]))
        else:
            s2.append(int(peaks[i]))
    return float(bpm), s1, s2

# ── Processing thread ──────────────────────────────────────────────────────
def processing_loop():
    while True:
        time.sleep(0.1)
        if not audio_active.is_set():
            continue
        with buf_lock:
            raw = buffer.copy()

        try:
            filt     = sosfiltfilt(bp_sos, raw)
            rect     = np.abs(hilbert(filt))
            envelope = sosfiltfilt(env_sos, rect)
            envelope = np.clip(envelope, 0, cfg.AMP_LIMIT)

            freqs, bins, Zxx = stft(
                filt, fs=cfg.SR, nperseg=cfg.NFFT, noverlap=cfg.OVERLAP,
                window="hann", boundary=None, padded=False
            )
            Pxx    = np.abs(Zxx) ** 2
            Pdb    = 10 * np.log10(Pxx + 1e-12)
            Pnorm  = (Pdb - Pdb.min()) / (np.ptp(Pdb) + 1e-12)

            prob = 0.5
            if MODEL_LOADED and TORCH_AVAILABLE:
                spec_t = torch.tensor(Pnorm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                env_t  = torch.tensor(envelope, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
                with torch.no_grad():
                    prob = model(spec_t, env_t).item()

            conf_buffer.append(prob)
            mean_conf = float(np.mean(conf_buffer))
            heartbeat = (
                len(conf_buffer) == cfg.CONF_HISTORY and
                mean_conf > cfg.MEAN_THRESH and
                sum(v > cfg.CONF_THRESH for v in conf_buffer) >= 3
            )

            bpm_val = None
            s1_times, s2_times = [], []
            if heartbeat:
                bpm_raw, s1, s2 = detect_bpm(envelope, cfg.SR)
                if bpm_raw:
                    bpm_buffer.append(bpm_raw)
                    bpm_val = float(median(bpm_buffer))
                    HB_HISTORY.append(bpm_val)
                    bpm_time_buffer.append(bpm_val)
                    s1_times = [float(i) / cfg.SR - cfg.BLOCK_SEC for i in s1]
                    s2_times = [float(i) / cfg.SR - cfg.BLOCK_SEC for i in s2]
            else:
                if bpm_time_buffer:
                    bpm_time_buffer.append(bpm_time_buffer[-1])

            # Downsample envelope for transfer (max 512 pts)
            step = max(1, len(envelope) // 512)
            env_ds = envelope[::step].tolist()

            # Spectrogram: send only freq rows up to HIGH, downsample cols
            freq_mask = freqs <= cfg.HIGH
            spec_slice = Pnorm[freq_mask, :]
            # Downsample cols to 128
            col_step = max(1, spec_slice.shape[1] // 128)
            spec_slice = spec_slice[:, ::col_step]
            # Flatten row-major
            spec_flat = spec_slice.flatten().tolist()
            spec_rows = int(spec_slice.shape[0])
            spec_cols = int(spec_slice.shape[1])

            bpm_hist = list(bpm_time_buffer)

            frame = {
                "heartbeat":  heartbeat,
                "confidence": round(float(prob), 3),
                "mean_conf":  round(mean_conf, 3),
                "bpm":        round(bpm_val, 1) if bpm_val else None,
                "s1_times":   s1_times,
                "s2_times":   s2_times,
                "envelope":   env_ds,
                "amp_limit":  cfg.AMP_LIMIT,
                "spec_flat":  spec_flat,
                "spec_rows":  spec_rows,
                "spec_cols":  spec_cols,
                "bpm_hist":   bpm_hist,
                "block_sec":  cfg.BLOCK_SEC,
            }

            if data_queue.full():
                try:
                    data_queue.get_nowait()
                except queue.Empty:
                    pass
            data_queue.put(frame)

        except Exception as e:
            pass  # keep running

# ── Flask app ──────────────────────────────────────────────────────────────
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html", deps=DEPS, model_loaded=MODEL_LOADED)

@app.route("/stream")
def stream():
    def event_gen():
        while True:
            try:
                frame = data_queue.get(timeout=1.0)
                yield f"data: {json.dumps(frame)}\n\n"
            except queue.Empty:
                yield "data: {}\n\n"
    return Response(event_gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.route("/start", methods=["POST"])
def start_audio():
    global bp_sos, env_sos, BLOCK, buffer
    if not AUDIO_AVAILABLE:
        return jsonify({"ok": False, "error": "sounddevice not installed"})
    try:
        BLOCK  = int(cfg.SR * cfg.BLOCK_SEC)
        buffer = np.zeros(BLOCK)
        bp_sos, env_sos = make_filters()
        audio_active.set()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.route("/stop", methods=["POST"])
def stop_audio():
    audio_active.clear()
    return jsonify({"ok": True})

@app.route("/settings", methods=["POST"])
def update_settings():
    global bp_sos, env_sos
    data = request.get_json()
    if "high" in data:
        val = float(data["high"])
        if 100 <= val <= 1000:
            cfg.HIGH = val
    if "amp_limit" in data:
        val = float(data["amp_limit"])
        if 0.1 <= val <= 10.0:
            cfg.AMP_LIMIT = val
    if "low" in data:
        val = float(data["low"])
        if 10 <= val <= 100:
            cfg.LOW = val
    bp_sos, env_sos = make_filters()
    return jsonify({"ok": True, "high": cfg.HIGH, "amp_limit": cfg.AMP_LIMIT, "low": cfg.LOW})

@app.route("/devices")
def get_devices():
    if not AUDIO_AVAILABLE:
        return jsonify({"devices": []})
    devs = sd.query_devices()
    result = []
    for i, d in enumerate(devs):
        if d["max_input_channels"] > 0:
            result.append({"id": i, "name": d["name"], "sr": d["default_samplerate"]})
    return jsonify({"devices": result})

@app.route("/history")
def get_history():
    return jsonify({"history": HB_HISTORY})

@app.route("/status")
def status():
    return jsonify({
        "deps":        DEPS,
        "model":       MODEL_LOADED,
        "audio":       AUDIO_AVAILABLE,
        "active":      audio_active.is_set(),
        "config": {
            "low":       cfg.LOW,
            "high":      cfg.HIGH,
            "amp_limit": cfg.AMP_LIMIT,
            "sr":        cfg.SR,
        }
    })

# ── Audio stream object ────────────────────────────────────────────────────
def run_audio():
    """Runs in a thread, manages sounddevice stream lifecycle."""
    stream = None
    while True:
        if audio_active.is_set():
            if stream is None or not stream.active:
                try:
                    stream = sd.InputStream(
                        samplerate=cfg.SR,
                        channels=1,
                        callback=audio_callback,
                        blocksize=256
                    )
                    stream.start()
                except Exception as e:
                    audio_active.clear()
            time.sleep(0.05)
        else:
            if stream and stream.active:
                stream.stop()
                stream.close()
                stream = None
            time.sleep(0.1)

# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t_proc  = threading.Thread(target=processing_loop, daemon=True)
    t_audio = threading.Thread(target=run_audio, daemon=True)
    t_proc.start()
    t_audio.start()

    port = 5757
    # open browser after short delay
    def open_browser():
        time.sleep(1.2)
        webbrowser.open(f"http://127.0.0.1:{port}")
    threading.Thread(target=open_browser, daemon=True).start()

    print(f"\n  🩺  Digital Stethoscope running at http://127.0.0.1:{port}\n")
    app.use_reloader = False
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
