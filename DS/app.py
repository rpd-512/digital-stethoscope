"""
CardioScope v3.0 — Digital Stethoscope Backend
New features:
  - Particle Swarm Optimisation (PSO) for auto-tuning bandpass params
  - Session recording with audio, spectrogram PNG, envelope PNG, JSON metadata
  - Patient history browser
"""

import sys, os, json, threading, time, queue, uuid, shutil, re
import numpy as np
import webbrowser
from datetime import datetime
from collections import deque
from statistics import median
from pathlib import Path
from flask import Flask, render_template, Response, request, jsonify, send_from_directory
from scipy.signal import butter, sosfiltfilt, stft, hilbert, find_peaks

# ── Optional heavy deps ────────────────────────────────────────────────────
try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except Exception:
    AUDIO_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except Exception:
    MPL_AVAILABLE = False

try:
    import soundfile as sf
    SF_AVAILABLE = True
except Exception:
    SF_AVAILABLE = False

DEPS = {
    "numpy":       True,
    "scipy":       True,
    "flask":       True,
    "sounddevice": AUDIO_AVAILABLE,
    "torch":       TORCH_AVAILABLE,
    "matplotlib":  MPL_AVAILABLE,
    "soundfile":   SF_AVAILABLE,
}

# ── Model ──────────────────────────────────────────────────────────────────
if TORCH_AVAILABLE:
    class CNNEnvFusion(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.spec_net = torch.nn.Sequential(
                torch.nn.Conv2d(1,16,5,padding=2),torch.nn.BatchNorm2d(16),torch.nn.ReLU(),torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(16,32,3,padding=1),torch.nn.BatchNorm2d(32),torch.nn.ReLU(),torch.nn.MaxPool2d(2),
                torch.nn.Conv2d(32,64,3,padding=1),torch.nn.BatchNorm2d(64),torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d((1,1)))
            self.spec_fc = torch.nn.Linear(64,64)
            self.env_net = torch.nn.Sequential(
                torch.nn.Conv1d(1,16,9,padding=4),torch.nn.BatchNorm1d(16),torch.nn.ReLU(),
                torch.nn.Conv1d(16,32,7,padding=3),torch.nn.BatchNorm1d(32),torch.nn.ReLU(),
                torch.nn.Conv1d(32,64,5,padding=2),torch.nn.BatchNorm1d(64),torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool1d(1))
            self.env_fc = torch.nn.Linear(64,64)
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(128,64),torch.nn.ReLU(),torch.nn.Dropout(0.3),
                torch.nn.Linear(64,32),torch.nn.ReLU(),torch.nn.Dropout(0.2),
                torch.nn.Linear(32,1))

        def forward(self, spec, env):
            x = self.spec_net(spec)
            x = torch.relu(self.spec_fc(x.view(x.size(0),-1)))
            env = env.permute(0,2,1)
            y = self.env_net(env)
            y = torch.relu(self.env_fc(y.view(y.size(0),-1)))
            return torch.sigmoid(self.classifier(torch.cat([x,y],dim=1))).squeeze(1)

    DEVICE = torch.device("cpu")
    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_006.pt")
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

# ── Config ─────────────────────────────────────────────────────────────────
class Config:
    SR           = 4000
    BLOCK_SEC    = 3.0
    LOW          = 30
    HIGH         = 500
    ENV_CUTOFF   = 8
    AMP_LIMIT    = 1.5
    NFFT         = 512
    OVERLAP      = 448
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

# ── Shared audio state ──────────────────────────────────────────────────────
BLOCK    = int(cfg.SR * cfg.BLOCK_SEC)
buffer   = np.zeros(BLOCK)
buf_lock = threading.Lock()

conf_buffer     = deque(maxlen=cfg.CONF_HISTORY)
bpm_buffer      = deque(maxlen=cfg.BPM_HISTORY)
bpm_time_buffer = deque(maxlen=60)
HB_HISTORY      = []

data_queue   = queue.Queue(maxsize=5)
audio_active = threading.Event()

# ── Recordings dir ─────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
RECORDINGS_DIR = os.path.join(BASE_DIR, "recordings")
os.makedirs(RECORDINGS_DIR, exist_ok=True)

# ── Recording state ─────────────────────────────────────────────────────────
rec_lock      = threading.Lock()
is_recording  = False
rec_audio_buf = []
rec_env_buf   = []
rec_bpm_buf   = []
rec_conf_buf  = []
rec_spec_buf  = []
rec_start_ts  = None
rec_id        = None

# ── PSO state ──────────────────────────────────────────────────────────────
pso_lock       = threading.Lock()
pso_active     = False
pso_stop_event = threading.Event()
pso_status     = {
    "running": False, "iteration": 0, "best_fitness": 0.0,
    "best_low": cfg.LOW, "best_high": cfg.HIGH, "best_amp": cfg.AMP_LIMIT,
    "log": []
}

# ═══════════════════════════════════════════════════════════════════════════
#  AUDIO CALLBACK
# ═══════════════════════════════════════════════════════════════════════════
def audio_callback(indata, frames, time_info, status):
    global buffer
    chunk = indata[:, 0].copy()
    with buf_lock:
        buffer = np.roll(buffer, -len(chunk))
        buffer[-len(chunk):] = chunk
    if is_recording:
        with rec_lock:
            rec_audio_buf.append(chunk)

# ═══════════════════════════════════════════════════════════════════════════
#  BPM DETECTION
# ═══════════════════════════════════════════════════════════════════════════
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
    for i in range(1, len(peaks)-1):
        if (peaks[i+1]-peaks[i]) < (peaks[i]-peaks[i-1]):
            s1.append(int(peaks[i]))
        else:
            s2.append(int(peaks[i]))
    return float(bpm), s1, s2

# ═══════════════════════════════════════════════════════════════════════════
#  DSP FITNESS FUNCTION  (used by PSO)
# ═══════════════════════════════════════════════════════════════════════════
def evaluate_params(low, high, amp_limit, raw_snapshot):
    try:
        if low >= high or high > cfg.SR/2 - 1 or low < 5:
            return 0.0
        bp    = butter(4, [low, high], btype="band", fs=cfg.SR, output="sos")
        env_f = butter(2, cfg.ENV_CUTOFF, btype="low", fs=cfg.SR, output="sos")
        filt     = sosfiltfilt(bp, raw_snapshot)
        rect     = np.abs(hilbert(filt))
        envelope = sosfiltfilt(env_f, rect)
        envelope = np.clip(envelope, 0, amp_limit)

        # Component 1 — model confidence
        model_score = 0.5
        if MODEL_LOADED and TORCH_AVAILABLE:
            _, _, Zxx = stft(filt, fs=cfg.SR, nperseg=cfg.NFFT,
                              noverlap=cfg.OVERLAP, window="hann",
                              boundary=None, padded=False)
            Pdb   = 10 * np.log10(np.abs(Zxx)**2 + 1e-12)
            Pnorm = (Pdb - Pdb.min()) / (np.ptp(Pdb) + 1e-12)
            spec_t = torch.tensor(Pnorm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            env_t  = torch.tensor(envelope, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            with torch.no_grad():
                model_score = model(spec_t, env_t).item()

        # Component 2 — rhythmic regularity
        rhythmic_score = 0.0
        bpm_val, s1, s2 = detect_bpm(envelope, cfg.SR)
        if bpm_val and 40 < bpm_val < 180:
            peaks_all = sorted(s1 + s2)
            if len(peaks_all) >= 3:
                ivs = np.diff(peaks_all) / cfg.SR
                cv  = np.std(ivs) / (np.mean(ivs) + 1e-6)
                rhythmic_score = max(0, 1.0 - cv) * 0.8 + 0.2
            else:
                rhythmic_score = 0.3

        # Component 3 — SNR proxy
        snr_score = float(np.clip(np.var(filt) / (np.var(raw_snapshot) + 1e-12) * 5, 0, 1))

        return float(0.50 * model_score + 0.35 * rhythmic_score + 0.15 * snr_score)
    except Exception:
        return 0.0

# ═══════════════════════════════════════════════════════════════════════════
#  PARTICLE SWARM OPTIMISATION
# ═══════════════════════════════════════════════════════════════════════════
PSO_N  = 12
PSO_IT = 40
PSO_W, PSO_C1, PSO_C2 = 0.6, 1.5, 1.5

BOUNDS = np.array([[10., 100.], [150., 900.], [0.2, 5.0]])

def _pso_thread():
    global pso_active, bp_sos, env_sos
    pso_stop_event.clear()
    n, d   = PSO_N, 3
    lo, hi = BOUNDS[:, 0], BOUNDS[:, 1]
    rng    = np.random.default_rng()

    pos = rng.uniform(lo, hi, (n, d))
    vel = rng.uniform(-(hi-lo)*0.1, (hi-lo)*0.1, (n, d))
    pos[0] = [cfg.LOW, cfg.HIGH, cfg.AMP_LIMIT]   # seed with current

    pbest, pbest_fit = pos.copy(), np.zeros(n)
    gbest, gbest_fit = pos[0].copy(), 0.0

    with pso_lock:
        pso_status.update({"running": True, "iteration": 0,
                           "best_fitness": 0.0, "log": []})

    for it in range(PSO_IT):
        if pso_stop_event.is_set():
            break
        with buf_lock:
            snap = buffer.copy()
        for i in range(n):
            f = evaluate_params(pos[i,0], pos[i,1], pos[i,2], snap)
            if f > pbest_fit[i]:
                pbest_fit[i] = f
                pbest[i]     = pos[i].copy()
            if f > gbest_fit:
                gbest_fit = f
                gbest     = pos[i].copy()
        r1 = rng.random((n, d))
        r2 = rng.random((n, d))
        vel = PSO_W*vel + PSO_C1*r1*(pbest-pos) + PSO_C2*r2*(gbest-pos)
        pos = np.clip(pos + vel, lo, hi)

        with pso_lock:
            pso_status.update({
                "iteration":    it+1,
                "best_fitness": round(gbest_fit, 4),
                "best_low":     round(float(gbest[0]), 1),
                "best_high":    round(float(gbest[1]), 1),
                "best_amp":     round(float(gbest[2]), 2),
            })
            pso_status["log"].append({
                "iter": it+1, "fitness": round(gbest_fit, 4),
                "low":  round(float(gbest[0]),1),
                "high": round(float(gbest[1]),1),
                "amp":  round(float(gbest[2]),2),
            })
        time.sleep(0.05)

    # Apply best result
    cfg.LOW       = round(float(gbest[0]), 1)
    cfg.HIGH      = round(float(gbest[1]), 1)
    cfg.AMP_LIMIT = round(float(gbest[2]), 2)
    bp_sos, env_sos = make_filters()

    with pso_lock:
        pso_status["running"] = False
    pso_active = False

# ═══════════════════════════════════════════════════════════════════════════
#  MAIN PROCESSING LOOP
# ═══════════════════════════════════════════════════════════════════════════
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

            freqs, _, Zxx = stft(filt, fs=cfg.SR, nperseg=cfg.NFFT,
                                  noverlap=cfg.OVERLAP, window="hann",
                                  boundary=None, padded=False)
            Pxx   = np.abs(Zxx)**2
            Pdb   = 10 * np.log10(Pxx + 1e-12)
            Pnorm = (Pdb - Pdb.min()) / (np.ptp(Pdb) + 1e-12)

            prob = 0.5
            if MODEL_LOADED and TORCH_AVAILABLE:
                spec_t = torch.tensor(Pnorm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                env_t  = torch.tensor(envelope, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
                with torch.no_grad():
                    prob = model(spec_t, env_t).item()

            conf_buffer.append(prob)
            mean_conf = float(np.mean(conf_buffer))
            heartbeat = (len(conf_buffer) == cfg.CONF_HISTORY and
                         mean_conf > cfg.MEAN_THRESH and
                         sum(v > cfg.CONF_THRESH for v in conf_buffer) >= 3)

            bpm_val, s1_times, s2_times = None, [], []
            if heartbeat:
                bpm_raw, s1, s2 = detect_bpm(envelope, cfg.SR)
                if bpm_raw:
                    bpm_buffer.append(bpm_raw)
                    bpm_val = float(median(bpm_buffer))
                    HB_HISTORY.append(bpm_val)
                    bpm_time_buffer.append(bpm_val)
                    s1_times = [float(i)/cfg.SR - cfg.BLOCK_SEC for i in s1]
                    s2_times = [float(i)/cfg.SR - cfg.BLOCK_SEC for i in s2]
            else:
                if bpm_time_buffer:
                    bpm_time_buffer.append(bpm_time_buffer[-1])

            if is_recording:
                ts = time.time() - (rec_start_ts or time.time())
                with rec_lock:
                    step = max(1, len(envelope)//256)
                    rec_env_buf.append({"t": round(ts,2), "env": envelope[::step].tolist()})
                    if bpm_val:
                        rec_bpm_buf.append({"t": round(ts,2), "bpm": round(bpm_val,1)})
                    rec_conf_buf.append({"t": round(ts,2), "conf": round(float(prob),3)})
                    if len(rec_spec_buf) < 30:
                        rec_spec_buf.append(Pnorm.copy())

            step     = max(1, len(envelope)//512)
            env_ds   = envelope[::step].tolist()
            freq_mask = freqs <= cfg.HIGH
            spec_sl  = Pnorm[freq_mask, :]
            cs       = max(1, spec_sl.shape[1]//128)
            spec_sl  = spec_sl[:, ::cs]

            with pso_lock:
                pso_snap = {k: pso_status[k] for k in
                            ("running","iteration","best_fitness","best_low","best_high","best_amp")}

            frame = {
                "heartbeat":  heartbeat,
                "confidence": round(float(prob), 3),
                "mean_conf":  round(mean_conf, 3),
                "bpm":        round(bpm_val, 1) if bpm_val else None,
                "s1_times":   s1_times,
                "s2_times":   s2_times,
                "envelope":   env_ds,
                "amp_limit":  cfg.AMP_LIMIT,
                "spec_flat":  spec_sl.flatten().tolist(),
                "spec_rows":  int(spec_sl.shape[0]),
                "spec_cols":  int(spec_sl.shape[1]),
                "bpm_hist":   list(bpm_time_buffer),
                "block_sec":  cfg.BLOCK_SEC,
                "recording":  is_recording,
                "rec_elapsed": round(time.time() - rec_start_ts, 1) if is_recording and rec_start_ts else 0,
                "pso":        pso_snap,
                "cfg":        {"low": cfg.LOW, "high": cfg.HIGH, "amp_limit": cfg.AMP_LIMIT},
            }

            if data_queue.full():
                try: data_queue.get_nowait()
                except queue.Empty: pass
            data_queue.put(frame)
        except Exception:
            pass

# ═══════════════════════════════════════════════════════════════════════════
#  RECORDING SAVE
# ═══════════════════════════════════════════════════════════════════════════
def save_recording(patient):
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe   = re.sub(r"[^a-zA-Z0-9_\- ]", "", patient.get("name","unknown")).strip().replace(" ","_")
    folder = os.path.join(RECORDINGS_DIR, f"{ts_str}_{safe}")
    os.makedirs(folder, exist_ok=True)

    with rec_lock:
        audio_chunks = list(rec_audio_buf)
        env_frames   = list(rec_env_buf)
        bpm_frames   = list(rec_bpm_buf)
        conf_frames  = list(rec_conf_buf)
        spec_frames  = list(rec_spec_buf)

    # WAV
    audio_filename = None
    if audio_chunks:
        audio_arr = np.concatenate(audio_chunks).astype(np.float32)
        if SF_AVAILABLE:
            wav_path = os.path.join(folder, "audio.wav")
            sf.write(wav_path, audio_arr, cfg.SR)
            audio_filename = "audio.wav"
        else:
            audio_arr.tofile(os.path.join(folder, "audio.raw"))
            audio_filename = "audio.raw"

    # Spectrogram PNG
    spec_filename = None
    if MPL_AVAILABLE and spec_frames:
        avg_spec = np.mean(spec_frames, axis=0)
        fig, ax = plt.subplots(figsize=(12, 4), facecolor="#040b14")
        ax.imshow(avg_spec, origin="lower", aspect="auto", cmap="inferno", vmin=0, vmax=1)
        ax.set_facecolor("#040b14")
        ax.set_title(f"Spectrogram — {patient.get('name','')}  [{datetime.now().strftime('%Y-%m-%d %H:%M')}]",
                     color="#c8dff0", fontsize=11)
        ax.set_xlabel("Time bins", color="#7aaabf"); ax.set_ylabel("Freq bins", color="#7aaabf")
        ax.tick_params(colors="#3d6a88")
        for sp in ax.spines.values(): sp.set_edgecolor("#1a3a5c")
        plt.tight_layout()
        spec_filename = "spectrogram.png"
        plt.savefig(os.path.join(folder, spec_filename), dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)

    # Envelope + BPM PNG
    env_filename = None
    if MPL_AVAILABLE and env_frames:
        fig, axes = plt.subplots(2, 1, figsize=(12, 5), facecolor="#040b14")
        last_env = env_frames[-1]["env"]
        t_env = np.linspace(0, cfg.BLOCK_SEC, len(last_env))
        axes[0].plot(t_env, last_env, color="#00d4ff", lw=1.2)
        axes[0].fill_between(t_env, last_env, alpha=0.15, color="#00d4ff")
        axes[0].set_facecolor("#071220")
        axes[0].set_title("Signal Envelope (last window)", color="#c8dff0")
        axes[0].set_ylabel("Amplitude", color="#7aaabf")
        axes[0].tick_params(colors="#3d6a88")
        if bpm_frames:
            bpm_t = [b["t"] for b in bpm_frames]
            bpm_v = [b["bpm"] for b in bpm_frames]
            axes[1].plot(bpm_t, bpm_v, color="#ff8c00", lw=1.5, marker="o", ms=3)
            axes[1].axhline(60,  color="#4488ff", lw=0.8, ls="--", alpha=0.5)
            axes[1].axhline(100, color="#00ff88", lw=0.8, ls="--", alpha=0.5)
        axes[1].set_facecolor("#071220")
        axes[1].set_title("BPM Over Session", color="#c8dff0")
        axes[1].set_xlabel("Time (s)", color="#7aaabf"); axes[1].set_ylabel("BPM", color="#7aaabf")
        axes[1].tick_params(colors="#3d6a88")
        for ax in axes:
            for sp in ax.spines.values(): sp.set_edgecolor("#1a3a5c")
        plt.tight_layout()
        env_filename = "envelope_bpm.png"
        plt.savefig(os.path.join(folder, env_filename), dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)

    duration  = round(time.time() - rec_start_ts, 1) if rec_start_ts else 0.0
    bpm_vals  = [b["bpm"] for b in bpm_frames]
    folder_name = os.path.basename(folder)

    meta = {
        "id":         rec_id,
        "_folder":    folder_name,
        "timestamp":  datetime.now().isoformat(),
        "patient":    patient,
        "duration_s": duration,
        "session": {
            "mean_bpm":  round(float(np.mean(bpm_vals)), 1) if bpm_vals else None,
            "min_bpm":   round(float(np.min(bpm_vals)),  1) if bpm_vals else None,
            "max_bpm":   round(float(np.max(bpm_vals)),  1) if bpm_vals else None,
            "detections": len(bpm_frames),
            "mean_conf":  round(float(np.mean([c["conf"] for c in conf_frames])), 3) if conf_frames else None,
        },
        "config": {"low": cfg.LOW, "high": cfg.HIGH, "amp_limit": cfg.AMP_LIMIT, "sr": cfg.SR},
        "files": {
            "audio":       audio_filename,
            "spectrogram": spec_filename,
            "envelope":    env_filename,
        },
        "bpm_timeline":  bpm_frames,
        "conf_timeline": conf_frames,
    }
    with open(os.path.join(folder, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta

# ═══════════════════════════════════════════════════════════════════════════
#  HISTORY
# ═══════════════════════════════════════════════════════════════════════════
def load_all_recordings():
    records = []
    if not os.path.isdir(RECORDINGS_DIR):
        return records
    for entry in sorted(os.listdir(RECORDINGS_DIR), reverse=True):
        mp = os.path.join(RECORDINGS_DIR, entry, "metadata.json")
        if os.path.exists(mp):
            try:
                with open(mp) as f:
                    m = json.load(f)
                m["_folder"] = entry
                records.append(m)
            except Exception:
                pass
    return records

# ═══════════════════════════════════════════════════════════════════════════
#  FLASK APP
# ═══════════════════════════════════════════════════════════════════════════
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html", deps=DEPS, model_loaded=MODEL_LOADED)

@app.route("/stream")
def stream():
    def gen():
        while True:
            try:
                frame = data_queue.get(timeout=1.0)
                yield f"data: {json.dumps(frame)}\n\n"
            except queue.Empty:
                yield "data: {}\n\n"
    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

@app.route("/start", methods=["POST"])
def start_audio():
    global bp_sos, env_sos, BLOCK, buffer
    if not AUDIO_AVAILABLE:
        return jsonify({"ok": False, "error": "sounddevice not installed"})
    try:
        BLOCK = int(cfg.SR * cfg.BLOCK_SEC); buffer = np.zeros(BLOCK)
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
        v = float(data["high"])
        if 100 <= v <= 1000: cfg.HIGH = v
    if "amp_limit" in data:
        v = float(data["amp_limit"])
        if 0.1 <= v <= 10.0: cfg.AMP_LIMIT = v
    if "low" in data:
        v = float(data["low"])
        if 10 <= v <= 100: cfg.LOW = v
    bp_sos, env_sos = make_filters()
    return jsonify({"ok": True, "high": cfg.HIGH, "amp_limit": cfg.AMP_LIMIT, "low": cfg.LOW})

@app.route("/status")
def status_route():
    return jsonify({
        "deps": DEPS, "model": MODEL_LOADED, "audio": AUDIO_AVAILABLE,
        "active": audio_active.is_set(),
        "config": {"low": cfg.LOW, "high": cfg.HIGH, "amp_limit": cfg.AMP_LIMIT, "sr": cfg.SR},
    })

# PSO
@app.route("/pso/start", methods=["POST"])
def pso_start():
    global pso_active
    if pso_active: return jsonify({"ok": False, "error": "PSO already running"})
    if not audio_active.is_set(): return jsonify({"ok": False, "error": "Start audio first"})
    pso_active = True
    threading.Thread(target=_pso_thread, daemon=True).start()
    return jsonify({"ok": True})

@app.route("/pso/stop", methods=["POST"])
def pso_stop():
    pso_stop_event.set()
    return jsonify({"ok": True})

@app.route("/pso/status")
def pso_status_route():
    with pso_lock:
        return jsonify(dict(pso_status))

# Recording
@app.route("/record/start", methods=["POST"])
def record_start():
    global is_recording, rec_start_ts, rec_id
    global rec_audio_buf, rec_env_buf, rec_bpm_buf, rec_conf_buf, rec_spec_buf
    if not audio_active.is_set():
        return jsonify({"ok": False, "error": "Start audio first"})
    with rec_lock:
        rec_audio_buf = []; rec_env_buf = []; rec_bpm_buf = []
        rec_conf_buf  = []; rec_spec_buf = []
        rec_start_ts  = time.time()
        rec_id        = str(uuid.uuid4())[:8]
    is_recording = True
    return jsonify({"ok": True, "rec_id": rec_id})

@app.route("/record/stop", methods=["POST"])
def record_stop():
    global is_recording
    is_recording = False
    data    = request.get_json() or {}
    patient = data.get("patient", {})
    try:
        meta = save_recording(patient)
        return jsonify({"ok": True, "meta": meta})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

@app.route("/record/cancel", methods=["POST"])
def record_cancel():
    global is_recording
    global rec_audio_buf, rec_env_buf, rec_bpm_buf, rec_conf_buf, rec_spec_buf
    is_recording = False
    with rec_lock:
        rec_audio_buf = []; rec_env_buf = []; rec_bpm_buf = []
        rec_conf_buf  = []; rec_spec_buf = []
    return jsonify({"ok": True})

# History
@app.route("/history/list")
def history_list():
    return jsonify({"records": load_all_recordings()})

@app.route("/history/<folder>/<filename>")
def history_file(folder, filename):
    return send_from_directory(os.path.join(RECORDINGS_DIR, folder), filename)

@app.route("/history/<folder>/delete", methods=["DELETE"])
def history_delete(folder):
    path = os.path.join(RECORDINGS_DIR, folder)
    if os.path.isdir(path):
        shutil.rmtree(path); return jsonify({"ok": True})
    return jsonify({"ok": False, "error": "Not found"})

@app.route("/history")
def get_history():
    return jsonify({"history": HB_HISTORY})

@app.route("/devices")
def get_devices():
    if not AUDIO_AVAILABLE: return jsonify({"devices": []})
    devs = sd.query_devices()
    return jsonify({"devices": [
        {"id": i, "name": d["name"], "sr": d["default_samplerate"]}
        for i, d in enumerate(devs) if d["max_input_channels"] > 0
    ]})

# ═══════════════════════════════════════════════════════════════════════════
#  AUDIO THREAD
# ═══════════════════════════════════════════════════════════════════════════
def run_audio():
    stream_obj = None
    while True:
        if audio_active.is_set():
            if stream_obj is None or not stream_obj.active:
                try:
                    stream_obj = sd.InputStream(
                        samplerate=cfg.SR, channels=1,
                        callback=audio_callback, blocksize=256)
                    stream_obj.start()
                except Exception:
                    audio_active.clear()
            time.sleep(0.05)
        else:
            if stream_obj and stream_obj.active:
                stream_obj.stop(); stream_obj.close(); stream_obj = None
            time.sleep(0.1)

# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    threading.Thread(target=processing_loop, daemon=True).start()
    threading.Thread(target=run_audio,       daemon=True).start()
    port = 5757
    threading.Thread(target=lambda: (time.sleep(1.2), webbrowser.open(f"http://127.0.0.1:{port}")), daemon=True).start()
    print(f"\n  🩺  CardioScope v3.0  →  http://127.0.0.1:{port}")
    print(f"  📁  Recordings       →  {RECORDINGS_DIR}\n")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
