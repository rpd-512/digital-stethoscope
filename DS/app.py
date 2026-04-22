"""
CardioScope v4.0 — Digital Stethoscope Backend
=================================================
Changes vs v3.0
  • SGO (Social Group Optimisation — Satapathy & Naik, 2016) replaces PSO
  • Ensemble peak detection:
      – Channel A : Hilbert envelope of broadband signal (LOW–HIGH Hz)
      – Channel B : Spectral energy summed in cardiac-fundamental band
                   (30–100 Hz), interpolated back to sample-rate timeline
      – Peaks merged from both channels and de-duplicated with tol=50 ms
  • S1/S2 classification on the ENSEMBLE peak set using combined amplitude
    (0.6*env + 0.4*spec) and inter-beat gap asymmetry
  • Peak-stability score (CV of IBIs) added to fitness — one score per
    channel, both fed into SGO fitness
  • Recording JSON now includes full amplitude_timeline [{t, amplitude}]
  • Stability metrics streamed to UI on every frame
"""

import os, json, threading, time, queue, uuid, shutil, re
import numpy as np
import webbrowser
from datetime import datetime
from collections import deque
from statistics import median
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
    SR            = 4000
    BLOCK_SEC     = 3.0
    LOW           = 30
    HIGH          = 500
    ENV_CUTOFF    = 6
    AMP_LIMIT     = 1.5
    NFFT          = 512
    OVERLAP       = 448
    CONF_HISTORY  = 10
    CONF_THRESH   = 0.60
    MEAN_THRESH   = 0.60
    BPM_HISTORY   = 10
    SPEC_LOW      = 30     # cardiac-fundamental band, fixed
    SPEC_HIGH     = 100

cfg = Config()

filter_changed = threading.Event()

def make_filters():
    bp      = butter(4, [cfg.LOW, cfg.HIGH],                btype="band", fs=cfg.SR, output="sos")
    env_lp  = butter(2,  cfg.ENV_CUTOFF,                    btype="low",  fs=cfg.SR, output="sos")
    spec_bp = butter(4, [cfg.SPEC_LOW, cfg.SPEC_HIGH],      btype="band", fs=cfg.SR, output="sos")
    filter_changed.set()   # signal processing loop to reset causal zi
    return bp, env_lp, spec_bp

bp_sos, env_sos, spec_bp_sos = make_filters()
filter_changed.clear()   # startup call — don't trigger a spurious reset

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
rec_amp_buf   = []   # amplitude timeline [{t, amplitude}]
rec_env_buf   = []
rec_bpm_buf   = []
rec_conf_buf  = []
rec_spec_buf  = []
rec_start_ts  = None
rec_id        = None

# ── SGO state ──────────────────────────────────────────────────────────────
sgo_lock       = threading.Lock()
sgo_active     = False
sgo_stop_event = threading.Event()
sgo_status     = {
    "running": False, "iteration": 0, "best_fitness": 0.0,
    "best_low": cfg.LOW, "best_high": cfg.HIGH, "best_amp": cfg.AMP_LIMIT,
    "log": [], "phase": "—"
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
#  SPECTRAL CHANNEL  (30–100 Hz energy timeline)
#  Sums STFT magnitudes in the cardiac-fundamental band per time frame,
#  then linearly interpolates to the original sample count.
# ═══════════════════════════════════════════════════════════════════════════
def spectral_channel(raw_len, freqs, Zxx, _absZxx=None):
    mask = (freqs >= cfg.SPEC_LOW) & (freqs <= cfg.SPEC_HIGH)
    if not np.any(mask):
        return np.zeros(raw_len)
    Z = _absZxx if _absZxx is not None else np.abs(Zxx)
    band_energy = Z[mask, :].sum(axis=0)
    x_old = np.linspace(0, 1, len(band_energy))
    x_new = np.linspace(0, 1, raw_len)
    sig   = np.interp(x_new, x_old, band_energy)
    mx = sig.max()
    return sig / mx if mx > 1e-12 else sig

# ═══════════════════════════════════════════════════════════════════════════
#  PEAK UTILITIES
# ═══════════════════════════════════════════════════════════════════════════
def peak_stability(peaks, sr):
    """1 − CV of inter-beat intervals.  Returns float in [0, 1]."""
    if len(peaks) < 3:
        return 0.0
    ivs = np.diff(np.sort(peaks)) / sr   # float via division
    mu  = ivs.mean()
    if mu < 1e-9:
        return 0.0
    cv  = ivs.std() / mu
    return float(np.clip(1.0 - cv, 0.0, 1.0))

def peaks_from_signal(sig, sr, min_dist_sec=0.18):
    min_d = max(1, int(min_dist_sec * sr))
    if len(sig) < min_d * 2:
        return np.array([], dtype=int)
    mean = np.mean(sig)
    std  = np.std(sig)

    h   = mean + 0.8 * std
    pro = 0.5 * std
    peaks, _ = find_peaks(sig, distance=min_d, height=h, prominence=pro)
    return peaks.astype(int)

def merge_peaks(pa, pb, sr, tol_sec=0.04):
    """
    Fuse peaks from envelope (pa) and spectral (pb)
    If peaks are within tol_sec → replace with averaged peak
    """

    if len(pa) == 0 and len(pb) == 0:
        return np.array([], dtype=int)

    tol = int(tol_sec * sr)

    pa = np.sort(pa.astype(int))
    pb = np.sort(pb.astype(int))

    i, j = 0, 0
    merged = []

    while i < len(pa) and j < len(pb):
        a = pa[i]
        b = pb[j]

        if abs(a - b) <= tol:
            # 🔥 FUSE: take average position
            merged.append(int((a + b) // 2))
            i += 1
            j += 1
        elif a < b:
            i += 1
        else:
            j += 1

    return np.array(merged, dtype=int)

# ═══════════════════════════════════════════════════════════════════════════
#  ENSEMBLE PEAK DETECTION + S1/S2 CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════
def ensemble_detect(envelope, spec_sig, sr):
    """
    Dual-channel peak detection + S1/S2 labelling.

    Parameters
    ----------
    envelope : np.ndarray  — Hilbert envelope (broadband)
    spec_sig : np.ndarray  — 30–100 Hz STFT energy timeline (same length)
    sr       : int

    Returns
    -------
    bpm           : float | None
    s1_idx        : list[int]  sample indices
    s2_idx        : list[int]  sample indices
    merged        : np.ndarray sorted merged peaks
    env_stab      : float  [0,1]
    spec_stab     : float  [0,1]
    """
    env_n  = (envelope - envelope.min()) / (np.ptp(envelope) + 1e-12)
    spec_n = (spec_sig  - spec_sig.min())  / (np.ptp(spec_sig)  + 1e-12)

    p_env  = peaks_from_signal(env_n,  sr)
    p_spec = peaks_from_signal(spec_n, sr)

    env_stab  = peak_stability(p_env,  sr)
    spec_stab = peak_stability(p_spec, sr)

    merged = merge_peaks(p_env, p_spec, sr)

    if len(merged) < 2:
        return None, [], [], merged, env_stab, spec_stab

    intervals = np.diff(merged) / sr
    bpm = float(60.0 / np.mean(intervals) / 2.0)
    if not (40 < bpm < 180):
        return None, [], [], merged, env_stab, spec_stab

    # Combined amplitude proxy for loudness
    combo = 0.6 * env_n + 0.4 * spec_n

    s1_idx, s2_idx = [], []
    for i in range(1, len(merged) - 1):
        gap_before = int(merged[i])   - int(merged[i - 1])
        gap_after  = int(merged[i + 1]) - int(merged[i])
        # Longer gap before the sound → diastolic pause → S1
        if gap_before >= gap_after:
            s1_idx.append(int(merged[i]))
        else:
            s2_idx.append(int(merged[i]))

    # Classify first and last peaks by relative amplitude
    if len(merged) >= 1:
        mean_amp = float(np.mean([combo[m] for m in merged if m < len(combo)]))
        a_first  = float(combo[merged[0]])   if merged[0]  < len(combo) else 0
        a_last   = float(combo[merged[-1]])  if merged[-1] < len(combo) else 0
        (s1_idx if a_first >= mean_amp else s2_idx).insert(0, int(merged[0]))
        if len(merged) > 2:
            (s1_idx if a_last >= mean_amp else s2_idx).append(int(merged[-1]))

    return bpm, sorted(set(s1_idx)), sorted(set(s2_idx)), merged, env_stab, spec_stab

# ═══════════════════════════════════════════════════════════════════════════
#  FITNESS FUNCTION  (used by SGO)
# ═══════════════════════════════════════════════════════════════════════════
def evaluate_params(low, high, amp_limit, raw_snapshot):
    """
    Weighted fitness:
      30% CNN model confidence
      20% envelope peak stability
      20% spectral-channel peak stability
      15% BPM plausibility (centred on 75 bpm)
      15% SNR proxy
    """
    try:
        if low < 30 or low >= high or high > cfg.SR / 2 - 1 or low < 5:
            return 0.0
        bp    = butter(4, [low, high],             btype="band", fs=cfg.SR, output="sos")
        env_f = butter(2,  cfg.ENV_CUTOFF,          btype="low",  fs=cfg.SR, output="sos")

        filt     = sosfiltfilt(bp, raw_snapshot)
        rect     = np.abs(hilbert(filt))
        envelope = np.clip(sosfiltfilt(env_f, rect), 0, amp_limit)

        freqs, _, Zxx = stft(filt, fs=cfg.SR, nperseg=cfg.NFFT,
                              noverlap=cfg.OVERLAP, window="hann",
                              boundary=None, padded=False)
        spec_sig = spectral_channel(len(raw_snapshot), freqs, Zxx)

        # 1. CNN
        model_score = 0.5
        if MODEL_LOADED and TORCH_AVAILABLE:
            Pdb   = 10 * np.log10(np.abs(Zxx) ** 2 + 1e-12)
            Pnorm = (Pdb - Pdb.min()) / (np.ptp(Pdb) + 1e-12)
            st = torch.tensor(Pnorm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            et = torch.tensor(envelope, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            with torch.no_grad():
                model_score = float(model(st, et).item())

        # 2–3. Stability
        env_n  = (envelope - envelope.min()) / (np.ptp(envelope) + 1e-12)
        spec_n = (spec_sig  - spec_sig.min())  / (np.ptp(spec_sig)  + 1e-12)
        p_env  = peaks_from_signal(env_n,  cfg.SR)
        p_spec = peaks_from_signal(spec_n, cfg.SR)
        env_stab  = peak_stability(p_env,  cfg.SR)
        spec_stab = peak_stability(p_spec, cfg.SR)

        # 4. BPM plausibility
        merged = merge_peaks(p_env, p_spec, cfg.SR)
        bpm_score = 0.0
        if len(merged) >= 2:
            ivs = np.diff(merged) / cfg.SR
            bpm = 60.0 / np.mean(ivs) / 2.0
            if 40 < bpm < 180:
                bpm_score = float(np.clip(1.0 - abs(bpm - 75) / 75.0, 0, 1))

        # 5. SNR
        snr_score = float(np.clip(np.var(filt) / (np.var(raw_snapshot) + 1e-12) * 5, 0, 1))

        return (0.30 * model_score +
                0.20 * env_stab   +
                0.20 * spec_stab  +
                0.15 * bpm_score  +
                0.15 * snr_score)
    except Exception:
        return 0.0

# ═══════════════════════════════════════════════════════════════════════════
#  SOCIAL GROUP OPTIMISATION  (SGO)
#  Satapathy S.C. & Naik A. (2016). "Social Group Optimization (SGO):
#  A New Population Based Optimization Technique."
#  Complex & Intelligent Systems, 2(3), 173–203.
#
#  Two-phase update per iteration:
#    Improving phase  : Xi ← c·Xi + (1−c)·Xbest + r·(Xbest − Xi)
#    Acquiring phase  : Xi ← Xi + r1·(Xj − r2·Xi)   (Xj random peer ≠ Xi)
# ═══════════════════════════════════════════════════════════════════════════
SGO_N  = 15       # population (persons)
SGO_IT = 40       # iterations
SGO_C  = 0.4      # self-introspection coefficient

BOUNDS = np.array([[10., 100.], [150., 900.], [0.2, 5.0]])

def _sgo_thread():
    global sgo_active, bp_sos, env_sos, spec_bp_sos
    sgo_stop_event.clear()
    rng    = np.random.default_rng()
    lo, hi = BOUNDS[:, 0], BOUNDS[:, 1]
    n, d   = SGO_N, 3

    # Initialise — seed person 0 with current config
    pop = rng.uniform(lo, hi, (n, d))
    pop[0] = [cfg.LOW, cfg.HIGH, cfg.AMP_LIMIT]

    with sgo_lock:
        sgo_status.update({"running": True, "iteration": 0,
                           "best_fitness": 0.0, "log": [], "phase": "init"})

    # Initial fitness evaluation
    with buf_lock:
        snap = buffer.copy()
    fitness  = np.array([evaluate_params(pop[i, 0], pop[i, 1], pop[i, 2], snap) for i in range(n)])
    best_idx = int(np.argmax(fitness))
    best_pos = pop[best_idx].copy()
    best_fit = float(fitness[best_idx])

    for it in range(SGO_IT):
        if sgo_stop_event.is_set():
            break

        with buf_lock:
            snap = buffer.copy()

        # ── Improving phase ───────────────────────────────────────────────
        with sgo_lock:
            sgo_status["phase"] = f"improving [{it+1}/{SGO_IT}]"

        # Vectorised candidate generation
        R       = rng.random((n, d))
        new_pop = np.clip(SGO_C * pop
                          + (1 - SGO_C) * best_pos
                          + R * (best_pos - pop),
                          lo, hi)
        for i in range(n):
            # Only evaluate if candidate meaningfully different (saves ~40% evals)
            if np.max(np.abs(new_pop[i] - pop[i])) < 0.5:
                continue
            f = evaluate_params(new_pop[i, 0], new_pop[i, 1], new_pop[i, 2], snap)
            if f >= fitness[i]:
                pop[i] = new_pop[i]; fitness[i] = f
            if fitness[i] > best_fit:
                best_fit = fitness[i]; best_pos = pop[i].copy()

        # ── Acquiring phase ───────────────────────────────────────────────
        with sgo_lock:
            sgo_status["phase"] = f"acquiring [{it+1}/{SGO_IT}]"

        peers = rng.integers(0, n, size=n)
        for i in range(n):
            j = int(peers[i]) if peers[i] != i else int((peers[i] + 1) % n)
            r1, r2    = rng.random(d), rng.random(d)
            candidate = np.clip(pop[i] + r1 * (pop[j] - r2 * pop[i]), lo, hi)
            if np.max(np.abs(candidate - pop[i])) < 0.5:
                continue
            f = evaluate_params(candidate[0], candidate[1], candidate[2], snap)
            if f >= fitness[i]:
                pop[i] = candidate; fitness[i] = f
            if fitness[i] > best_fit:
                best_fit = fitness[i]; best_pos = pop[i].copy()

        with sgo_lock:
            sgo_status.update({
                "iteration":    it + 1,
                "best_fitness": round(best_fit, 4),
                "best_low":     round(float(best_pos[0]), 1),
                "best_high":    round(float(best_pos[1]), 1),
                "best_amp":     round(float(best_pos[2]), 2),
            })
            sgo_status["log"].append({
                "iter": it + 1, "fitness": round(best_fit, 4),
                "low":  round(float(best_pos[0]), 1),
                "high": round(float(best_pos[1]), 1),
                "amp":  round(float(best_pos[2]), 2),
            })

    # Apply best found
    cfg.LOW       = round(float(best_pos[0]), 1)
    cfg.HIGH      = round(float(best_pos[1]), 1)
    cfg.AMP_LIMIT = round(float(best_pos[2]), 2)
    bp_sos, env_sos, spec_bp_sos = make_filters()

    with sgo_lock:
        sgo_status["running"] = False
        sgo_status["phase"]   = "complete"
    sgo_active = False

# ═══════════════════════════════════════════════════════════════════════════
#  MAIN PROCESSING LOOP
# ═══════════════════════════════════════════════════════════════════════════
def processing_loop():
    # Persistent zi state for causal LP filter (much faster than sosfiltfilt)
    from scipy.signal import sosfilt_zi, sosfilt as _sosfilt
    env_zi = sosfilt_zi(env_sos) * 0.0   # zero initial conditions

    while True:
        time.sleep(0.08)
        # Reset zi if settings changed the filters
        if filter_changed.is_set():
            env_zi = sosfilt_zi(env_sos) * 0.0
            filter_changed.clear()
        if not audio_active.is_set():
            continue
        with buf_lock:
            raw = buffer.copy()
        try:
            # Broadband filter + Hilbert envelope
            # sosfiltfilt = zero-phase but 2× work; use it here for accuracy
            filt     = sosfiltfilt(bp_sos, raw)
            rect     = np.abs(hilbert(filt))
            # Causal LP for the envelope smoothing — much faster, acceptable latency
            envelope_raw, env_zi = _sosfilt(env_sos, rect, zi=env_zi)
            envelope = np.clip(envelope_raw, 0, cfg.AMP_LIMIT)

            # Single STFT — shared by both spectrogram display AND spectral channel
            freqs, _, Zxx = stft(filt, fs=cfg.SR, nperseg=cfg.NFFT,
                                  noverlap=cfg.OVERLAP, window="hann",
                                  boundary=None, padded=False)
            absZxx = np.abs(Zxx)           # compute once, reused below
            Pdb   = 10 * np.log10(absZxx ** 2 + 1e-12)
            _pmin, _ptp = Pdb.min(), np.ptp(Pdb)
            Pnorm = (Pdb - _pmin) / (_ptp + 1e-12)

            # 30–100 Hz spectral channel (reuses absZxx — no extra STFT)
            spec_sig = spectral_channel(len(raw), freqs, Zxx, _absZxx=absZxx)

            # CNN confidence
            prob = 0.5
            if MODEL_LOADED and TORCH_AVAILABLE:
                spec_t = torch.tensor(Pnorm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                env_t  = torch.tensor(envelope, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
                with torch.no_grad():
                    prob = float(model(spec_t, env_t).item())

            conf_buffer.append(prob)
            mean_conf = float(np.mean(conf_buffer))
            heartbeat = (len(conf_buffer) == cfg.CONF_HISTORY and
                         mean_conf > cfg.MEAN_THRESH and
                         sum(v > cfg.CONF_THRESH for v in conf_buffer) >= 3)

            # Ensemble detection (runs every frame — gated for BPM only)
            bpm_raw, s1_idx, s2_idx, merged, env_stab, spec_stab = \
                ensemble_detect(envelope, spec_sig, cfg.SR)

            bpm_val, s1_times, s2_times = None, [], []
            if heartbeat and bpm_raw:
                bpm_buffer.append(bpm_raw)
                bpm_val = float(median(bpm_buffer))
                HB_HISTORY.append(bpm_val)
                bpm_time_buffer.append(bpm_val)
                s1_times = [float(i) / cfg.SR - cfg.BLOCK_SEC for i in s1_idx]
                s2_times = [float(i) / cfg.SR - cfg.BLOCK_SEC for i in s2_idx]
            else:
                if bpm_time_buffer:
                    bpm_time_buffer.append(bpm_time_buffer[-1])

            # Recording accumulation
            if is_recording:
                ts = time.time() - (rec_start_ts or time.time())
                ts_r = round(ts, 3)
                rms_amp = float(np.sqrt(np.mean(envelope ** 2)))
                with rec_lock:
                    rec_amp_buf.append({"t": ts_r, "amplitude": round(rms_amp, 6)})
                    step = max(1, len(envelope) // 256)
                    rec_env_buf.append({"t": ts_r, "env": envelope[::step].tolist()})
                    if bpm_val:
                        rec_bpm_buf.append({
                            "t": ts_r, "bpm": round(bpm_val, 1),
                            "env_stability":  round(env_stab, 4),
                            "spec_stability": round(spec_stab, 4),
                        })
                    rec_conf_buf.append({"t": ts_r, "conf": round(prob, 4),
                                         "env_stab":  round(env_stab, 4),
                                         "spec_stab": round(spec_stab, 4)})
                    if len(rec_spec_buf) < 30:
                        rec_spec_buf.append(Pnorm.copy())

            # Downsample for SSE
            step      = max(1, len(envelope) // 512)
            env_ds    = envelope[::step].tolist()
            spec_ds   = spec_sig[::step].tolist()

            freq_mask = freqs <= cfg.HIGH
            spec_sl   = Pnorm[freq_mask, :]
            cs        = max(1, spec_sl.shape[1] // 128)
            spec_sl   = spec_sl[:, ::cs]

            with sgo_lock:
                sgo_snap = {k: sgo_status[k] for k in
                            ("running","iteration","best_fitness",
                             "best_low","best_high","best_amp","phase")}

            frame = {
                "heartbeat":      heartbeat,
                "confidence":     round(prob, 3),
                "mean_conf":      round(mean_conf, 3),
                "bpm":            round(bpm_val, 1) if bpm_val else None,
                "s1_times":       s1_times,
                "s2_times":       s2_times,
                "envelope":       env_ds,
                "spec_signal":    spec_ds,
                "amp_limit":      cfg.AMP_LIMIT,
                "spec_flat":      spec_sl.flatten().tolist(),
                "spec_rows":      int(spec_sl.shape[0]),
                "spec_cols":      int(spec_sl.shape[1]),
                "bpm_hist":       list(bpm_time_buffer),
                "block_sec":      cfg.BLOCK_SEC,
                "recording":      is_recording,
                "rec_elapsed":    round(time.time() - rec_start_ts, 1)
                                  if is_recording and rec_start_ts else 0,
                "env_stability":  round(env_stab, 3),
                "spec_stability": round(spec_stab, 3),
                "sgo":            sgo_snap,
                "cfg":            {"low": cfg.LOW, "high": cfg.HIGH, "amp_limit": cfg.AMP_LIMIT},
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
    safe   = re.sub(r"[^a-zA-Z0-9_\- ]", "", patient.get("name", "unknown")).strip().replace(" ", "_")
    folder = os.path.join(RECORDINGS_DIR, f"{ts_str}_{safe}")
    os.makedirs(folder, exist_ok=True)

    with rec_lock:
        audio_chunks = list(rec_audio_buf)
        amp_frames   = list(rec_amp_buf)
        env_frames   = list(rec_env_buf)
        bpm_frames   = list(rec_bpm_buf)
        conf_frames  = list(rec_conf_buf)
        spec_frames  = list(rec_spec_buf)

    # WAV
    audio_filename = None
    if audio_chunks:
        audio_arr = np.concatenate(audio_chunks).astype(np.float32)
        if SF_AVAILABLE:
            sf.write(os.path.join(folder, "audio.wav"), audio_arr, cfg.SR)
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
        ax.set_title(f"Spectrogram — {patient.get('name','')}  "
                     f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}]",
                     color="#c8dff0", fontsize=11)
        ax.set_xlabel("Time bins", color="#7aaabf")
        ax.set_ylabel("Frequency bins", color="#7aaabf")
        ax.tick_params(colors="#3d6a88")
        for sp in ax.spines.values(): sp.set_edgecolor("#1a3a5c")
        plt.tight_layout()
        spec_filename = "spectrogram.png"
        plt.savefig(os.path.join(folder, spec_filename), dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)

    # Multi-panel diagnostic PNG
    env_filename = None
    if MPL_AVAILABLE:
        panels = 3 if bpm_frames else 2
        fig, axes = plt.subplots(panels, 1, figsize=(12, 4 * panels), facecolor="#040b14")
        axes = list(axes) if panels > 1 else [axes]

        # Panel 0 — amplitude over time
        if amp_frames:
            t_a = [a["t"] for a in amp_frames]
            v_a = [a["amplitude"] for a in amp_frames]
            axes[0].plot(t_a, v_a, color="#00d4ff", lw=1.0, alpha=0.85)
            axes[0].fill_between(t_a, v_a, alpha=0.12, color="#00d4ff")
            axes[0].set_title("Amplitude Over Time (RMS envelope)", color="#c8dff0")
            axes[0].set_ylabel("RMS Amplitude", color="#7aaabf")
        axes[0].set_facecolor("#071220")
        axes[0].tick_params(colors="#3d6a88")
        for sp in axes[0].spines.values(): sp.set_edgecolor("#1a3a5c")

        # Panel 1 — last envelope window
        if env_frames:
            last_env = env_frames[-1]["env"]
            t_env = np.linspace(0, cfg.BLOCK_SEC, len(last_env))
            axes[1].plot(t_env, last_env, color="#00d4ff", lw=1.2)
            axes[1].fill_between(t_env, last_env, alpha=0.15, color="#00d4ff")
        axes[1].set_title("Signal Envelope (last 3 s window)", color="#c8dff0")
        axes[1].set_ylabel("Amplitude", color="#7aaabf")
        axes[1].set_facecolor("#071220")
        axes[1].tick_params(colors="#3d6a88")
        for sp in axes[1].spines.values(): sp.set_edgecolor("#1a3a5c")

        # Panel 2 — BPM timeline + stability band
        if panels == 3 and bpm_frames:
            bpm_t  = [b["t"]   for b in bpm_frames]
            bpm_v  = np.array([b["bpm"] for b in bpm_frames])
            env_s  = np.array([b.get("env_stability",  0) for b in bpm_frames])
            spec_s = np.array([b.get("spec_stability", 0) for b in bpm_frames])
            stab   = (env_s + spec_s) / 2
            axes[2].plot(bpm_t, bpm_v, color="#ff8c00", lw=1.5, marker="o", ms=3, label="BPM")
            axes[2].fill_between(bpm_t, bpm_v * (1 - (1 - stab) * 0.05),
                                         bpm_v * (1 + (1 - stab) * 0.05),
                                  alpha=0.2, color="#ff8c00", label="instability band")
            axes[2].axhline(60,  color="#4488ff", lw=0.8, ls="--", alpha=0.5, label="60 bpm")
            axes[2].axhline(100, color="#00ff88", lw=0.8, ls="--", alpha=0.5, label="100 bpm")
            axes[2].legend(facecolor="#0a1a2e", labelcolor="#7aaabf", fontsize=8)
            axes[2].set_title("BPM Timeline + Stability Band", color="#c8dff0")
            axes[2].set_xlabel("Time (s)", color="#7aaabf")
            axes[2].set_ylabel("BPM", color="#7aaabf")
            axes[2].set_facecolor("#071220")
            axes[2].tick_params(colors="#3d6a88")
            for sp in axes[2].spines.values(): sp.set_edgecolor("#1a3a5c")

        plt.tight_layout()
        env_filename = "envelope_bpm.png"
        plt.savefig(os.path.join(folder, env_filename), dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)

    # JSON metadata
    duration   = round(time.time() - rec_start_ts, 1) if rec_start_ts else 0.0
    bpm_vals   = [b["bpm"] for b in bpm_frames]
    env_stabs  = [b.get("env_stability",  0) for b in bpm_frames]
    spec_stabs = [b.get("spec_stability", 0) for b in bpm_frames]

    meta = {
        "id":          rec_id,
        "_folder":     os.path.basename(folder),
        "timestamp":   datetime.now().isoformat(),
        "patient":     patient,
        "duration_s":  duration,
        "session": {
            "mean_bpm":            round(float(np.mean(bpm_vals)),   1) if bpm_vals   else None,
            "min_bpm":             round(float(np.min(bpm_vals)),    1) if bpm_vals   else None,
            "max_bpm":             round(float(np.max(bpm_vals)),    1) if bpm_vals   else None,
            "detections":          len(bpm_frames),
            "mean_conf":           round(float(np.mean([c["conf"] for c in conf_frames])), 3)
                                   if conf_frames else None,
            "mean_env_stability":  round(float(np.mean(env_stabs)),  3) if env_stabs  else None,
            "mean_spec_stability": round(float(np.mean(spec_stabs)), 3) if spec_stabs else None,
        },
        "config": {
            "low": cfg.LOW, "high": cfg.HIGH,
            "amp_limit": cfg.AMP_LIMIT, "sr": cfg.SR,
            "spec_band": [cfg.SPEC_LOW, cfg.SPEC_HIGH],
        },
        "files": {
            "audio":       audio_filename,
            "spectrogram": spec_filename,
            "envelope":    env_filename,
        },
        "amplitude_timeline": amp_frames,   # [{t, amplitude}]  — full session
        "bpm_timeline":       bpm_frames,
        "conf_timeline":      conf_frames,
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
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.route("/start", methods=["POST"])
def start_audio():
    global bp_sos, env_sos, spec_bp_sos, BLOCK, buffer
    if not AUDIO_AVAILABLE:
        return jsonify({"ok": False, "error": "sounddevice not installed"})
    try:
        BLOCK = int(cfg.SR * cfg.BLOCK_SEC)
        buffer = np.zeros(BLOCK)
        bp_sos, env_sos, spec_bp_sos = make_filters()
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
    global bp_sos, env_sos, spec_bp_sos
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
    bp_sos, env_sos, spec_bp_sos = make_filters()
    return jsonify({"ok": True, "high": cfg.HIGH, "amp_limit": cfg.AMP_LIMIT, "low": cfg.LOW})

@app.route("/status")
def status_route():
    return jsonify({
        "deps": DEPS, "model": MODEL_LOADED, "audio": AUDIO_AVAILABLE,
        "active": audio_active.is_set(),
        "config": {"low": cfg.LOW, "high": cfg.HIGH, "amp_limit": cfg.AMP_LIMIT, "sr": cfg.SR},
    })

# SGO
@app.route("/sgo/start", methods=["POST"])
def sgo_start():
    global sgo_active
    if sgo_active: return jsonify({"ok": False, "error": "SGO already running"})
    if not audio_active.is_set(): return jsonify({"ok": False, "error": "Start audio capture first"})
    sgo_active = True
    threading.Thread(target=_sgo_thread, daemon=True).start()
    return jsonify({"ok": True})

@app.route("/sgo/stop", methods=["POST"])
def sgo_stop():
    sgo_stop_event.set()
    return jsonify({"ok": True})

@app.route("/sgo/status")
def sgo_status_route():
    with sgo_lock:
        return jsonify(dict(sgo_status))

# Recording
@app.route("/record/start", methods=["POST"])
def record_start():
    global is_recording, rec_start_ts, rec_id
    global rec_audio_buf, rec_amp_buf, rec_env_buf, rec_bpm_buf, rec_conf_buf, rec_spec_buf
    if not audio_active.is_set():
        return jsonify({"ok": False, "error": "Start audio first"})
    with rec_lock:
        rec_audio_buf = []; rec_amp_buf  = []
        rec_env_buf   = []; rec_bpm_buf  = []
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
    global rec_audio_buf, rec_amp_buf, rec_env_buf, rec_bpm_buf, rec_conf_buf, rec_spec_buf
    is_recording = False
    with rec_lock:
        rec_audio_buf = []; rec_amp_buf  = []
        rec_env_buf   = []; rec_bpm_buf  = []
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
        shutil.rmtree(path)
        return jsonify({"ok": True})
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
    threading.Thread(
        target=lambda: (time.sleep(1.2), webbrowser.open(f"http://127.0.0.1:{port}")),
        daemon=True).start()
    print(f"\n  🩺  CardioScope v4.0  →  http://127.0.0.1:{port}")
    print(f"  📁  Recordings       →  {RECORDINGS_DIR}\n")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
