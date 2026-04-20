# CardioScope — Digital Stethoscope
### Professional Auscultation Platform · v2.0

---

## Architecture

```
cardioscope/
├── app.py                  ← Python backend (Flask + DSP + PyTorch)
├── templates/index.html    ← Full-screen web UI
├── requirements.txt
├── cardioscope.spec        ← PyInstaller build config
├── build_linux.sh          ← One-click Linux build
├── build_windows.ps1       ← One-click Windows build
└── model_006.pt            ← (your trained model — place here)
```

---

## Quick Start (Development — no build needed)

### Linux / Ubuntu
```bash
# Install system audio lib (one time)
sudo apt install libportaudio2

# Install Python deps
pip install numpy scipy flask sounddevice
pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU-only

# Run
python app.py
# Opens at http://127.0.0.1:5757
```

### Windows
```powershell
pip install numpy scipy flask sounddevice
pip install torch --index-url https://download.pytorch.org/whl/cpu
python app.py
```

---

## Building a Standalone Executable (No Python Required on Target)

### Linux → Linux Binary
```bash
chmod +x build_linux.sh
./build_linux.sh
# Output: dist/CardioScope  (single file, ~300-500 MB)
```

### Windows → Windows .exe
```powershell
# In PowerShell:
.\build_windows.ps1
# Output: dist\CardioScope.exe
```

> **Cross-compilation is NOT supported by PyInstaller.**
> You must run the build script on the same OS as the target.
> Build on Ubuntu → Linux binary. Build on Windows → .exe.

---

## Distributing to End Users

### Linux
- Copy `dist/CardioScope` to the target machine
- Run: `./CardioScope`
- No Python, no pip, no dependencies needed
- **PortAudio system lib is required**: `sudo apt install libportaudio2`
  (automatically installed by build script; end users need it separately)

### Windows
- Copy `dist/CardioScope.exe` to the target machine
- Double-click to run
- **No installation required** — all deps bundled
- Windows may show a SmartScreen warning (unsigned binary) — click "More info → Run anyway"
- For proper distribution, consider code-signing the .exe

---

## Dependency Notes

| Package      | Required | Purpose                        |
|-------------|----------|-------------------------------|
| flask        | ✅ Yes   | Web server + SSE stream        |
| numpy        | ✅ Yes   | Array math                     |
| scipy        | ✅ Yes   | Bandpass filter, STFT, Hilbert |
| sounddevice  | ✅ Yes   | Microphone / audio input       |
| torch        | ⚠ Optional | CNN model inference          |

If `torch` is not installed or `model_006.pt` is not present, the app runs in
**signal-only mode** — envelope and spectrogram work, but heartbeat detection
confidence will default to 0.5 (you'll see heartbeats only via BPM peaks).

---

## Settings (In-App)

Click **⚙ SETTINGS** to adjust:

- **Upper Cutoff Frequency** (100–1000 Hz): Default 500 Hz. Heart sounds are
  typically 20–500 Hz. Raise to capture higher harmonics.
- **Lower Cutoff Frequency** (10–100 Hz): Default 30 Hz. Lower = more bass,
  more noise.
- **AMP_LIMIT** (0.1–5.0): Envelope amplitude ceiling. Reduce for quiet signals.

Settings take effect immediately without restart.

---

## Troubleshooting

**No audio / "sounddevice not installed"**
- Linux: `sudo apt install libportaudio2 && pip install sounddevice`
- Windows: `pip install sounddevice` (PortAudio bundled in wheel)

**Model not loaded warning**
- Place `model_006.pt` in the same directory as `app.py` or the executable

**Port 5757 already in use**
- Edit `port = 5757` in `app.py` to any free port

**PyInstaller build fails on torch**
- Try: `pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu`
- Or exclude torch from spec by removing it from `hiddenimports`

---

## Clinical Ranges Reference

| Range        | BPM       |
|-------------|-----------|
| Bradycardia  | < 60      |
| Normal       | 60 – 100  |
| Tachycardia  | 101 – 149 |
| Critical     | ≥ 150     |

---

*CardioScope is a research and educational tool. Not cleared for clinical diagnosis.*
