# Digital Stethoscope

Small utilities to capture and analyse heart sounds from a microphone or WAV files.

## Requirements
- Linux/Windows
- Python 3.8+
- A working microphone (for real-time capture)
- Recommended: virtualenv

## Install
Install dependencies using
   ```
   pip install -r requirements.txt
   ```

If there is no requirements.txt, install common packages used for audio processing:
```
pip install numpy scipy sounddevice soundfile matplotlib
```

## Usage

Run the live microphone capture CLI (ensure microphone is connected):
```
python3 MicInputStethoscopeCLI.py
```
- This script captures audio from the microphone and processes it in real time.

Analyse a recorded WAV file:
```
python3 FileBPM.py <audiofile.wav>
```
- Replace `<audiofile.wav>` with the path to your WAV file.


## Notes & Troubleshooting
- If you see audio dropouts, try lowering the sample rate or buffer size in the script.
- Use `python3 -m pip install --upgrade pip` if dependency installs fail.
