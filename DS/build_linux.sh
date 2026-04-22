#!/bin/bash
# ─────────────────────────────────────────────────────────────────
#  CardioScope — Linux Build Script
#  Produces:  dist/CardioScope  (single executable, ~200-400 MB)
#  Run with:  chmod +x build.sh && ./build.sh
# ─────────────────────────────────────────────────────────────────

set -e  # Exit on error

VENV_DIR=".venv_build"
DIST_DIR="dist"
APP_NAME="CardioScope"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║    CardioScope — Linux Build Pipeline    ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# ── 1. Check Python ───────────────────────────────────────
PYTHON=$(command -v python3 || command -v python)
if [ -z "$PYTHON" ]; then
  echo "❌  Python 3 not found. Install with: sudo apt install python3 python3-venv"
  exit 1
fi

PY_VER=$($PYTHON --version 2>&1)
echo "✔  Python: $PY_VER"

# ── 2. Create virtual environment ────────────────────────
echo ""
echo "→  Creating virtual environment..."
$PYTHON -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# ── 3. Upgrade pip ───────────────────────────────────────
echo "→  Upgrading pip..."
pip install --upgrade pip --quiet

# ── 4. Install dependencies ───────────────────────────────
echo "→  Installing dependencies (this may take a few minutes)..."
echo "   • numpy scipy flask sounddevice"
pip install numpy scipy flask sounddevice matplotlib soundfile --quiet

echo "   • PyTorch (CPU only — smaller build)"
# CPU-only torch is ~700 MB smaller than GPU version
pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet

echo "   • PyInstaller"
pip install pyinstaller --quiet

# ── 5. System libs check (sounddevice needs PortAudio) ───
echo ""
echo "→  Checking system audio libraries..."
if ! ldconfig -p | grep -q libportaudio 2>/dev/null; then
  echo "   ⚠  PortAudio not found. Installing..."
  sudo apt-get install -y libportaudio2 libportaudiocpp0 portaudio19-dev 2>/dev/null || \
  echo "   ⚠  Could not auto-install PortAudio. Run: sudo apt install libportaudio2"
else
  echo "   ✔  PortAudio found"
fi

# ── 6. Build ──────────────────────────────────────────────
echo ""
echo "→  Building executable with PyInstaller..."
pyinstaller cardioscope.spec --clean --noconfirm

# ── 7. Package ────────────────────────────────────────────
echo ""
echo "→  Packaging..."
cp README_LINUX.txt "$DIST_DIR/" 2>/dev/null || true

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║              BUILD COMPLETE               ║"
echo "╠══════════════════════════════════════════╣"
SIZE=$(du -sh "$DIST_DIR/$APP_NAME" 2>/dev/null | cut -f1)
echo "║  Output:  dist/$APP_NAME"
echo "║  Size:    $SIZE"
echo "║"
echo "║  Run:     ./dist/CardioScope"
echo "╚══════════════════════════════════════════╝"
echo ""

deactivate
