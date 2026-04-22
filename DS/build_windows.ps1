# ─────────────────────────────────────────────────────────────────
#  CardioScope — Windows Build Script (PowerShell)
#  Produces:  dist\CardioScope.exe  (single executable)
#  Run with:  Right-click → "Run with PowerShell"
#             Or in PowerShell:  .\build_windows.ps1
# ─────────────────────────────────────────────────────────────────
$ErrorActionPreference = "Stop"

$VENV_DIR   = ".venv_build"
$APP_NAME   = "CardioScope"

Write-Host ""
Write-Host "╔══════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║   CardioScope — Windows Build Pipeline   ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# ── 1. Check Python ───────────────────────────────────────
$PythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $PythonCmd) {
    Write-Host "❌  Python not found!" -ForegroundColor Red
    Write-Host "    Download from: https://www.python.org/downloads/" -ForegroundColor Yellow
    Write-Host "    Ensure 'Add Python to PATH' is checked during install." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}
$PY_VER = & python --version
Write-Host "✔  $PY_VER" -ForegroundColor Green

# ── 2. Create virtual environment ────────────────────────
Write-Host ""
Write-Host "→  Creating virtual environment..."
python -m venv $VENV_DIR
& "$VENV_DIR\Scripts\Activate.ps1"

# ── 3. Upgrade pip ───────────────────────────────────────
Write-Host "→  Upgrading pip..."
python -m pip install --upgrade pip --quiet

# ── 4. Install dependencies ───────────────────────────────
Write-Host "→  Installing dependencies..."
Write-Host "   • numpy scipy flask sounddevice"
pip install numpy scipy flask sounddevice matplotlib soundfile --quiet

Write-Host "   • PyTorch (CPU only)"
pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet

Write-Host "   • PyInstaller"
pip install pyinstaller --quiet

# ── 5. Build ──────────────────────────────────────────────
Write-Host ""
Write-Host "→  Building executable..."
pyinstaller cardioscope.spec --clean --noconfirm

# ── 6. Done ───────────────────────────────────────────────
Write-Host ""
Write-Host "╔══════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║              BUILD COMPLETE               ║" -ForegroundColor Green
Write-Host "╠══════════════════════════════════════════╣" -ForegroundColor Green
Write-Host "║  Output:  dist\$APP_NAME.exe"              -ForegroundColor Green
Write-Host "║"                                            -ForegroundColor Green
Write-Host "║  Run:     dist\CardioScope.exe"            -ForegroundColor Green
Write-Host "╚══════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Read-Host "Press Enter to exit"
