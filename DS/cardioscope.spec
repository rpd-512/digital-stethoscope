# cardiosope.spec
# PyInstaller spec for CardioScope — Digital Stethoscope
# Build with: pyinstaller cardioscope.spec

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# ── Collect Flask templates + static ──────────────────────
added_files = [
    ('templates', 'templates'),
    ('static',    'static'),
]

# Include model weights if present
if os.path.exists('model_006.pt'):
    added_files.append(('model_006.pt', '.'))

# ── Hidden imports (Flask, sounddevice, scipy internals) ──
hiddenimports = [
    'flask',
    'flask.templating',
    'jinja2',
    'jinja2.ext',
    'werkzeug',
    'werkzeug.serving',
    'werkzeug.debug',
    'scipy.signal',
    'scipy.signal._upfirdn_apply',
    'scipy._lib.messagestream',
    'sounddevice',
    'numpy',
    'numpy.core._multiarray_umath',
    'numpy.core._methods',
    'cffi',
    '_cffi_backend',
]

# Add torch if available
try:
    import torch
    hiddenimports += collect_submodules('torch')
except ImportError:
    pass

a = Analysis(
    ['app.py'],
    pathex=['.'],
    binaries=[],
    datas=added_files,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'tkinter', '_tkinter', 'PyQt5', 'wx'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# ── One-file executable ────────────────────────────────────
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='CardioScope',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,             # no terminal window on Windows
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,                 # add icon path here if you have one: 'assets/icon.ico'
)
