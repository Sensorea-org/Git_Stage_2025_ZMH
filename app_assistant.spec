# app_assistant.spec  — PyInstaller (onedir)
# -*- mode: python ; coding: utf-8 -*-

import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# --- Paquets à imports/data dynamiques (transformers, sklearn, etc.) ---
_dyn_pkgs = [
    "sentence_transformers",
    "transformers",
    "tokenizers",
    "sklearn",
    "streamlit_authenticator",
    "graphviz",
    "streamlit_echarts",
    "streamlit",
]
hiddenimports = []
datas = []
hiddenimports += collect_submodules("streamlit")
datas        += collect_data_files("streamlit")
for pkg in _dyn_pkgs:
    try:
        hiddenimports += collect_submodules(pkg)
    except Exception:
        pass
    try:
        datas += collect_data_files(pkg)
    except Exception:
        pass
# --- Tes fichiers/projets à embarquer ---
# NB: 'dest' est un dossier DANS le bundle (relatif à l'exe)
datas += [
    (".streamlit\config.toml", ".streamlit"),          # st.config
    (".streamlit\secrets.toml", ".streamlit"),         # st.secrets
    ("assistant_0.2.py", "."),              # lancé par app_assistant
    ("sensorea_ia_lib.py", "."),            # lib locale
    ("datarecup_ beta.py", "."),            # script externe (nom avec espace => gardé en data)
]


# Dossiers (si présents) — adapte au besoin
if os.path.isdir("data"):
    datas.append(("data", "data"))

# Exemple de modèles à inclure si utilisés :
# if os.path.isfile("modele_multioutput_classification.pkl"):
#     datas.append(("modele_multioutput_classification.pkl", "."))
# if os.path.isfile("modele_multioutput_regression.pkl"):
#     datas.append(("modele_multioutput_regression.pkl", "."))

a = Analysis(
    ["app_assistant.py"],   # point d'entrée
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],       # tu peux ajouter un runtime hook si besoin
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="app/assistant_runner",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    
    console=True,           # False si tu ne veux pas la console
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="assistant_runner",
)
