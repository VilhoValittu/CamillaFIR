# CamillaFIR Installation Guide

This guide contains the detailed installation and update instructions that were previously embedded in the main `README.md`.

## Recommended path

For most users, the recommended option is:

1. Download the latest packaged release
2. Extract it
3. Run CamillaFIR
4. Open `http://127.0.0.1:8080` if the browser does not open automatically

Latest release:
- https://github.com/VilhoValittu/CamillaFIR/releases/latest

All releases:
- https://github.com/VilhoValittu/CamillaFIR/releases

## Run from release package

### Windows

1. Download `CamillaFIR_<version>_windows.zip` from Releases.
2. Extract the ZIP.
3. Run `CamillaFIR.exe`.
4. If SmartScreen appears, choose `More info` -> `Run anyway`.
5. Open `http://127.0.0.1:8080` if the browser does not open automatically.

### Ubuntu / Debian Linux

1. Download `CamillaFIR_<version>_linux.tar.gz` from Releases.
2. Extract the archive.
3. Open Terminal in the extracted folder and run:

```bash
./run.sh
```

4. Open `http://127.0.0.1:8080` if the browser does not open automatically.

### macOS (Intel + Apple Silicon)

1. Download `CamillaFIR_<version>_macos.tar.gz` from Releases.
2. Extract the archive.
3. Open Terminal in the extracted folder and run:

```bash
chmod +x CamillaFIR
./CamillaFIR
```

4. If macOS blocks first launch, open `System Settings -> Privacy & Security -> Open Anyway`.
5. Open `http://127.0.0.1:8080` if the browser does not open automatically.

## Output path

Output ZIP files are saved by default to:

```text
Documents/CamillaFIR/filters/<version>/
```

If that path is not writable, CamillaFIR falls back to a safe writable directory and reports the final path in Results.

## Browser and PNG notes

- CamillaFIR UI is browser-based.
- Interactive graphs can be saved from the graph download button in the UI.
- ZIP export is focused on filter artifacts and summary data.
- Dashboard image inclusion can be disabled in performance mode.

## Known issue: Windows + Vivaldi

In some Windows setups, using Vivaldi can trigger NumPy `MemoryError` under browser memory pressure.

Workarounds:

- use Chrome, Edge, or Firefox
- close extra Vivaldi tabs or extensions
- re-run the process in another browser if needed

