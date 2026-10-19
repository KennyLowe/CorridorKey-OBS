# Installing the CorridorKey OBS filter (v0.1.0-rc.1, owner test build)

This is for a **clean Windows 11 machine with an NVIDIA RTX GPU** (Ampere or newer; tested on an
RTX 3060, target RTX A2000). Nothing from the development machine is assumed. v1 is an owner-only
build (no installer): you install a few prerequisites, clone this repository for its helper, and
copy the plugin into OBS's plugin folder.

How it fits together:
- The **plugin** (`corridorkey-obs.dll`) runs inside OBS. It keys the source on the GPU with
  TensorRT, on its own thread, and falls back to a colour key if anything goes wrong.
- The **helper** (`ckbench`, in this repository's `bench/`) runs outside OBS. It downloads the
  official CorridorKey weights and builds the TensorRT engine for *your* GPU, which takes minutes and
  gigabytes of RAM. The plugin starts it for you and shows its progress.

## 1. Prerequisites (direct downloads, no winget or choco)

The exact, hash-checked download commands are in
[issue #2](https://github.com/KennyLowe/CorridorKey-OBS/issues/2). Run PowerShell as
Administrator.

| What | Version | Where it goes |
|---|---|---|
| NVIDIA driver | **≥ 576** (597.16 known good on the RTX A2000) | `nvidia-smi` shows it |
| OBS Studio | **32.2.2** (installer from the OBS GitHub release) | default location |
| TensorRT | **10.16.1.11 for CUDA 12.9** | `C:\TensorRT-10.16.1.11` |
| Git | 2.55.0.5 (installer) | default location |
| uv | 0.13.1 (zip) | `C:\tools\uv`, on your user PATH |
| Python | 3.12, fetched by uv automatically | n/a |
| ffmpeg | 7.1.1 full build (zip; only for the benchmark) | `C:\tools\ffmpeg-7.1.1-full_build\bin`, on PATH |

TensorRT, if you haven't installed it yet:

```powershell
$zip = "$env:TEMP\TensorRT-10.16.1.11.zip"
curl.exe -L -o $zip https://developer.download.nvidia.com/compute/machine-learning/tensorrt/10.16.1/zip/TensorRT-10.16.1.11.Windows.amd64.cuda-12.9.zip
(Get-FileHash $zip -Algorithm SHA256).Hash   # must be 80BBE8E07A4FCCDE33AE7D210FFC0E6D1AF96DAE5FFAAEED60209F775FBF4275
Expand-Archive $zip -DestinationPath C:\
Test-Path C:\TensorRT-10.16.1.11\bin\nvinfer_10.dll   # True
```

You don't need the CUDA toolkit or Visual Studio: the plugin links the CUDA runtime statically and
loads only `nvinfer_10.dll` from the TensorRT folder. Don't install CUDA 13 or TensorRT 11; they
need a driver ≥ 580 and the plugin uses neither.

Check the driver and record it: `nvidia-smi --query-gpu=name,driver_version --format=csv`.

## 2. The helper: clone the repository and set it up

```powershell
git clone https://github.com/KennyLowe/CorridorKey-OBS C:\git\corridorkey
cd C:\git\corridorkey
git fetch --tags
git checkout v0.1.0-rc.1
cd bench
.\setup.ps1
```

- Clone to **`C:\git\corridorkey`**. The plugin's default helper command is
  `uv run --project "C:/git/corridorkey/bench" ckbench`. If you clone elsewhere, change
  *Advanced → Helper command* in the filter.
- `setup.ps1` clones CorridorKey at the pinned commit (the model code the helper needs to build
  engines), downloads the official weights for the benchmark (sha256-checked against
  `models/manifest.json`), and builds the Python environment (`uv sync`). It ends with
  `torch … NVIDIA …` and an NVENC check.
- Open a **new** PowerShell afterwards so `uv` is on PATH, and **restart OBS** if it was open, so it
  sees the new PATH too.

## 3. Install the plugin

1. Download `corridorkey-obs-v0.1.0-rc.1-windows-x64.zip` from the
   [release](https://github.com/KennyLowe/CorridorKey-OBS/releases/tag/v0.1.0-rc.1) and check it:
   `(Get-FileHash .\corridorkey-obs-v0.1.0-rc.1-windows-x64.zip -Algorithm SHA256).Hash` must equal
   the value in the `.sha256` file next to it.
2. Unzip it, **close OBS**, and run, as Administrator:

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\install.ps1
   ```

   It copies `corridorkey-obs\` to `C:\ProgramData\obs-studio\plugins\corridorkey-obs\`, with
   `bin\64bit\corridorkey-obs.dll` and `data\` (effects, the Auto table `perf-table.json`, the model
   manifest). It refuses to run while OBS is open.

The zip has no weights, ONNX files, engines, TensorRT or CUDA DLLs, and no footage.

## 4. First run in OBS

1. Start OBS 32.2.2. **Help → Log Files → View Current Log** should show
   `[corridorkey] loaded (version 0.1.0)`.
2. On your camera (or any video source): **Filters → + → CorridorKey**. The source passes through
   unkeyed and the Status line says *Weights not downloaded*.
3. Under **Model**, tick *I accept the CorridorKey licence* (CC BY-NC-SA 4.0 plus Corridor
   Digital's terms; *Read the CorridorKey licence* opens it). This is remembered on this machine.
4. Press **Download weights** (about 400 MB from Corridor Digital's official Hugging Face
   repository). Press *Refresh status* to see the progress. The file is checked against the pinned
   sha256 and saved in `%LOCALAPPDATA%\corridorkey-obs\weights\`.
   - Already have the file? Use **Use a local file…** instead. `setup.ps1` put one in
     `C:\git\corridorkey\bench\vendor\CorridorKey\CorridorKeyModule\checkpoints\`. A file with
     the pinned hash shows as *verified*; any other shows as *unverified model* and is still allowed.
5. Preparation starts by itself: *Preparing 1024×576 — building engine NN%*. That takes 2–5 minutes
   per quality setting, once per GPU and driver. OBS keeps running and the source passes through
   or shows the colour key meanwhile. Then the key appears and Status shows
   *Keying — 1024x576, 26.7 ms, 0.0% skipped, +33 ms (1.0 frames)*.
6. On later OBS starts the key is back within seconds (measured 4.6 s), without re-preparing.
   A driver, GPU, TensorRT or weights change is detected and the engine is re-prepared.

**Quality.** *Auto* picks the best setting measured for your GPU and the source's mode (for
example, 1080p30 on an RTX 3060 → 1024×576). It never picks an unprepared setting: it runs the best
prepared one and offers the better one; *Prepare selected* builds it. Until the A2000 benchmark
results are in `perf-table.json`, Auto treats the A2000 as an unmeasured GPU and uses the fastest
setting (512×512) with a warning. Pick a setting by hand to override it.

**Screen colour.** The default (*Fixed green*) is the hint the spike used. If the key is poor (for
example on dim or unevenly lit screens), choose *Auto-calibrate* or press **Calibrate now**. It
samples the frame border over 30 frames. *Show hint* displays the rough mask for checking.

**Fallback.** If keying can't keep up, fails or isn't ready, the filter shows a colour-key matte
(default), the last good matte, or the unkeyed source after 250 ms. OBS keeps rendering in every
case.

## 5. Audio sync offset

The key adds a fixed delay to the video: the Status line shows it, for example *+33 ms (1.0
frames)* at 1080p30. (On the RTX 3060 it was measured on OBS's recorded output: exactly 1 frame.)
Delay your microphone by the same amount: **Edit → Advanced Audio Properties → Sync Offset** for
the mic, the shown ms (e.g. 33). Re-check it if you change the quality setting, frame rate or GPU.

## 6. Troubleshooting

- **Logs.** OBS log lines start with `[corridorkey]` (Help → Log Files): every state change, every
  error with its cause, and a 60-second performance summary. The helper's own output is in
  `%LOCALAPPDATA%\corridorkey-obs\logs\`.
- **Unsupported: nvinfer_10.dll not found in …**: the TensorRT folder is wrong (*Advanced →
  TensorRT folder*, default `C:\TensorRT-10.16.1.11\bin`). Changing it takes effect after an OBS
  restart if another folder was already loaded.
- **Unsupported: OBS renders on "…", which is not a CUDA (NVIDIA) GPU**: OBS is running on another
  adapter. Set OBS (and Windows' graphics settings for `obs64.exe`) to the NVIDIA GPU.
- **Cannot start the helper**: `uv` isn't on OBS's PATH (restart OBS after installing uv) or the
  repository isn't at `C:\git\corridorkey` (fix *Advanced → Helper command*).
- **Corporate network:**
  - `CRYPT_E_NO_REVOCATION_CHECK` / `CRYPT_E_REVOCATION_OFFLINE` (curl exit 35, or git
    "schannel"): the network blocks the certificate revocation server. `setup.ps1` retries once
    with revocation checking off for that one command (`curl --ssl-no-revoke`,
    `git -c http.schannelCheckRevoke=false`) and still verifies the sha256.
  - `uv`: `invalid peer certificate: UnknownIssuer` (HTTPS inspection): `setup.ps1` retries
    `uv sync` once with `UV_SYSTEM_CERTS=1` (current uv) and `UV_NATIVE_TLS=1` (older uv). For other
    uv commands, set both yourself first: `$env:UV_SYSTEM_CERTS="1"; $env:UV_NATIVE_TLS="1"`.
  - The in-OBS weights download uses Python's TLS (the Windows certificate store, no revocation
    check), so it isn't affected by the schannel issue. If it still can't get through, download
    `CorridorKey_v1.0.safetensors` elsewhere from
    <https://huggingface.co/nikopueringer/CorridorKey_v1.0> and use **Use a local file…**.
- **git refuses the checkout ("dubious ownership")**: `setup.ps1` / `ckbench` print the exact fix,
  e.g. `git config --global --add safe.directory C:/git/corridorkey/bench/vendor/CorridorKey`.
  Run it yourself; nothing changes your global git config automatically.

## 7. Uninstall

Close OBS and delete `C:\ProgramData\obs-studio\plugins\corridorkey-obs\`. Optionally also delete
`%LOCALAPPDATA%\corridorkey-obs\` (weights, engines, helper logs).

## Licensing

- The plugin (`corridorkey-obs/`, `LICENSE` in the zip) is GPL-2.0-or-later. It contains no
  CorridorKey code or weights.
- The helper (`bench/`), docs and results are CC BY-NC-SA 4.0.
- The CorridorKey weights are downloaded by you from Corridor Digital's official source under
  CC BY-NC-SA 4.0 plus Corridor Digital's terms: non-commercial, with attribution.
- Powered by CorridorKey by Corridor Digital. Not affiliated with or endorsed by Corridor Digital.
