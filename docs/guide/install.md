# Installing hxex

*hxex* runs on Windows, macOS, and Linux. You do not need to know Python or install it yourself:
the steps below install everything *hxex* needs, including its own copy of Python, in a place of
its own, without changing anything else on your computer.

You need an internet connection and about 2 GB of free disk space (more on Linux, see step 3).
Installing takes a few minutes, mostly downloading.

## Step 1: open a terminal

*hxex* is run by typing commands in a *terminal*, a window where you type a command, press
Enter, and the computer answers with text.

- **Windows**: open the Start menu, type `PowerShell`, and click **Windows PowerShell**.
- **macOS**: press Cmd + Space, type `Terminal`, and press Enter.
- **Linux**: open the **Terminal** application (often Ctrl + Alt + T).

In the steps below, copy each command, paste it into the terminal, and press Enter. (To paste in
PowerShell, right-click; on macOS, press Cmd + V; in a Linux terminal, Ctrl + Shift + V.)

## Step 2: install uv

[uv](https://docs.astral.sh/uv/) is a free installer for Python programs. It downloads *hxex*
and everything it needs.

**Windows** (in PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS and Linux**:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

When it has finished, **close the terminal and open a new one**, so that it finds the new
program.

## Step 3: install hxex

**Windows and macOS**:

```bash
uv tool install https://github.com/tuttlelm/hxex/archive/refs/heads/main.zip
```

**Linux**: the command above works too, but it downloads a version of PyTorch (the library
*hxex* fits with) for NVIDIA graphics cards, which is several gigabytes. If you will not use a
graphics card, this smaller version is enough:

```bash
uv tool install https://github.com/tuttlelm/hxex/archive/refs/heads/main.zip --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
```

The terminal lists what it installs and ends with `Installed 1 executable: hxex`.

## Step 4: make the `hxex` command available

```bash
uv tool update-shell
```

Close the terminal and open a new one, then check that *hxex* answers:

```bash
hxex --version
```

It prints `hxex` and its version number. If it says the command is not found, close and reopen
the terminal once more; if that does not help, run step 4 again.

## Updating hxex

To install the latest version, run the same command you used in step 3, with
`--reinstall --refresh` added at the end.

If you installed with the first command of step 3 (Windows, macOS, or Linux):

```bash
uv tool install https://github.com/tuttlelm/hxex/archive/refs/heads/main.zip --reinstall --refresh
```

If you installed with the smaller Linux command:

```bash
uv tool install https://github.com/tuttlelm/hxex/archive/refs/heads/main.zip --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match --reinstall --refresh
```

If you installed for an NVIDIA graphics card on Windows (see
[Optional extras](#optional-extras)):

```powershell
uv tool install https://github.com/tuttlelm/hxex/archive/refs/heads/main.zip --index https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match --reinstall --refresh
```

`hxex --version` shows the version you have. *hxex* is still in development: note the version
you used for each analysis.

## Removing hxex

```bash
uv tool uninstall hxex
```

## Optional extras

**An NVIDIA graphics card** makes fitting faster by utilizing the GPU in addition to the CPU. On
Linux, the first command of step 3 already installs a PyTorch that can use one. On Windows it
does not; there, use this command for step 3 instead (it suits recent cards and drivers; whoever
looks after your computer can tell you which you have):

```powershell
uv tool install https://github.com/tuttlelm/hxex/archive/refs/heads/main.zip --index https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match
```

*hxex* uses the GPU when you run `hxex run hxex.toml --device cuda`, or set `device = "cuda"`
under `[run.fit.backend]` in the project file. GPU runs on macOS are not currently supported.

**Thermo `.raw` files** can be read directly, but this needs extra software from Thermo and
Microsoft (see [Input formats](inputs.md#thermo-raw-files)). The alternative is to convert the
files to mzML first, with the free
[ThermoRawFileParser](https://github.com/compomics/ThermoRawFileParser), and read the mzML
files. Converted files lose the resolving power the instrument recorded for each scan. That
matters only with `pick = "window_max"`, where the window can follow it; with the default
`pick = "peak"` the window is the ppm tolerance either way (see [The picking
window](picking-window.md)).

Next: [Your first project](first-project.md).
