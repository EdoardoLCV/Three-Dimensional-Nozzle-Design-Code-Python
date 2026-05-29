# Using the graphical interface

This tool has a point-and-click graphical interface for all three programs
(2D MOC, Streamline Tracing, and 3D MOC). There are two ways to get it,
depending on whether you just want to *use* it or you are setting up the
GitHub repository.

---

## For users: just download and run (no Python, no compiling)

1. Go to the **Releases** page of this repository (right-hand side of the
   GitHub front page, or add `/releases` to the repo URL).
2. Download the file for your operating system:
   * Windows → `pynozzle-gui-windows.exe`
   * macOS → `pynozzle-gui-macos`
   * Linux → `pynozzle-gui-linux`
3. Also download `pynozzle-examples.zip` and unzip it somewhere — it holds
   ready-to-run example cases.
4. Double-click the downloaded program. A window opens with three tabs.
   Pick an input file (start with one from the `examples` folder), choose
   an output folder, and click **Run**.

That is the whole process. The program is self-contained; nothing else
needs to be installed.

> First-launch notes: the binary takes a few extra seconds to start the
> first time (it unpacks itself). On macOS, the first time you may need to
> right-click the file and choose **Open** to get past Gatekeeper. On
> Windows, SmartScreen may warn about an unsigned app — choose **More
> info → Run anyway**.

---

## For developers: run it from source

If you have Python 3.9+ installed:

```bash
cd python
pip install -e .          # installs numpy + scipy
python pynozzle_gui.py    # opens the graphical interface
```

If you get `No module named 'tkinter'`, install Tk (it ships with most
Python builds but not all):

* Ubuntu/Debian: `sudo apt install python3-tk`
* conda: `conda install tk`
* Windows/macOS python.org installers already include it.

You can also still use the command line directly:

```bash
pynozzle-moc2d examples/M3.5Perf.inp -o out
pynozzle-stt   examples/stt/M3.5Perf.inp -i examples/stt -o stt_out
pynozzle-moc3d examples/moc3d/cone10.geo --z-step 1 -o moc3d_out
```

---

## For the maintainer: how the downloadable apps get built

You never compile the apps by hand. A GitHub Actions workflow does it for
all three operating systems automatically. To publish a new set of
downloads:

```bash
git tag v1.0.0
git push origin v1.0.0
```

GitHub then builds `pynozzle-gui-windows.exe`, `pynozzle-gui-macos`,
`pynozzle-gui-linux` and `pynozzle-examples.zip` and attaches them to a
Release named after the tag. Progress is visible under the **Actions**
tab; the finished downloads appear under **Releases**.

The workflow lives in `.github/workflows/release.yml`. If your Python
package is not in a `python/` subfolder, edit the `WORKDIR:` line near the
top of that file to point at the folder containing `pyproject.toml`.
