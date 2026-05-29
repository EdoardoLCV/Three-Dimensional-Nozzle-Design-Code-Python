# Three-Dimensional Nozzle Design Code — Python port (with GUI)

A cross-platform **Python port** of the JHU/APL *Three-Dimensional Nozzle
Design Code* (originally three Microsoft Visual C++ / MFC Windows
applications), packaged with a **graphical interface** and downloadable,
double-click applications for Windows, macOS, and Linux.

The original suite is three tools:

* **MOC_Grid_BDE** — 2D / axisymmetric Method-of-Characteristics nozzle designer
* **STT2001** — streamline-tracing tool that turns a 2D MOC flow field into a 3D nozzle
* **3D_MOC** — 3D Method-of-Characteristics flow-field solver for a supplied wall contour

This port re-implements all three as Python that reads the **same `.inp`
input files** and writes the **same output files** as the originals, so
existing cases and sample outputs work unchanged. On top of the solvers it
adds a single graphical app that recreates the original dialogs and draws
the results (nozzle contour, characteristic mesh, Mach field, traced
streamlines).

---

## Download and run (no Python, no setup)

1. Go to the **[Releases](../../releases)** page.
2. Download the file for your operating system:
   * **Windows** → `pynozzle-gui-windows.exe`
   * **macOS** → `pynozzle-gui-macos`
   * **Linux** → `pynozzle-gui-linux`
3. Download **`pynozzle-examples.zip`** and unzip it — it holds ready-to-run
   example cases.
4. Double-click the program. A window opens with three tabs; pick an input
   file from the `examples` folder, choose an output folder, and click **Run**.

That is the whole process — the solver is bundled inside the one file you
download.

> **First-launch notes.** The app takes a few extra seconds to start the
> first time (it unpacks itself). On **macOS**, right-click the file →
> **Open** the first time to get past Gatekeeper. On **Windows**, if
> SmartScreen warns about an unrecognized app, choose **More info → Run
> anyway**. These are normal for unsigned applications.

---

## Using the graphical interface

The app has one tab per tool. Every tab lets you **type the inputs by hand**
(like the original dialogs) and also **Load / Save** the matching `.inp`
file, so cases round-trip with the Windows tool.

* **2D MOC** — set Nozzle Geometry, Nozzle Type, Design Parameter, Throat
  Geometry, Flow Properties, output streamlines, print option and MOC
  limiters, then **Calculate MOC Grid**. The plot panel offers three views:
  * *Nozzle Contour* — the wall, mirrored about the axis
  * *Characteristic Mesh* — the full left/right-running characteristic net
  * *Mach Field* — a filled Mach contour, clipped to the nozzle wall
* **Streamline Trace** — set the file inputs, other inputs, the flow-field
  centre sweep and the five trimming/throat slots, then **Execute**. Outputs
  (surface area, thrust, Isp, Cfg…) fill in, and the plot shows the *Traced
  Nozzle Contour* (X vs R) or the *Traced Streamlines* in 3D.
* **3D MOC** — choose a `.geo` wall contour, set the initial-plane
  properties, grid setup, print strides and surface fit, then **Calculate
  Nozzle**. The plot shows the input geometry, the marched *Wall Mach*
  field, or the *3D wall surface* coloured by Mach.

---

## Run from source (for developers)

If you have Python 3.9+:

```bash
cd python
pip install -e .            # installs numpy + scipy
pip install matplotlib      # needed for the GUI plots
python pynozzle_gui.py      # opens the graphical interface
```

If you see `No module named 'tkinter'`, install Tk:
* Ubuntu/Debian: `sudo apt install python3-tk`
* conda: `conda install tk`
* Windows / macOS python.org installers already include it.

### Command line

The three tools are also available as commands after `pip install -e .`:

```bash
pynozzle-moc2d  examples/M3.5Perf.inp                       -o out_moc2d
pynozzle-stt    examples/stt/M3.5Perf.inp -i examples/stt   -o out_stt
pynozzle-moc3d  examples/moc3d/cone10.geo --z-step 1        -o out_moc3d
```

The `.inp` files use the same format as the original GUIs' File → Save As
output, and the result files keep their original names so they drop straight
into the downstream workflow.

---

## Validation

The port is validated against the original distribution's sample outputs.

* **2D MOC (MOC_Grid_BDE).** `rao.dat` is bit-for-bit identical; `summary.out`
  matches except two documented artifacts; the streamline data file is
  numerically identical. One `summary.out` line (the *"Massflow error"*
  figure) differs because the C++ reads an **uninitialized variable** — the
  port computes the real value.
* **Streamline Trace (STT2001).** About **160,000 lines of output reproduced
  exactly.** The only ten differing lines all trace to a single C++
  undefined-behaviour bug: a scalar is parsed out of `summary.out` through a
  non-null-terminated 10-char buffer, so the original reads a throat radius
  of `1.00099` instead of `1.000`, and that error flows into thrust, Isp and
  Cfg. The port produces the mathematically correct values.
* **3D MOC (3D_MOC).** The 3D march is an iterative solver (Newton-Raphson at
  every body point, fixed-point iteration per field point, a spline surface
  fit per plane), so exact bit-for-bit output isn't achievable. Validated
  against the 10° cone sample: the radial/axial geometry and total
  conditions match exactly, and the flow field (pressure, temperature,
  density, Mach, velocity) agrees to about **3–4 significant figures**.
  Because the only sample case is axisymmetric, the solver's fully
  three-dimensional behaviour is exercised but not independently validated.

Run the test suite from `python/`:

```bash
python -m unittest discover -v
```

---

## Credit and license

This is a Python port. The original *Three-Dimensional Nozzle Design Code*
was written by **Tharen Rice (JHU/APL)** — JHU/APL Report RTDC-TPS-481,
2003 — and is distributed publicly (NASA NTRS). The original license
(`license.txt`) is retained and applies to this port. This repository adds a
Python re-implementation, a graphical interface, and automated cross-platform
builds; it claims no ownership of the original algorithms.
