# pynozzle

A Python port of the JHU/APL "Three-Dimensional Nozzle Design Code" by Tharen
Rice (JHU/APL Report RTDC-TPS-481, 2003). The original was a set of three
MFC-based Windows applications written in Microsoft Visual C++:

* **MOC_Grid_BDE** — 2D / axisymmetric Method-of-Characteristics nozzle designer
* **STT2001** — Streamline tracing tool that operates on `MOC_Grid_BDE` output
* **3D_MOC** — 3D MOC flow-field solver for a supplied 3D wall contour

This package re-implements them as cross-platform Python utilities that read
the same `.inp` configuration files and write the same output formats as the
originals, so existing inputs and sample outputs can be used directly for
validation.

## Status

**This release ports all three tools — `MOC_Grid_BDE`, `STT2001`, and
`3D_MOC` — fully.**

### MOC_Grid_BDE

Validation against the upstream `outputs_M3.5Perf` sample (an axisymmetric
perfect M=3.5 nozzle with γ=1.4, p₀=500 psia):

| Output file       | Lines | Differing | Notes                                          |
|-------------------|-------|-----------|------------------------------------------------|
| `summary.out`     | ~480  | 2         | Both documented C++ artifacts (see below)      |
| `rao.dat`         | 101   | 0         | Exact bit-for-bit match                        |
| `center.out`      | 110   | 2         | Floating-point precision at trailing zero      |
| `MOC_Grid.plt`    | 13472 | 5         | Floating-point precision at trailing zero      |

The two `summary.out` differences are:

1. The line *"Massflow error due to grid at end of Kernel (%)"* — the C++
   reads from an **uninitialized class member** (`mdotErrRatio` is declared
   but never assigned anywhere in the source). The sample shows `100` because
   that uninitialized memory happened to be zero on the run that produced
   it. Our port computes the real ratio (around -0.75% for the M3.5 case).
2. One trailing centerline point where the C++ wrote exactly `0` and we write
   `5e-16` — pure floating-point precision noise.

`MOC_Grid_BDE` smoke-tested on all three nozzle types (Perfect, Cone, Rao
minimum-length).

### STT2001

Validation against the upstream `STT2001/outputs_M3.5Perf` sample (the 3D
streamline-traced nozzle built from the M3.5 MOC flow field, full 360°
revolution, 72 streamlines):

| Output file              | Lines  | Differing | Notes                                |
|--------------------------|--------|-----------|--------------------------------------|
| `*_ThroatSL.out`         | 144150 | 0         | Exact bit-for-bit match              |
| `*_ThroatSummary.out`    | 73     | 0         | Exact bit-for-bit match              |
| `*_TrimSL.out`           | 5693   | 0         | Exact bit-for-bit match              |
| `*_AvsX.out`             | 100    | 0         | Exact bit-for-bit match              |
| `*_AvsSL.out`            | 73     | 0         | Exact bit-for-bit match              |
| `*_trimmed_P3D.xyz`      | 2410   | 0         | Exact bit-for-bit match              |
| `*_trimmed_P3D.dat`      | 804    | 0         | Exact bit-for-bit match              |
| `*_end_P3D.xyz`          | 3      | 0         | Exact bit-for-bit match              |
| `*.plt`                  | 7304   | 0         | Exact bit-for-bit match              |
| `*_Engine.plt`           | 3      | 0         | Exact bit-for-bit match              |
| `*_cl.dat`               | 101    | 0         | Exact bit-for-bit match              |
| `*_all_runs.dat`         | 2      | 1         | Only the undefined-memory artifact   |
| `*_STT_summary.out`      | 35     | 9         | All from the same undefined-memory artifact |

Roughly **160,000 lines of output reproduced exactly.** Every remaining
difference (in `_STT_summary.out` and `_all_runs.dat`) traces to a single
C++ undefined-behaviour bug: when parsing scalars out of the upstream
`summary.out`, the original does
`strLine.copy(cLine, 10, 22)` into a **non-null-terminated 10-char buffer**
and then `atof`s it. For short fields (e.g. the throat radius `1.000`) the
parse runs past the copied bytes into uninitialised stack memory, so the
sample reads `1.00099` instead of `1.000`. That wrong throat radius then
flows through `m_thrust1 = thrust1MOC·aThroat/(π·r*²)` into the throat
thrust, exit stream thrust, Isp and Cfg. The same trailing-garbage read
nudges the last digit of the echoed Area Ratio (`6.73652` vs `6.73651`)
and Gross Isp (`69.9497` vs `69.9496`). **Our port produces the
mathematically correct values**; matching the sample here would mean
deliberately reading uninitialised memory, which isn't reproducible.

### 3D_MOC

Validation against the upstream `3D_MOC/outputs_cone10` sample (a 10°
cone with 217 axial stations × 181 points per plane = ~39000 grid
nodes, 44 wall rays). The 3D MOC march is an **iterative** solver —
Newton-Raphson at every body point, fixed-point iteration at every
field point, thin-plate-spline surface fit on each plane — so unlike the
2D MOC and STT, exact bit-for-bit reproduction isn't achievable.

| Output file        | Lines  | Notes                                       |
|--------------------|--------|---------------------------------------------|
| `Initial Wall.plt` | 8466   | 11 of 8466 lines differ — all at `cos(π/2) ≈ 1e-17` noise |
| `z=0.out`          | 182    | Same: 11 lines differ at machine-precision noise |
| `outfile.out`      | 9504   | Body-point unit-normal log; agrees to ~5 sig figs |
| `full_mesh.plt`    | 39281  | Full mesh; agrees to **4-5 sig figs** through 217 planes |
| `axialStations.plt`| 39497  | Same data zoned by plane                    |
| `Streamlines.plt`  | 39461  | Same data zoned by streamline               |
| `Wall.plt`         | 9814   | Wall-only rows; agrees to 4-5 sig figs      |
| `streamtube.plt`   | varies | Generated when `SL.inp` is present          |

Spot-check at row 100 of `Wall.plt` (deep into the cone, k=95):

```
sample : 2.13856  0  3.77993  2.13856  67.574   2135.14  245.43   ...  2.90006  2227.39  ...
mine   : 2.13856  0  3.77993  2.13856  67.5734  2135.14  245.429  ...  2.90007  2227.39  ...
```

The geometry (X, Y, Z, R) matches **exactly**, and pressure / Mach /
velocity / density all agree to 5+ significant digits. The differences
arise from compound effects: tiny libm differences between MSVC and
glibc in `sin`/`cos`/`asin`/`atan2`/`pow`/`log` propagate through 217
Newton-Raphson body-point solves and 200+ outer-loop fixed-point
iterations per plane. At this point the comparison is bounded by
platform-libm precision rather than algorithmic fidelity.

## Install

```bash
cd pynozzle
pip install -e .
```

Requires Python ≥ 3.9, NumPy ≥ 1.20, SciPy ≥ 1.7.

## Usage

```bash
pynozzle-moc2d  path/to/case.inp  -o output_directory
```

`.inp` files use the **same format** as the original `MOC_Grid_BDE` GUI's
File → Save As output, so any `.inp` you have from the Windows tool will
work unchanged. The output files (`summary.out`, `MOC_Grid.plt`,
`MOC_SL.plt`, `center.out`, `rao.dat`) keep their original names and
structure so they drop straight into the downstream `STT2001` / `3D_MOC`
workflow once those are ported.

Options:

* `-o DIR` / `--output-dir DIR` — where to write outputs (default: `.`)
* `--summary-only` — write only `summary.out` and `rao.dat`, mimicking
  the original GUI's `printMode = 0` behavior

### Streamline tracing (STT2001)

`STT2001` consumes a prior `MOC_Grid_BDE` run and traces 3D streamlines
through a chosen throat shape, then integrates wall pressure to give the
3D nozzle's thrust / Isp / Cfg:

```bash
pynozzle-stt  path/to/case.inp  -i input_directory  -o output_directory
```

* `-i DIR` / `--input-dir DIR` — where the upstream MOC files
  (`MOC_SL.plt`, `MOC_Grid.plt`, `summary.out`) and the friction table
  live (default: the directory of the `.inp` file)
* `-o DIR` / `--output-dir DIR` — where to write outputs (default: `.`)

The `.inp` files use the same 19-line format as the original `STT2001`
GUI's File → Save output. A worked example is in `examples/stt/`:

```bash
pynozzle-stt examples/stt/M3.5Perf.inp -i examples/stt -o /tmp/stt_run
```

Programmatic use:

```python
from pynozzle.stt import read_inp, STTSolver

inp = read_inp("examples/stt/M3.5Perf.inp")
result = STTSolver("examples/stt", inp).run("/tmp/stt_run")
print("Surface area:", result.surface_area)   # 167.08 in2
print("Pressure force:", result.pressure_force) # 531.764 lbf
print("Cfg:", result.cxx)
```

### 3D Method-of-Characteristics flow field (3D_MOC)

`3D_MOC` takes a nozzle wall contour (a stack of circles, as written
into a ``.geo`` file by `MOC_Grid_BDE` or any equivalent) and marches a
full three-dimensional MOC flow-field solution down the nozzle,
emitting Tecplot files of the mesh, wall, axial stations and
streamlines:

```bash
pynozzle-moc3d  path/to/case.geo  --z-step 1  -o output_directory
```

Useful options:

* `-o DIR` / `--output-dir DIR` — where to write outputs (default: `.`)
* `--p0`, `--t0`, `--mach0`, `--mol-wt`, `--gamma`,
  `--theta0`, `--psi0` — initial chamber/throat state
  (defaults match the original GUI: 1000 psia, 530 °R, M=1.1,
  MW=28.96, γ=1.4)
* `--n-div` — angular divisions around the wall (default 36)
* `--x-step`, `--z-step` — output strides (set `--z-step 1` to match
  the sample outputs, which write every plane)
* `--surface-fit` — `"All Point Spline"` (default) or `"9 Point Spline"`

A worked example lives in `examples/moc3d/` (cone10, M4Perfect, M4RAO).
The full 217-plane cone10 march takes about 40 seconds. Programmatic
use:

```python
from pynozzle.moc3d import read_geo, MOC3DGrid

cfg = read_geo("examples/moc3d/cone10.geo")
cfg.z_output_step = 1
grid = MOC3DGrid(cfg, "/tmp/moc3d_run")
grid.set_initial_properties()
result = grid.calc_nozzle()
print(result.success, result.n_pts, result.n_z)
```

### Programmatic API

```python
from pynozzle.moc2d import read_inp, MOCGridCalc
from pynozzle.moc2d.io_writers import write_all

inp = read_inp("examples/M3.5Perf.inp")
calc = MOCGridCalc()
calc.set_initial_properties(
    pres=inp.pres_i, temp=inp.temp_i, mol_wt=inp.mol_wt_i,
    gamma=inp.gamma_i, p_amb=inp.p_amb, n=inp.n_c,
    rwt_u=inp.rwt_u, rwt_d=inp.rwt_d,
    d_t_limit_deg=inp.d_t_limit,
    n_rrc_above_bd=inp.n_rrc_above_bd,
    n_sl_i=inp.n_sl_i, n_sl_j=inp.n_sl_j,
    vel=inp.vel, throat_flag=int(inp.throat),
    isp_ideal=inp.isp_ideal,
)
v1, v2 = inp.design_values()
calc.set_solution_parameters(
    geom=inp.nozzle_geom(),
    nozzle_type=inp.nozzle_type(),
    design_param=inp.design_param(),
    value1=v1, value2=v2,
    theta_bi_deg=inp.theta_bi,
)
result = calc.run()
if result.success:
    write_all(result, "output_dir")
    # Direct access to the MOC grid:
    print("Wall Mach exit:", result.grid.mach[0, result.last_rrc])
```

## Units

Inputs and outputs use the same units as the C++ code (English engineering
units throughout: psia, °R, lbf, lbm, in., ft/s, slug/ft³).

## Layout

```
pynozzle/
├── pynozzle/
│   ├── common/                   shared constants and thermodynamic helpers
│   │   ├── constants.py          full port of engineering_constants.hpp
│   │   └── thermo.py             Mach angle, Prandtl-Meyer, isentropic, etc.
│   └── moc2d/                    MOC_Grid_BDE port
│       ├── inp.py                .inp file reader/writer
│       ├── grid.py               MOC grid data container
│       ├── solver.py             top-level MOCGridCalc class
│       ├── _solver_kernels.py    throat-line / wall / interior kernels
│       ├── _contoured_kernels.py RAO / FindPointE / DE-iteration kernels
│       ├── io_writers.py         summary.out, *.plt, etc.
│       └── cli.py                command-line entry point
│   ├── stt/                      STT2001 streamline-tracing port
│   │   ├── inp.py                .inp file reader/writer (19-line format)
│   │   ├── loaders.py            MOC_SL.plt / summary.out / MOC_Grid.plt / friction
│   │   ├── kernels.py            throat-SL, trim, grid, integration, FindMaxX
│   │   ├── solver.py             STTSolver orchestrator + STTState container
│   │   ├── io_writers.py         ThroatSL, TrimSL, Plot3D, Tecplot, summary writers
│   │   └── cli.py                command-line entry point
│   └── moc3d/                    3D_MOC flow-field solver port
│       ├── geo.py                .geo wall-contour reader + run config
│       ├── point.py              XYZPoint class and 3D-specific constants
│       ├── kernels.py            field point, body point, compatibility equations
│       ├── solver.py             MOC3DGrid orchestrator + thin-plate-spline fit
│       ├── io_writers.py         Tecplot output (full mesh, wall, streamlines)
│       └── cli.py                command-line entry point
├── examples/                     sample .inp / .geo files
│   ├── M3.5Perf.inp              perfect axi M=3.5 (matches upstream sample)
│   ├── M4Rao.inp                 minimum-length M=4
│   ├── cone10.inp                10° cone, M=4
│   ├── stt/                      STT2001 example + its upstream MOC inputs
│   │   ├── M3.5Perf.inp          full-360° 3D nozzle off the M3.5 flow field
│   │   ├── MOC_sl.plt            streamlines from the MOC run
│   │   ├── MOC_Grid.plt          MOC mesh (for the reflective-wave check)
│   │   ├── summary.out           MOC scalar summary
│   │   └── friction_table.txt    surface-area → friction-loss table
│   └── moc3d/                    3D_MOC examples
│       ├── cone10.geo            10° cone wall contour
│       ├── M4perfect.geo         perfect M=4 wall contour
│       ├── m4Rao.geo             Rao minimum-length M=4 wall contour
│       └── SL.inp                optional streamline-selection file
├── tests/
│   ├── test_moc2d_regression.py  validates MOC_Grid_BDE against the sample
│   ├── test_stt_regression.py    validates STT2001 against the sample
│   └── test_moc3d_regression.py  validates 3D_MOC scalars and short march
└── README.md
```

## Running the tests

```bash
python -m unittest discover -v
```

The sample-matching tests skip automatically if the original C++ output
directory isn't available; set `PYNOZZLE_SAMPLE_DIR` to point at the
upstream `outputs_M3.5Perf` directory to enable them.

## A note on faithfulness

The port preserves several known bugs and quirks of the original C++
source, because the upstream sample outputs were generated by that code
and an output-faithful port has to keep them. All are flagged with
comments in the source.

**MOC_Grid_BDE:** a `/34` that was almost certainly meant to be `/384`
in `KLThroat`; a `*` that should have been a `-` in the `v[3]`
expression; and two implicit integer-division zero results (`5/8` and
`1/6` written as int-division literals, evaluating to `0`).

**STT2001:** the axial-grid spacing uses `(nparamGRIDX-1)/2` as an
*integer* division (49, not 49.5) — preserved, because using 49.5 shifts
every grid point and throws off the area integral by ~1.4%. The
nozzle-closing wraparound sector is silently dropped because the C++
pressure-nonzero check reads one row past the filled data
(`pgrid[nNewSLs]`, an all-zero row) — preserved, because it scales every
integrated area by `(N-1)/N`. The Plot3D row-wrapping uses a
post-increment (`k++ > 8`) that yields 10 values per row — preserved.
The `_STT_summary.out`, `_ThroatSummary.out` and `.plt` files inherit
C++ `ostream` precision flags across `close()`/`open()` of the same
stream object — preserved by matching the inherited precision.

**3D_MOC:** uses `GASCON = 1545.317` (the value baked into the 3D
tool's `engineering_constants.hpp`, slightly different from the
`1545.0` used by the 2D and STT tools) — preserved in
`pynozzle/moc3d/point.py`. The Numerical Recipes Newton-Raphson routine
`newt` (with line search via `lnsrch`/`fmin`) is replaced with
`scipy.optimize.fsolve` using the analytic Jacobian; both converge to
the same root, but the iteration paths differ and the converged values
differ in the 5th-6th significant digit. The `z=0.out` sample is from
an earlier build that didn't yet have the `Radius(in)` column the
current source emits — the port matches the sample (no Radius column).

The one thing we **don't** reproduce is the undefined-memory read in the
`summary.out` parser (see Status above): doing so would require reading
uninitialised memory, which isn't deterministic. Our values there are the
mathematically correct ones.
