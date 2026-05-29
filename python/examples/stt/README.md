# STT2001 example — M3.5Perf

This is the streamline-tracing example that pairs with the `M3.5Perf`
MOC nozzle (an axisymmetric perfect M=3.5 design). It builds a full-360°
3D nozzle from that flow field using 72 streamlines.

## Files

| File                | Role                                                           |
|---------------------|----------------------------------------------------------------|
| `M3.5Perf.inp`      | STT2001 input deck (the thing you pass to `pynozzle-stt`)      |
| `MOC_sl.plt`        | Streamlines emitted by the upstream `MOC_Grid_BDE` run         |
| `MOC_Grid.plt`      | MOC mesh, used for the reflective-wave validity check          |
| `summary.out`       | Scalar performance summary from the MOC run                    |
| `friction_table.txt`| Surface-area → friction-loss lookup table                     |

The three `MOC_*` files plus `summary.out` are exactly what
`pynozzle-moc2d` writes for the `examples/M3.5Perf.inp` case, so you can
regenerate them yourself and chain the two tools.

## Run it

```bash
pynozzle-stt M3.5Perf.inp -i . -o out
```

This writes the full STT2001 output set into `out/`:

* `M3.5Perf_ThroatSL.out` — throat-plane streamlines (ICEM CFD bulk data)
* `M3.5Perf_ThroatSummary.out` — per-streamline throat summary
* `M3.5Perf_TrimSL.out` — trimmed streamlines (ICEM CFD bulk data)
* `M3.5Perf_trimmed_P3D.xyz` / `.dat` — Plot3D grid + pressure
* `M3.5Perf_end_P3D.xyz` — streamline end-points
* `M3.5Perf.plt` / `M3.5Perf_Engine.plt` — Tecplot files
* `M3.5Perf_AvsX.out` / `M3.5Perf_AvsSL.out` — running area/force totals
* `M3.5Perf_cl.dat` — centerline geometry
* `M3.5Perf_all_runs.dat` — sweep summary row(s)
* `M3.5Perf_STT_summary.out` — human-readable performance summary

## Input format (19 lines)

```
line 1 : filePrefix  SL_file  MOC_grid_file  MOC_summary_file  friction_file
line 2 : RSL_start  RSL_end  RSL_step      (radial scale sweep of the SL field)
line 3 : XSL_start  XSL_end  XSL_step      (axial offset sweep)
line 4 : YSL_start  YSL_end  YSL_step      (lateral offset sweep)
line 5 : ZSL_start  ZSL_end  ZSL_step      (placeholders; unused)
line 6 : RC1 RC2 RC3 RC4 RC5               (throat/constraint circle radii)
line 7 : YC1..YC5                          (circle centre Y per slot)
line 8 : ZC1..ZC5                          (circle centre Z per slot)
line 9 : alpha1..alpha5                    (start angle, deg, per slot)
line 10: omega1..omega5                    (end angle, deg, per slot)
line 11: nSL1..nSL5                        (streamlines per slot)
line 12: XStart1..XStart5                  (constraint X-range start)
line 13: XEnd1..XEnd5                      (constraint X-range end)
line 14: pAmbient  aThroat  IspIdeal  MassFlow
line 15: RSim YSim ZSim nRev SL1Match SL2Match   (repeated-revolution symmetry)
line 16: Throat1..Throat5                   (1 = use slot as a throat boundary)
line 17: Constraint1..Constraint5           (1 = slot is active)
line 18: C1..C5                             (0 = inner surface, 1 = outer)
line 19: MaxLenCheck  MaxLenC  GridSF  XStatus  ContourFlag
```

For this example, only slot 1 is active: a full circle (`alpha=0`,
`omega=360`) of radius `RC1=1.0` with 72 streamlines, used as both the
throat boundary and the only constraint.

## A note on the summary numbers

`M3.5Perf_STT_summary.out` and `M3.5Perf_all_runs.dat` will *not* match
the Windows sample to the last digit. The original C++ parses scalars out
of `summary.out` with a fixed-width copy into a non-null-terminated
buffer and then `atof`s past the copied bytes into uninitialised memory —
so its sample reads the throat radius as `1.00099` instead of `1.000`,
which then perturbs the throat thrust, Isp and Cfg. This port reads the
correct `1.000`; every other output file matches bit-for-bit. See the top
level `README.md` for the full breakdown.
