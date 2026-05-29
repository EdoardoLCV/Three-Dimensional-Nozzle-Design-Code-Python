# Example input files

These reproduce the cases shipped with the original C++ distribution. The
`.inp` format is whitespace-separated and was designed for the original
GUI to round-trip its radio-button and text-field state, so it's not
self-documenting. The schema is below for reference; in practice you'll
get the best results by saving a case from the original Windows GUI or
by editing one of these as a starting point.

## File format

Line 1 (11 boolean flags, 0 or 1):

| field        | meaning                                              |
|--------------|------------------------------------------------------|
| `throat`     | 1 → inputs are throat static conditions, 0 → total   |
| `perfect`    | 1 → perfect nozzle                                   |
| `exit_mach`  | 1 → design parameter is exit Mach number             |
| `axi`        | 1 → axisymmetric geometry                            |
| `two_d`      | 1 → planar (2D) geometry                             |
| `cone`       | 1 → cone nozzle                                      |
| `min_length` | 1 → Rao minimum-length nozzle                        |
| `end_point`  | 1 → fixed end-point nozzle                           |
| `eps`        | 1 → design parameter is exit area ratio              |
| `length`     | 1 → design parameter is nozzle length                |
| `p_exit`     | 1 → design parameter is exit pressure ratio          |

Lines 2–8 (numeric inputs, fixed order — same order written by the GUI's
File→Save):

```
m_design  n_c  rwt_d  mol_wt_i
pres_i    temp_i  gamma_i  rwt_u
d_t_limit p_amb
eps       length  r_e  x_e
n_rrc     p_exit
n_sl_i    n_sl_j  vel  theta_bi
cone_angle
```

Where:

* `m_design` — design exit Mach number
* `n_c` — number of characteristic lines at the throat (odd; 101 is typical)
* `rwt_d` / `rwt_u` — downstream / upstream wall-radius ratio (R/R*)
* `mol_wt_i`, `gamma_i` — gas properties
* `pres_i`, `temp_i` — total pressure (psia) and total temperature (°R),
  unless `throat=1`, in which case these are throat static values
* `d_t_limit` — max angle (deg) between wall characteristics
* `p_amb` — ambient pressure (psia); 0 = vacuum
* `eps`, `length`, `r_e`, `x_e` — alternative design targets used when
  `eps` / `length` / `end_point` is the chosen design parameter
* `n_rrc` — number of RRCs above the BD region (50–100 typical)
* `p_exit` — total/exit pressure ratio used when `p_exit=1`
* `n_sl_i`, `n_sl_j` — streamline density (radial, axial)
* `vel` — throat velocity (ft/s) used when `throat=1`
* `theta_bi` — initial ThetaB guess (deg)
* `cone_angle` — cone half-angle (deg) used when `cone=1`

Unused values can be left at any default.

## The supplied examples

* `M3.5Perf.inp` — axisymmetric perfect nozzle, M_exit = 3.5, γ = 1.4,
  p₀ = 500 psia, T₀ = 530 °R. Reproduces the upstream `outputs_M3.5Perf`
  case bit-for-bit (modulo two documented cosmetic differences — see
  the project README).
* `M4Rao.inp` — Rao minimum-length nozzle, M_exit = 4.
* `cone10.inp` — 10° half-angle cone, M_exit = 4.
