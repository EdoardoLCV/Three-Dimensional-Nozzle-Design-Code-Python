"""Top-level STT2001 solver — direct port of ``CSTT2001Dlg::OnExeButton``.

The C++ version is mixed with GUI code (text status updates, button
state, etc.). Those are dropped here; the orchestration logic is
preserved.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..common.constants import PI, DEG_PER_RAD, RAD_PER_DEG
from .inp import STTInput
from .loaders import (
    SLData, MOCSummaryData, MOCGridData,
    read_sl_plt, read_moc_summary, read_moc_grid,
)
from . import kernels


_N_PARAM_PTS = 4200       # matches the C++ ``nparamPTS``
_BASE_GRID_X = 100        # matches the C++ ``nparamGRIDX = 100`` default
_MAX_SLS = 999            # matches C++ ``SLCount = 999`` initial allocation


@dataclass
class STTResult:
    """Output of :meth:`STTSolver.run`."""
    success: bool
    error_message: str = ""

    # MOC scalars (read from the upstream ``summary.out``)
    moc_data: Optional[MOCSummaryData] = None

    # Performance scalars
    surface_area: float = 0.0     # m_SurfaceArea
    projected_area: float = 0.0   # m_ProjectedArea
    pressure_force: float = 0.0   # m_Force
    throat_thrust: float = 0.0    # m_thrust1
    fric_loss: float = 0.0        # m_fricLoss
    pa_loss: float = 0.0          # m_pALoss
    isp_calc: float = 0.0         # m_IspCalc
    a_exit: float = 0.0
    eps_moc: float = 0.0
    cxx: float = 0.0
    max_length: float = 0.0
    min_y: float = 0.0
    max_y: float = 0.0
    x_at_min_y: float = 0.0
    x_at_max_y: float = 0.0
    x_status_y: float = 0.0
    x_status_theta: float = 0.0

    # The full state, in case the caller wants raw arrays
    state: Optional["STTState"] = None


class STTState:
    """Container for all of the per-streamline arrays.

    Mirrors the dynamically-allocated arrays in :class:`CSTT2001Dlg`.
    Allocation happens once in :meth:`STTSolver._allocate`, matching the
    layout of ``InitializeArrays``.
    """

    def __init__(self, n_sl_max: int, n_param_pts: int, n_param_grid_x: int,
                 sl_data: SLData, inp: STTInput,
                 moc_grid: Optional[MOCGridData] = None):
        self.n_param_pts = n_param_pts
        self.n_param_grid_x = n_param_grid_x

        # Loaded streamlines and grids
        self.sl = sl_data
        self.moc_grid = moc_grid

        # Geometry of the SL "flow field" placement
        self.m_RSL = inp.RSL_start
        self.m_XSL = inp.XSL_start
        self.m_YSL = inp.YSL_start
        self.m_ZSL = 0.0  # placeholder per the C++

        # Throat-SL working arrays
        self.xt          = np.zeros((n_sl_max, n_param_pts))
        self.yt          = np.zeros((n_sl_max, n_param_pts))
        self.zt          = np.zeros((n_sl_max, n_param_pts))
        self.pt          = np.zeros((n_sl_max, n_param_pts))
        self.xt_end      = np.zeros(n_sl_max)
        self.yt_end      = np.zeros(n_sl_max)
        self.zt_end      = np.zeros(n_sl_max)
        self.new_sl_pts  = np.zeros(n_sl_max, dtype=np.int64)
        self.new_psi     = np.zeros(n_sl_max)
        self.nt          = np.zeros(n_sl_max, dtype=np.int64)
        self.rotate_sl   = np.zeros(n_sl_max, dtype=np.int64)
        self.circle_flag = 0
        self.n_new_sls   = 0

        # Resampled grid
        self.x_grid = np.zeros((n_sl_max, n_param_grid_x))
        self.y_grid = np.zeros((n_sl_max, n_param_grid_x))
        self.z_grid = np.zeros((n_sl_max, n_param_grid_x))
        self.p_grid = np.zeros((n_sl_max, n_param_grid_x))

        # Geometry extrema and bookkeeping
        self.x1 = sl_data.x_min
        self.x2 = sl_data.x_max
        self.max_length = 0.0
        self.min_y = 9e9
        self.max_y = -9e9
        self.n_at_max_x = 0
        self.n_at_max_y = 0
        self.n_at_min_y = 0
        self.x_at_max_y = 0.0
        self.x_at_min_y = 0.0

        # Performance scalars
        self.a_throat = inp.a_throat
        self.m_surface_area = 0.0
        self.m_projected_area = 0.0
        self.m_force = 0.0

        # Symmetry inputs
        self.sym_sl1_match = inp.sym.sl1_match
        self.sym_sl2_match = inp.sym.sl2_match
        # number of streamlines in the first slot, used by the circleFlag
        # path of TrimSLsDueToAxiRevolution. We pull it from inp.slots[0].
        self.m_nSL1 = inp.slots[0].nSL


class STTSolver:
    """Top-level orchestrator for the STT2001 streamline-tracing run.

    Typical use::

        result = STTSolver(input_dir, inp).run(out_dir)

    ``input_dir`` is where ``MOC_SL.plt``, ``summary.out``, and
    ``MOC_Grid.plt`` (as named in the .inp file) live.

    The constructor doesn't touch disk; :meth:`run` is where loading,
    solving, and writing happen.
    """

    def __init__(self, input_dir: str | Path, inp: STTInput):
        self.input_dir = Path(input_dir)
        self.inp = inp

    # ------------------------------------------------------------------
    def _resolve(self, name: str) -> Path:
        """Find an input file, looking first in input_dir then cwd."""
        if not name:
            return Path("")
        p = self.input_dir / name
        if p.is_file():
            return p
        if Path(name).is_file():
            return Path(name)
        return p  # caller will raise on FileNotFoundError later

    # ------------------------------------------------------------------
    def run(self, output_dir: str | Path) -> STTResult:
        from . import io_writers

        inp = self.inp
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # ---- Load input ----------------------------------------------
        try:
            sl_data = read_sl_plt(self._resolve(inp.sl_filename), _N_PARAM_PTS)
        except FileNotFoundError as e:
            return STTResult(success=False,
                             error_message=f"SL file not found: {e.filename}")

        moc_grid = None
        if inp.contour_flag and inp.moc_grid_file:
            try:
                moc_grid = read_moc_grid(self._resolve(inp.moc_grid_file))
            except (FileNotFoundError, ValueError) as e:
                # not fatal -- C++ would popup but we keep going
                moc_grid = None

        # ---- Allocate state ------------------------------------------
        n_grid_x = _BASE_GRID_X * max(1, inp.grid_sf)
        n_sl_max = _MAX_SLS
        for slot in inp.slots:
            if slot.throat and slot.constraint:
                n_sl_max += slot.nSL
        state = STTState(n_sl_max, _N_PARAM_PTS, n_grid_x,
                         sl_data, inp, moc_grid)

        # ---- Loops over SL location sweep ---------------------------
        if inp.RSL_step == 0.0:
            inp.RSL_step = inp.RSL_end - inp.RSL_start
        if inp.YSL_step == 0.0:
            inp.YSL_step = inp.YSL_end - inp.YSL_start
        if inp.XSL_step == 0.0:
            inp.XSL_step = inp.XSL_end - inp.XSL_start

        # The C++ runs nested loops over the sweep; we keep a single set
        # of result fields and only output the *last* iteration's files
        # (which matches the C++ behaviour of overwriting per iter).
        result = STTResult(success=False)
        sweep_ok = True

        # The all-runs summary file is opened once, before the sweep, and a
        # row is appended for every sweep iteration (matches the C++).
        all_runs_f = io_writers.write_all_runs_header(out, inp.file_prefix)

        rsl = inp.RSL_start
        while rsl <= inp.RSL_end + 1e-12:
            ysl = inp.YSL_start
            while ysl <= inp.YSL_end + 1e-12:
                xsl = inp.XSL_start
                while xsl <= inp.XSL_end + 1e-12:
                    state.m_RSL = rsl
                    state.m_XSL = xsl
                    state.m_YSL = ysl
                    state.m_ZSL = 0.0
                    state.n_new_sls = 0
                    state.m_force = 0.0
                    state.m_surface_area = 0.0
                    state.m_projected_area = 0.0

                    # 1. Throat SLs for every constraint slot that is
                    #    flagged as both constraint and throat.
                    go = True
                    for slot in inp.slots:
                        if not (slot.constraint and slot.throat) or not go:
                            continue
                        go = kernels.calc_throat_sls(
                            state, slot.yc, slot.zc, slot.rc,
                            slot.alpha, slot.omega, slot.nSL, slot.surface)
                    if not go:
                        sweep_ok = False
                        result.error_message = "Could not calculate throat streamlines"
                        break

                    # 2. Trim for non-throat constraints
                    any_constraint = False
                    for slot in inp.slots:
                        if slot.constraint and not slot.throat:
                            any_constraint = True
                            kernels.trim_sls(
                                state, slot.yc, slot.zc, slot.rc,
                                slot.alpha, slot.omega, slot.x_start, slot.x_end)
                    if not any_constraint:
                        # huge dummy that never triggers
                        kernels.trim_sls(state, 0.0, 0.0, 9.99e5,
                                         0.0, 0.0, -9.99e5, 9.99e5)

                    # 3. Reflective-wave-aware max-X
                    max_x_rw = -1.0
                    if inp.contour_flag and moc_grid is not None:
                        max_x_rw = kernels.find_max_x(state)
                    if inp.max_length_check:
                        target = inp.max_length_c
                        if max_x_rw > 0.0:
                            target = min(target, max_x_rw)
                        kernels.trim_sls_to_max_length(state, target)
                    elif max_x_rw > 0.0:
                        kernels.trim_sls_to_max_length(state, max_x_rw)

                    # 4. Resample to axial grid
                    kernels.calc_grid_sls(state)

                    # 5. Repeated-revolution trim
                    if inp.sym.n_rev > 1:
                        kernels.trim_sls_due_to_axi_revolution(
                            state, inp.sym.n_rev,
                            inp.sym.Y_sim, inp.sym.Z_sim, inp.sym.R_sim)

                    # 6. Performance integration
                    prefix = inp.file_prefix
                    kernels.calc_nozzle_parameters(
                        state,
                        a_vs_x_path=str(out / f"{prefix}_AvsX.out"),
                        a_vs_sl_path=str(out / f"{prefix}_AvsSL.out"),
                    )

                    # 7. Read upstream MOC summary, compute final scalars
                    try:
                        moc_data = read_moc_summary(
                            self._resolve(inp.moc_summary_filename))
                    except FileNotFoundError as e:
                        return STTResult(
                            success=False,
                            error_message=f"MOC summary file not found: {e.filename}",
                        )
                    result.moc_data = moc_data
                    if (moc_data.r_star_moc and moc_data.thrust_1_moc
                            and inp.a_throat):
                        result.throat_thrust = (
                            moc_data.thrust_1_moc * inp.a_throat
                            / (PI * moc_data.r_star_moc * moc_data.r_star_moc)
                        )
                    a_exit = inp.a_throat + state.m_projected_area
                    pa_loss = inp.p_ambient * a_exit
                    # Friction loss from the friction table, interpolated
                    # at the computed surface area (matches the C++ call
                    # GetFrictionLoss(m_SurfaceArea)).
                    from .loaders import get_friction_loss
                    fric_path = self._resolve(inp.friction_file)
                    fric_loss = get_friction_loss(fric_path, state.m_surface_area)
                    isp_calc = 0.0
                    if inp.mass_flow:
                        isp_calc = (
                            (result.throat_thrust + state.m_force
                             - fric_loss - pa_loss) / inp.mass_flow
                        )

                    result.surface_area     = state.m_surface_area
                    result.projected_area   = state.m_projected_area
                    result.pressure_force   = state.m_force
                    result.fric_loss        = fric_loss
                    result.pa_loss          = pa_loss
                    result.a_exit           = a_exit
                    result.eps_moc          = moc_data.eps_moc
                    result.isp_calc         = isp_calc
                    result.cxx              = (isp_calc / inp.isp_ideal
                                               if inp.isp_ideal else 0.0)
                    result.max_length       = state.max_length
                    result.min_y            = state.min_y
                    result.max_y            = state.max_y
                    result.x_at_min_y       = state.x_at_min_y
                    result.x_at_max_y       = state.x_at_max_y

                    # X-status: Y and wall-angle at the min-Y SL at the
                    # requested axial station (matches the C++ guard
                    # ``if (m_NAtMinY < nNewSLs && m_NAtMinY >= 0)``).
                    y_x_status = 0.0
                    theta_x_status = 0.0
                    if 0 <= state.n_at_min_y < state.n_new_sls:
                        y_x_status, theta_x_status = kernels.calc_x_status(
                            n_grid_x, inp.x_status,
                            state.x_grid[state.n_at_min_y],
                            state.y_grid[state.n_at_min_y])
                    result.x_status_y = y_x_status
                    result.x_status_theta = theta_x_status

                    # ---- Per-run summary row -----------------------------
                    io_writers.write_all_runs_row(
                        all_runs_f, inp, state, result,
                        z_sl=state.m_ZSL, x_sl=xsl, y_sl=ysl, r_sl=rsl,
                        y_x_status=y_x_status, theta_x_status=theta_x_status)

                    # ---- Output files (overwritten each iter; last wins) -
                    io_writers.write_throat_sls(state, out, prefix)
                    io_writers.write_stt_summary(state, inp, moc_data,
                                                 result, out)

                    if inp.XSL_step == 0.0:
                        break
                    xsl += inp.XSL_step
                if not sweep_ok:
                    break
                if inp.YSL_step == 0.0:
                    break
                ysl += inp.YSL_step
            if not sweep_ok:
                break
            if inp.RSL_step == 0.0:
                break
            rsl += inp.RSL_step

        all_runs_f.close()

        # The trimmed-SL files and centerline plot are written once, after
        # the sweep completes, using the final iteration's state (matches
        # the C++ flow where they're emitted after the sweep loops).
        if sweep_ok and result.state is not None or state.n_new_sls > 0:
            io_writers.write_trimmed_sls(state, inp, out, inp.file_prefix)
            io_writers.write_centerline_plot(state, out, inp.file_prefix)

        result.success = sweep_ok
        result.state = state
        return result
