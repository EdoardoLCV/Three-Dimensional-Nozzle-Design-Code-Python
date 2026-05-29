"""Engineering constants used by all three nozzle tools.

Direct port of ``engineering_constants.hpp`` from the original distribution.
Values are taken from the CRC Handbook (page references kept in comments).
"""
from __future__ import annotations

import math

# --- mathematical -----------------------------------------------------------
ZERO = 0.0
PI = 3.14159265358979323846264338327950288419716939937511

# --- physical ---------------------------------------------------------------
GASCON = 1545.0        # Universal Gas Constant [ft-lbf / (lbm-mol-R)]

# gravity
GRAV = 32.174
GRAV_SI = 9.80665

# mass
KG_PER_SLUG = 14.5939                            # F-345
SLUG_PER_KG = 1.0 / KG_PER_SLUG
LBM_PER_KG = 2.20462
KG_PER_LBM = 1.0 / LBM_PER_KG

# force
LBF_PER_N = 0.22480894                           # F-343
N_PER_LBF = 1.0 / LBF_PER_N

# pressure
PA_PER_PSF = 47.8803                             # F-344
PSF_PER_PA = 1.0 / PA_PER_PSF
PA_PER_ATM = 1.01325e5                           # F-326
ATM_PER_PA = 1.0 / PA_PER_ATM
ATM_PER_PSF = 0.000472541                        # F-344
PSF_PER_ATM = 1.0 / ATM_PER_PSF
ATM_PER_PSI = ATM_PER_PSF / 144.0
PSI_PER_ATM = 1.0 / ATM_PER_PSI
PA_PER_PSI = PA_PER_PSF * 144.0
PSI_PER_PA = 1.0 / PA_PER_PSI

# length
M_PER_FT = 0.3048                                # F-333
FT_PER_M = 1.0 / M_PER_FT
M_PER_IN = 0.0254                                # F-333
IN_PER_M = 1.0 / M_PER_IN
FT_PER_KM = 3280.8399                            # F-338
KM_PER_FT = 1.0 / FT_PER_KM
FT_PER_NMI = 6076.1155                           # F-342
NMI_PER_FT = 1.0 / FT_PER_NMI

# area / volume
IN2_PER_M2 = IN_PER_M * IN_PER_M
M2_PER_IN2 = 1.0 / IN2_PER_M2
FT2_PER_M2 = IN2_PER_M2 / 144.0
M2_PER_FT2 = 1.0 / FT2_PER_M2
FT3_PER_M3 = FT2_PER_M2 * FT_PER_M
M3_PER_FT3 = 1.0 / FT3_PER_M3

# angles
DEG_PER_RAD = 180.0 / PI
RAD_PER_DEG = 1.0 / DEG_PER_RAD

# temperature
R_PER_K = 1.8
K_PER_R = 1.0 / R_PER_K

# power
BTU_OVER_S_PER_WATT = 0.0009486608
WATT_PER_BTU_OVER_S = 1.0 / BTU_OVER_S_PER_WATT

# density
KG_OVER_M3_PER_LBM_OVER_FT3 = KG_PER_LBM / M3_PER_FT3
LBM_OVER_FT3_PER_KG_OVER_M3 = 1.0 / KG_OVER_M3_PER_LBM_OVER_FT3
KG_OVER_M3_PER_SLUG_OVER_FT3 = 16.01846 * GRAV
SLUG_OVER_FT3_PER_KG_OVER_M3 = 1.0 / KG_OVER_M3_PER_LBM_OVER_FT3 / GRAV

# enthalpy, energy, entropy
BTU_OVER_LBM_PER_KJ_OVER_KG = 0.430081
KJ_OVER_KG_PER_BTU_OVER_LBM = 1.0 / BTU_OVER_LBM_PER_KJ_OVER_KG
J_PER_BTU = 1055.056
BTU_PER_J = 1.0 / J_PER_BTU
BTU_OVER_LBM_R_PER_KJ_OVER_KG_K = BTU_OVER_LBM_PER_KJ_OVER_KG / R_PER_K
KJ_OVER_KG_K_PER_BTU_OVER_LBM_R = 1.0 / BTU_OVER_LBM_R_PER_KJ_OVER_KG_K

# viscosity / thermal
LBF_S_OVER_FT2_PER_MILLIPOISE = PSF_PER_PA / 10000.0
MILLIPOISE_PER_LBF_S_OVER_FT2 = 1.0 / LBF_S_OVER_FT2_PER_MILLIPOISE
BTU_OVER_FT2_S_R_PER_WATT_OVER_M2_K = BTU_OVER_S_PER_WATT * M2_PER_FT2 * K_PER_R
WATT_OVER_M2_K_PER_BTU_OVER_FT2_S = BTU_OVER_FT2_S_R_PER_WATT_OVER_M2_K
BTU_OVER_LBM_FT2_R_PER_KJ_OVER_KG_M2_K = (
    BTU_OVER_LBM_PER_KJ_OVER_KG * M2_PER_FT2 * K_PER_R
)
KJ_OVER_KG_M2_K_PER_BTU_OVER_LBM_FT2_R = 1.0 / BTU_OVER_LBM_FT2_R_PER_KJ_OVER_KG_M2_K


# --- enums ------------------------------------------------------------------
# These mirror the C++ enums; integer values preserved so input files keep working.
# nozzleGeom
NOGEOM = 0
TWOD = 1
AXI = 2

# nozzleType
NOTYPE = 0
RAO = 1          # minimum length (Rao)
CONE = 2
PERFECT = 3
FIXEDEND = 4

# designParam
NOPARAM = 0
EXITMACH = 1
EPS = 2
NOZZLELENGTH = 3
ENDPOINT = 4
EXITPRESSURE = 5

# Secant / convergence return flags
SEC_FAIL = 0
SEC_FAIL_LOW = 1
SEC_FAIL_HIGH = 2
SEC_OK = 3

# convergence_flag
NOT_CONVERGED = 0
CONVERGED = 1

# fail_flag
FAIL = 0
OK = 1

# flow_regime
SUBSONIC = 0
SONIC = 1
SUPERSONIC = 2


# --- inline helpers ---------------------------------------------------------
def xmin(a: float, b: float) -> float:
    """Equivalent of the ``xmin`` macro in the original header."""
    return a if a < b else b


def xmax(a: float, b: float) -> float:
    """Equivalent of the ``xmax`` macro in the original header."""
    return a if a > b else b


def deg2rad(deg: float) -> float:
    return deg * RAD_PER_DEG


def rad2deg(rad: float) -> float:
    return rad * DEG_PER_RAD


# Make ``math.pi`` and the legacy constant consistent enough for tests.
assert abs(PI - math.pi) < 1e-12
