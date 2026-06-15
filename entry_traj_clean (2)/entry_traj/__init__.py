"""entry_traj -- Phase-0 Venus entry trajectory tool (FA1).

Nominal RK4 ballistic entry, Monte Carlo dispersions with Sobol attribution,
Allen-Eggers / Pioneer Venus anchored validation, TPS sizing basis, EFPA
feasibility sweeps and VIRA profile ensembles.
"""

from .config import config_from_mapping, load_config
from .integrator import RunSample, integrate
from .models import Config, Ensemble, Report, Summary, Trajectory
from .monte_carlo import monte_carlo, sobol_indices, summarize
from .sweep import sweep_fpa
from .validation import validate

__version__ = "0.2.0"

__all__ = [
    "Config",
    "Ensemble",
    "Report",
    "RunSample",
    "Summary",
    "Trajectory",
    "__version__",
    "config_from_mapping",
    "integrate",
    "load_config",
    "monte_carlo",
    "sobol_indices",
    "summarize",
    "sweep_fpa",
    "validate",
]
