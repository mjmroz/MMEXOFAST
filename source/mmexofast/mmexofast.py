# mm_exofast_fitter.py
"""
High-level class and convenience wrapper for fitting microlensing events
with MM-EXOFASTv2.
"""

from __future__ import annotations

import inspect
import json
import logging
import os.path
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import MulensModel
import numpy as np
import pandas as pd
from scipy.special import erfcinv

from .classifier import AnomalyClassifier
from .estimate_params import (
    AnomalyPropertyEstimator,
    BinaryLensParams,
    CloseAxisGridSearchEstimator,
    CloseLowerGridSearchEstimator,
    CloseUpperGridSearchEstimator,
    ParameterEstimator,
    WideAxisGridSearchEstimator,
    WideLowerGridSearchEstimator,
    WideUpperGridSearchEstimator,
    get_PSPL_params,
)
from .fit_types import (
    FitKey,
    LensOrbMotion,
    LensType,
    ParallaxBranch,
    SourceType,
    label_to_model_key,
    model_key_to_label,
)
from .fitters import AnomalyFitter, EmceeFitResults, SFitFitter
from .gridsearches import (
    AnomalyFinderGridSearch,
    EventFinderGridSearch,
    NoAnomalyFoundError,
    ParallaxGridSearch,
)
from .mulens_object_config import EventConfig, ModelConfig
from .observatories import get_kwargs, get_telescope_band_from_filename
from .results import (
    AllFitResults,
    FitRecord,
    IntermediateResults,
    MMEXOFASTFitResults,
)
from .workflow_step import StepStatus, WorkflowStep

logger = logging.getLogger(__name__)


# ===========================================================================
# Module-level constants for table formatting
# ===========================================================================

_PARAMETER_DECIMAL_PLACES = {
    "chi2": 2,
    "N_data": 0,
    "t_0": 6,
    "u_0": 6,
    "t_E": 2,
    "rho": 6,
    "log_rho": 3,
    "t_star": 6,
    "pi_E_N": 4,
    "pi_E_E": 4,
    "t_0_par": 6,
    "s": 6,
    "log_s": 3,
    "q": 6,
    "log_q": 3,
    "alpha": 2,
    "convergence_K": 6,
    "shear_G": 6,
    "ds_dt": 3,
    "dalpha_dt": 3,
    "s_z": 6,
    "ds_z_dt": 3,
    "t_0_kep": 6,
    "x_caustic_in": 6,
    "x_caustic_out": 6,
    "t_caustic_in": 6,
    "t_caustic_out": 6,
    "xi_period": 3,
    "xi_semimajor_axis": 6,
    "xi_inclination": 2,
    "xi_Omega_node": 2,
    "xi_argument_of_latitude_reference": 2,
    "xi_eccentricity": 4,
    "xi_omega_periapsis": 2,
    "q_source": 6,
    "t_0_xi": 6,
}

# Parameters that are epochs, and so move when the time origin changes.
# Deliberately excludes the durations t_E, t_star, t_star_1, t_star_2 and
# xi_period, which are invariant under a shift: adding a JD offset to a
# duration would be a bug, not a rounding difference. Used by
# MMEXOFASTFitter.initialize_exozippy to convert reduced HJD to full JD.
EPOCH_PARAMETERS = frozenset(
    {
        "t_0",
        "t_0_1",
        "t_0_2",
        "t_0_kep",
        "t_0_par",
        "t_0_xi",
        "t_caustic_in",
        "t_caustic_out",
    }
)

_FLUX_PARAM_DECIMAL_PLACES = 3


def _get_decimal_places(param_name: str) -> Optional[int]:
    """
    Return the number of decimal places for formatting a parameter value.

    Returns None if the parameter is not in the known list.

    Handles binary source parameters (e.g. ``'t_0_1'``, ``'t_0_2'``) by
    stripping the trailing source index and looking up the base name.

    Handles flux parameters (e.g. ``'I_S_OGLE'``, ``'R_B_MOA'``) by
    detecting the ``'_S_'`` or ``'_B_'`` pattern.

    Parameters
    ----------
    param_name : str
        Parameter name to look up.

    Returns
    -------
    int or None
        Number of decimal places, or None if not in the known list.
    """
    if param_name in _PARAMETER_DECIMAL_PLACES:
        return _PARAMETER_DECIMAL_PLACES[param_name]

    # Binary source parameters: e.g. 't_0_1' -> 't_0'
    parts = param_name.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        base = parts[0]
        if base in _PARAMETER_DECIMAL_PLACES:
            return _PARAMETER_DECIMAL_PLACES[base]

    # Flux parameters: e.g. 'I_S_OGLE'
    parts = param_name.split("_")
    if len(parts) >= 3 and parts[1] in ("S", "B"):
        return _FLUX_PARAM_DECIMAL_PLACES

    return None


def _format_results_column(df: pd.DataFrame, pm_symbol: str) -> pd.DataFrame:
    """
    Format values and sigma columns in a single-model results DataFrame.

    Applies parameter-specific decimal places to ``'values'``,
    ``'sigmas'``, ``'sigma_minus'``, and ``'sigma_plus'`` columns.  NaN
    sigmas become empty strings.  Sigma values are prefixed with the
    appropriate symbol:

    - ``sigmas``:      ``"{pm_symbol} {value}"``
    - ``sigma_minus``: ``"- {value}"``
    - ``sigma_plus``:  ``"+ {value}"``

    String values (e.g. ``'neg flux'``) are passed through unchanged.
    Parameters not in the known list are formatted with ``str()``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ``'parameter_names'``, ``'values'``, and
        optionally ``'sigmas'``, ``'sigma_minus'``, ``'sigma_plus'``.
    pm_symbol : str
        Symbol to prefix symmetric sigma values: ``'+/-'`` for ASCII,
        ``r'$\\pm$'`` for LaTeX.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with formatted string values.
    """
    df = df.copy()

    def fmt(param_name, value, prefix=""):
        if pd.isna(value):
            return ""
        if isinstance(value, str):
            return f"{prefix}{value}" if prefix else value
        decimal_places = _get_decimal_places(param_name)
        if decimal_places is None:
            return f"{prefix}{value}" if prefix else str(value)
        formatted = f"{value:.{decimal_places}f}"
        return f"{prefix}{formatted}" if prefix else formatted

    df["values"] = [
        fmt(param, val)
        for param, val in zip(df["parameter_names"], df["values"])
    ]
    if "sigmas" in df.columns:
        df["sigmas"] = [
            fmt(param, val, prefix=f"{pm_symbol} ")
            for param, val in zip(df["parameter_names"], df["sigmas"])
        ]
    if "sigma_minus" in df.columns:
        df["sigma_minus"] = [
            fmt(param, val, prefix="- ")
            for param, val in zip(df["parameter_names"], df["sigma_minus"])
        ]
    if "sigma_plus" in df.columns:
        df["sigma_plus"] = [
            fmt(param, val, prefix="+ ")
            for param, val in zip(df["parameter_names"], df["sigma_plus"])
        ]

    return df


# ===========================================================================
# OutputConfig
# ===========================================================================


@dataclass
class OutputConfig:
    """
    Lightweight configuration for file output.

    Parameters
    ----------
    output_dir : Path
        Directory to write output files.  Created if it does not exist.
    file_prefix : str
        Prefix added to all output file names.
    save_plots : bool
        Whether to save figures to disk.
    save_grid_results : bool
        Whether to save raw grid search results to text files.
    save_table : bool
        Whether to save fit results tables to disk.
    table_formats : str or list of str
        Table format(s) to save.  Each entry must be ``'ascii'`` or
        ``'latex'``.  A bare string is accepted and wrapped in a list.
        Defaults to ``['latex']``.
    save_exozippy_init : bool
        Whether to save the EXOZIPPy initialization dict to a JSON file.

    """

    output_dir: Path = field(default_factory=Path)
    file_prefix: str = ""
    save_plots: bool = True
    save_grid_results: bool = False
    save_table: bool = False
    table_formats: list = field(default_factory=lambda: ["latex"])
    save_exozippy_init: bool = False

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(self.table_formats, str):
            self.table_formats = [self.table_formats]

    def plot_path(self, name: str, ext: str = "pdf") -> Path:
        """
        Return the full path for a named plot file.

        Parameters
        ----------
        name : str
            Base name of the plot.
        ext : str
            File extension without leading dot.

        Returns
        -------
        Path
        """
        prefix = f"{self.file_prefix}_" if self.file_prefix else ""
        return self.output_dir / f"{prefix}{name}.{ext}"

    def grid_path(self, name: str) -> Path:
        """
        Return the full path for a named grid result file.

        Parameters
        ----------
        name : str
            Base name of the grid file.

        Returns
        -------
        Path
        """
        prefix = f"{self.file_prefix}_" if self.file_prefix else ""
        return self.output_dir / f"{prefix}{name}.txt"

    def table_path(self, fmt: str) -> Path:
        """
        Return the full path for a results table file.

        Parameters
        ----------
        fmt : str
            Table format: ``'ascii'`` or ``'latex'``.

        Returns
        -------
        Path
        """
        ext = "tex" if fmt == "latex" else "txt"
        prefix = f"{self.file_prefix}_" if self.file_prefix else ""
        return self.output_dir / f"{prefix}results.{ext}"

    def exozippy_init_path(self) -> Path:
        """
        Return the full path for the EXOZIPPy initialization JSON file.

        Returns
        -------
        Path
        """
        prefix = f"{self.file_prefix}_" if self.file_prefix else ""
        return self.output_dir / f"{prefix}exozippy_init.json"


# ===========================================================================
# Module-level convenience wrapper
# ===========================================================================


def fit(
    datasets=None,
    files=None,
    fit_type: str = "point_lens",
    **kwargs,
) -> "MMEXOFASTFitter":
    """
    Convenience wrapper: construct an ``MMEXOFASTFitter`` and run it.

    Parameters
    ----------
    datasets : list of MulensModel.MulensData, optional
        Pre-loaded dataset objects.
    files : str or list of str, optional
        Data file paths to load.
    fit_type : str
        ``'point_lens'`` or ``'binary_lens'``.
    **kwargs
        Passed directly to ``MMEXOFASTFitter.__init__``.

    Returns
    -------
    MMEXOFASTFitter
        The fitter after ``fit()`` has been called.
    """
    with MMEXOFASTFitter(
        datasets=datasets,
        files=files,
        fit_type=fit_type,
        **kwargs,
    ) as fitter:
        return fitter.fit()


# ===========================================================================
# MMEXOFASTFitter
# ===========================================================================


class MMEXOFASTFitter:
    """
    Orchestrates the full MM-EXOFASTv2 microlensing fitting workflow.

    Parameters
    ----------
    datasets : list of MulensModel.MulensData, optional
        Pre-loaded dataset objects, each with a unique label in
        ``plot_properties['label']``.  Mutually exclusive with *files*.
    files : str or list of str, optional
        Data file paths to load.  Mutually exclusive with *datasets*.
    coords : str or MulensModel.Coordinates, optional
        Sky coordinates of the event.
    fit_type : str
        ``'point_lens'`` or ``'binary_lens'``.
    finite_source_point_lens : bool or ``u_0<float``
        If True, include FSPL fitting steps after PSPL.
        if e.g.``u_0<0.001`` then FSPL is only run if the fitted u_0 from PSPL model is less than 0.001.
    source_type : str
        ``'dwarf'`` or ``'giant'``. Default is ``'giant'``.  Used to set a initial value of rho for FSPL fitting.
    mag_methods : list, optional
        Magnification methods in MulensModel convention.
    vbbl_accuracy : float, optional
        VBBL/VBM integration tolerance for binary-lens models. Default
        0.01, which is VBMicrolensing's own default; MulensModel
        otherwise applies 0.001, roughly 3x slower for a magnification
        difference of ~6e-4. See :class:`BinaryLensParams`.
    limb_darkening_coeffs_u : dict, optional
        Linear limb-darkening coefficients keyed by bandpass.
    limb_darkening_coeffs_gamma : dict, optional
        Gamma limb-darkening coefficients keyed by bandpass.
    fix_blend_flux : dict, optional
        Mapping of dataset label to blend flux fixing flag.
    fix_source_flux : dict, optional
        Mapping of dataset label to source flux fixing flag.
    renormalize_errors : bool
        Whether to renormalize dataset errors during the workflow.
    parallax_point_lens : bool or str
        Whether to include parallax in the point lens fitting workflow.
        Default is True (include parallax).
        If 'grid' a parallax grid search will be performed at the end of point lens workflow. TODO: but what is the reason for that?
    parallax_binary_lens : bool or str
        Whether to include parallax in the binary lens fitting workflow.
        Default is True (include parallax after static binary lens fit).
    primary_location : str, optional
        Location name to treat as primary (e.g. ``'ground'``, ``'Spitzer'``).
    primary_dataset : str, optional
        Label of dataset used to identify the primary location.
    emcee_settings : dict, optional
        Settings passed to the EMCEE-based binary fitter.
    dry_run : bool
        If True, build the workflow but do not execute any steps.
    stop_before : str, optional
        Name of the step before which execution halts.
    stop_after : str, optional
        Name of the step after which execution halts.
    restart_file : path-like, optional
        Path to a restart pickle file.  If the file exists it is loaded
        to restore previous state.  After every completed step the
        current state is written back to this same path, so the file
        always reflects the latest checkpoint.
    restart_from : str, optional
        Step name from which to re-run; all completed steps recorded
        after this point are discarded.
    initial_results : dict, optional
        User-supplied fit results to seed the workflow.  See
        ``_load_initial_results`` for the expected key/value format.
        Mutually exclusive with ``restart_from``.  Only two entry points
        are supported: ``fit_type='point_lens'`` with a PSPL result
        (starts at ``fit_static_point_source_point_lens``), and ``fit_type='binary_lens'`` with any
        PSPL result (starts at ``select_best_point_lens_model``,
        skipping all point-lens stages).
    output_config : OutputConfig, optional
        Controls file output (plots, grid files).  If None, no files are
        written.
    verbose : bool
        If True, configure the module logger to emit DEBUG-level messages
        to stdout.
    log_file : path-like, optional
        Path to a file for DEBUG-level log output.  The file is created
        (or appended to) at construction time.  Call ``close()`` to
        release the file handle when the fitter is no longer needed.


    Notes
    -----
    **Workflow entry points via** ``initial_results``

    When the user supplies pre-computed fit results, the workflow skips
    the steps needed to produce those results.  Only the following
    combinations are supported:

    .. list-table::
       :header-rows: 1
       :widths: 20 25 55

       * - ``fit_type``
         - Result supplied
         - First step executed
       * - ``'point_lens'``
         - Static PSPL (``lens_type=POINT``, ``parallax_branch=NONE``)
         - ``fit_static_point_source_point_lens`` — the supplied params are used as the fitting
           seed; ``estimate_point_lens_parameters`` is skipped
       * - ``'binary_lens'``
         - Any PSPL (static or parallax)
         - ``select_best_point_lens_model`` — all point-lens stages are
           skipped; the supplied model is used immediately as the
           reference for the anomaly search

    ``initial_results`` and ``restart_from`` are mutually exclusive;
    providing both raises ``ValueError`` at construction time.

    **Restart behavior**

    When ``restart_file`` is provided, all saved state (fit results,
    completed steps, renormalization factors, datasets) is restored
    before any new steps run.  The ``restart_from`` parameter discards
    all recorded progress at and after the named step, forcing those
    steps to re-run.

    **Stop points**

    ``stop_before`` and ``stop_after`` accept either a stage name
    (e.g. ``'fit_static_point_lens'``) or a ``stage:step`` string
    (e.g. ``'fit_static_point_lens:fit_static_point_source_point_lens'``).  When a stage name is
    used, ``stop_before`` halts before the first step of that stage and
    ``stop_after`` halts after the last step of that stage.

    **Model and Event configuration**

    ``model_config`` and ``event_config`` are built from the supplied
    parameters at the end of ``__init__`` and serve as the single source of
    truth for all ``MulensModel.Model`` and ``MulensModel.Event`` construction
    within this fitter.  ``event_config`` is rebuilt after renormalization
    via ``_build_event_config()`` because renormalization replaces dataset
    objects, which invalidates the dataset-keyed flux-fixing maps.

    """

    CONFIG_KEYS = [
        "fit_type",
        "coords",
        "finite_source_point_lens",
        "source_type",
        "mag_methods",
        "vbbl_accuracy",
        "limb_darkening_coeffs_u",
        "limb_darkening_coeffs_gamma",
        "fix_blend_flux",
        "fix_source_flux",
        "renormalize_errors",
        "parallax_point_lens",
        "parallax_binary_lens",
        "primary_location",
        "primary_dataset",
        "emcee_settings",
    ]

    PARALLAX_GRID_PARAMS_COARSE = {
        "pi_E_E": [-1.0, 1.0, 0.15],
        "pi_E_N": [-1.5, 1.5, 0.30],
    }

    PARALLAX_GRID_PARAMS_FINE = {
        "pi_E_E": [-0.7, 0.7, 0.025],
        "pi_E_N": [-1.0, 1.0, 0.050],
    }

    RENORM_THRESHOLD = 0.02

    # Renormalization's outlier rejection compares the data to a point-lens
    # reference model; points the model magnifies above this are protected
    # from rejection, because there the true (binary, finite-source) light
    # curve can deviate from the reference by far more than the ~3-sigma
    # threshold at space-photometry precision. See _build_protected_mask.
    PEAK_PROTECTION_MAG = 1.1

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def __init__(
        self,
        datasets=None,
        files=None,
        coords=None,
        source_type: str = "giant",
        fit_type: str = "point_lens",
        finite_source_point_lens: bool = False,
        mag_methods=None,
        vbbl_accuracy: float = BinaryLensParams._DEFAULT_VBBL_ACCURACY,
        limb_darkening_coeffs_u=None,
        limb_darkening_coeffs_gamma=None,
        fix_blend_flux=None,  # TODO: Check whether fixed fluxes are implemented
        fix_source_flux=None,
        renormalize_errors: bool = True,  # TODO: ADD option for remove_outliers=True/False
        parallax_point_lens: bool or str = True,
        parallax_binary_lens: bool = False,
        primary_location=None,
        primary_dataset=None,
        emcee_settings=None,
        dry_run: bool = False,
        stop_before: Optional[str] = None,
        stop_after: Optional[str] = None,
        restart_file=None,
        restart_from=None,
        initial_results=None,  # TODO: Implement starting from prior results.
        output_config=None,
        verbose: bool = False,
        log_file=None,
        pool=None,
    ) -> None:
        # Mutually exclusive input validation
        if files is not None and datasets is not None:
            raise ValueError("Specify 'files' or 'datasets', not both.")
        if initial_results is not None and restart_from is not None:
            raise ValueError(
                "Specify 'initial_results' or 'restart_from', not both."
            )

        # Track handlers added by this instance for cleanup via close()
        self._log_handlers: list[logging.Handler] = []

        # verbose / log_file → configure module logger
        if verbose or log_file is not None:
            _mod_logger = logging.getLogger(__name__)
            _mod_logger.setLevel(logging.DEBUG)
            if verbose:
                _handler = logging.StreamHandler()
                _mod_logger.addHandler(_handler)
                self._log_handlers.append(_handler)
            if log_file is not None:
                _handler = logging.FileHandler(log_file)
                _mod_logger.addHandler(_handler)
                self._log_handlers.append(_handler)

        # Output config
        self._output_config: Optional[OutputConfig] = output_config

        # Config from restart file merged with current call
        saved_config, saved_state = self._load_restart_data(restart_file)
        self._restart_path = (
            Path(restart_file) if restart_file is not None else None
        )
        config = self._merge_config(saved_config, locals())
        self._set_config_attributes(config)

        # Derived from the binary parameter estimators once they run; see
        # est_binary_params.
        self.mag_methods_parameters = None

        # Execution-time controls (not persisted in CONFIG_KEYS)
        self.dry_run = dry_run
        self.stop_before = stop_before
        self.stop_after = stop_after
        # Multiprocessing for the emcee-based fitters (MulensFitter's `pool`
        # option): None/False = serial, True = one process per CPU, int = that
        # many processes (use the int form on shared cluster nodes, where
        # cpu_count() sees the whole node rather than the job's grant).
        self.pool = pool

        # WorkflowStep tracking
        self.completed_steps: list[WorkflowStep] = []
        self.planned_steps: list[WorkflowStep] = []

        # Restore computed state
        self._restore_state(saved_state)

        # Truncate completed_steps if restart_from is specified
        if restart_from is not None:
            idx = next(
                (
                    i
                    for i, s in enumerate(self.completed_steps)
                    if self._step_matches_stop_value(restart_from, s)
                ),
                None,
            )
            if idx is not None:
                self.completed_steps = self.completed_steps[:idx]

        # Dataset construction
        if files is not None:
            self.datasets = self._create_mulensdata_objects(
                files, saved_datasets=saved_state.get("datasets")
            )
        elif datasets is not None:
            self.datasets = datasets
            self._validate_dataset_labels()
        elif saved_state.get("datasets"):
            self.datasets = saved_state["datasets"]
        else:
            raise ValueError(
                "Provide at least one of: 'files', 'datasets', or "
                "'restart_file'."
            )

        self._check_dataset_labels_unique()

        # Flux-fixing maps (depend on self.fix_blend_flux /
        # self.fix_source_flux set by _set_config_attributes above)
        self.fix_blend_flux_map = self._map_label_dict_to_datasets(
            self.fix_blend_flux
        )
        self.fix_source_flux_map = self._map_label_dict_to_datasets(
            self.fix_source_flux
        )

        # Single source of truth for all Model and Event construction in this
        # fitter. event_config is rebuilt after renormalization because dataset
        # objects are replaced at that point.
        self.model_config = ModelConfig(
            coords=self.coords,
            limb_coeff_u=self.limb_darkening_coeffs_u,
            limb_coeff_gamma=self.limb_darkening_coeffs_gamma,
        )
        self.event_config = self._build_event_config()

        # Load initial results and infer entry point
        self._initial_entry_point: Optional[str] = None
        if initial_results is not None:
            self._load_initial_results(initial_results)
            if not self.completed_steps:
                self._initial_entry_point = (
                    self._infer_entry_point_from_initial_results()
                )
        self.finite_source_point_lens = self._parse_finite_source_point_lens(
            finite_source_point_lens
        )

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _parse_finite_source_point_lens(self, value):
        """
        Parse the finite_source_point_lens argument.

        Parameters
        ----------
        value : bool or float
            If True, include FSPL fitting steps after PSPL.
            If float, include FSPL only if fitted u_0 from PSPL is less than this value.
            if str, must be in the format 'parameter_name < value' or 'parameter_name > value',
            where parameter_name is one of 'u_0' or 't_E'.

        Returns
        -------
        bool or list
            Parsed value.
        """
        allowed_parameters = ["u_0", "t_E"]
        if isinstance(value, bool):
            return value
        elif isinstance(value, (int, float)):
            if value < 0:
                raise ValueError(
                    "finite_source_point_lens must be non-negative."
                )

            return ["u_0", np.less, value]
        elif isinstance(value, str):
            if len(value.split("<")) == 2:
                parts = value.split("<")
                function = np.less
            elif len(value.split(">")) == 2:
                parts = value.split(">")
                function = np.greater
            else:
                raise ValueError(
                    "finite_source_point_lens string must be in the format"
                    + " 'parameter_name < value' or 'parameter_name > value'."
                )
            if parts[0].strip() not in allowed_parameters:
                raise ValueError(
                    f"finite_source_point_lens parameter name must be one of {allowed_parameters}."
                )
            return [parts[0].strip(), function, float(parts[1].strip())]
        else:
            raise TypeError(
                "finite_source_point_lens must be a bool, string, or a non-negative float."
            )

    def _load_restart_data(self, restart_file) -> tuple[dict, dict]:
        """
        Load a saved restart file.

        Parameters
        ----------
        restart_file : path-like or None
            Path to the restart pickle file.

        Returns
        -------
        saved_config : dict
            Configuration dict stored at save time.
        saved_state : dict
            Runtime state dict stored at save time.
        """
        if restart_file is None:
            return {}, {}
        if not Path(restart_file).exists():
            logger.info(
                "No restart file found at %s; starting fresh.", restart_file
            )
            return {}, {}

        logger.info("Loading restart data from: %s", restart_file)
        with open(restart_file, "rb") as f:
            data = pickle.load(f)

        config = data.get("config", {})
        state = data.get("state", {})
        logger.info(
            "  Loaded %d fit result(s).",
            len(state.get("all_fit_results", AllFitResults())),
        )
        return config, state

    def _merge_config(self, saved_config: dict, locals_: dict) -> dict:
        """
        Merge a saved config with the current call's local config dict.

        Current call values win on conflict.

        Parameters
        ----------
        saved_config : dict
            Configuration loaded from the restart file.
        locals_ : dict
            Configuration derived from the current ``__init__`` arguments.

        Returns
        -------
        dict
            Merged configuration dict.
        """
        merged = {}
        for key in self.CONFIG_KEYS:
            if key in locals_ and locals_[key] is not None:
                merged[key] = locals_[key]
            elif key in saved_config:
                merged[key] = saved_config[key]
            else:
                merged[key] = None
        return merged

    def _set_config_attributes(self, config: dict) -> None:
        """
        Apply a config dict as attributes on self.

        Parameters
        ----------
        config : dict
            Key-value pairs to set as instance attributes.
        """
        for key in self.CONFIG_KEYS:
            setattr(self, key, config[key])

    def _restore_state(self, saved_state: dict) -> None:
        """
        Restore runtime state from a serialized dict.

        Parameters
        ----------
        saved_state : dict
            State dict previously produced by ``_get_state()``.

        Notes
        -----
        ``completed_steps`` may be stored as ``(name, stage)`` tuples in
        the restart file because ``WorkflowStep.func`` callables cannot be
        pickled.  Each tuple is reconstructed as a stub ``WorkflowStep``
        with a no-op func sufficient for tracking purposes.
        """
        self.all_fit_results = saved_state.get(
            "all_fit_results", AllFitResults()
        )

        raw_completed = saved_state.get("completed_steps", [])
        self.completed_steps = [
            step
            if isinstance(step, WorkflowStep)
            else WorkflowStep(
                name=step[0],
                func=lambda: None,
                stage=step[1],
                description="(restored from restart file)",
            )
            for step in raw_completed
        ]

        self.intermediate_results = saved_state.get(
            "intermediate_results", IntermediateResults()
        )
        self.renorm_factors: dict = saved_state.get("renorm_factors", {})
        self.residuals: dict = saved_state.get("residuals", None)

    def _get_state(self) -> dict:
        """
        Return a serializable dict of current runtime state.

        Returns
        -------
        dict
            Contains ``all_fit_results``, ``completed_steps`` (as
            ``(name, stage)`` tuples since callables cannot be pickled),
            ``intermediate_results``, ``renorm_factors``, and ``datasets``.
        """
        return {
            "all_fit_results": self.all_fit_results,
            "completed_steps": [
                (s.name, s.stage) for s in self.completed_steps
            ],
            "intermediate_results": self.intermediate_results,
            "renorm_factors": self.renorm_factors,
            "datasets": self.datasets,
            "residuals": self.residuals,
        }

    def _load_initial_results(self, initial_results) -> None:
        """
        Load user-supplied fit results into ``self.all_fit_results``.

        Parameters
        ----------
        initial_results : dict
            Mapping of model label strings to payload dicts.  Each
            payload must contain ``'params'`` and may contain
            ``'sigmas'``, ``'renorm_factors'``, and ``'fixed'``.
        """
        for label, payload in initial_results.items():
            key = label_to_model_key(label)
            record = FitRecord(
                model_key=key,
                params=payload["params"],
                sigmas=payload.get("sigmas"),
                renorm_factors=payload.get("renorm_factors"),
                full_result=None,
                fixed=payload.get("fixed", False),
                is_complete=False,
            )
            self.all_fit_results.set(record)

    def _build_event_config(self) -> EventConfig:
        """
        Build an ``EventConfig`` from the current flux-fixing maps.

        Separated from ``__init__`` because renormalization replaces dataset
        objects, which requires the flux-fixing maps — and therefore
        ``event_config`` — to be rebuilt with updated dataset keys.

        Returns
        -------
        EventConfig
        """
        return EventConfig(
            coords=self.coords,
            fix_blend_flux=self.fix_blend_flux_map,
            fix_source_flux=self.fix_source_flux_map,
        )

    # ------------------------------------------------------------------
    # Workflow execution
    # ------------------------------------------------------------------

    def fit(self) -> AllFitResults:
        """
        Main entry point; build and execute the workflow steps.

        Returns
        -------
        AllFitResults
            All fit records accumulated during the workflow.

        Raises
        ------
        ValueError
            If ``fit_type`` is not set or is unrecognized.
        """
        if self.fit_type is None:
            raise ValueError(
                "fit_type must be set before calling fit(): "
                "'point_lens' or 'binary_lens'."
            )

        self.planned_steps = self._build_remaining_steps()
        if len(self.completed_steps) > 0:
            logger.info(
                "\nCompleted steps \n%s\n",
                "\n".join(
                    [
                        "{0}: {1}".format(step.stage, step.name)
                        for step in self.completed_steps
                    ]
                ),
            )

        if len(self.planned_steps) == 0:
            logger.info("\nNo workflow steps to execute.\n")
        else:
            logger.info(
                "\nPlanned workflow: \n%s\n",
                "\n".join(
                    [
                        "{0}:{1}".format(step.stage, step.name)
                        for step in self.planned_steps
                    ]
                ),
            )

        i = 0
        while i < len(self.planned_steps):
            step = self.planned_steps[i]

            # debugging:
            logger.info(f"\nRunning step: {step.stage}:{step.name}")
            # logger.info(f'DEBUG remaining steps: %s', [f'{s.stage}:{s.name}' for s in self.planned_steps])

            if self.stop_before is not None and self._matches_stop_point(
                self.stop_before, step, mode="before"
            ):
                logger.info("Stopping before step '%s'.", step.name)
                break

            if self.dry_run:
                logger.info("[dry_run] Would execute: %s", step.name)
            else:
                step.run()
                if step.status == StepStatus.FAILED:
                    break

                logger.info(f"Step completed: {step.stage}:{step.name}")
                self.completed_steps.append(step)

                # Insert any dynamically generated follow-up steps
                if isinstance(step.result, list) and step.result:
                    # logger.info('DEBUG: steps to insert: %s', [f'{s.stage}:{s.name}' for s in step.result])

                    for j, dynamic_step in enumerate(step.result):
                        self.planned_steps.insert(i + 1 + j, dynamic_step)

                self._save_restart_state()

            # Lookahead uses the queue *after* any dynamic insertions
            remaining = self.planned_steps[i + 1 :]
            if self.stop_after is not None and self._matches_stop_point(
                self.stop_after, step, mode="after", remaining_steps=remaining
            ):
                logger.info("\nStopping after step '%s'.", step.name)
                break

            i += 1

        if (not self.dry_run) and self._output_config is not None:
            if self._output_config.save_table:
                for fmt in self._output_config.table_formats:
                    table_str = self.make_ulens_table(table_type=fmt)
                    path = self._output_config.table_path(fmt)
                    path.write_text(table_str)
                    logger.info("\nSaved %s results table to %s.", fmt, path)

            if self._output_config.save_exozippy_init:
                path = self._output_config.exozippy_init_path()
                with open(path, "w") as f:
                    json.dump(self.initialize_exozippy(), f, indent=4)
                logger.info("\nSaved EXOZIPPy init data to %s.", path)

            if self._output_config.save_plots:
                self._plot_best_fit_event()

        return self.all_fit_results

    def _build_remaining_steps(self) -> list[WorkflowStep]:
        """
        Build the planned step queue, skipping already completed steps.

        Steps are identified by ``(name, stage)`` pairs to allow the same
        step name to appear in multiple stages (e.g. ``renormalize_datasets``
        appears in both ``'renormalize'`` and ``'check_binary_renorm'``).

        Returns
        -------
        list of WorkflowStep
            Ordered steps remaining to be executed.

        Raises
        ------
        ValueError
            If ``fit_type`` is unrecognized.
        """
        completed_ids = {
            (step.name, step.stage) for step in self.completed_steps
        }
        for name, stage in completed_ids:
            if f"{stage}:{name}" in (self.stop_before, self.stop_after):
                return []

        if self.fit_type == "point_lens":
            all_steps = self._build_point_lens_steps()
        elif self.fit_type == "binary_lens":
            all_steps = self._build_binary_lens_steps()
        else:
            raise ValueError(
                f"Unknown fit_type {self.fit_type!r}. "
                "Expected 'point_lens' or 'binary_lens'."
            )

        # TODO: ADD something to the workflow that checks for negative fluxes and refits with fb=0 if True.
        if self._initial_entry_point is not None:
            entry_idx = next(
                (
                    i
                    for i, s in enumerate(all_steps)
                    if s.name == self._initial_entry_point
                ),
                0,
            )
            all_steps = all_steps[entry_idx:]

        return [s for s in all_steps if (s.name, s.stage) not in completed_ids]

    def _build_common_point_lens_steps(
        self, include_renormalize: bool = True
    ) -> list[WorkflowStep]:
        """
        Build the point-lens steps shared by both the point-lens and
        binary-lens workflows.

        Includes event search, static fitting, parallax fitting, and
        (when enabled) renormalization.  Does not include the parallax
        grid search.

        Returns
        -------
        list of WorkflowStep
        """
        steps: list[WorkflowStep] = []
        steps.extend(self._build_event_search_steps())
        steps.extend(self._build_static_fit_steps())
        steps.extend(self._build_parallax_point_lens_steps())

        if self.renormalize_errors and include_renormalize:
            steps.extend(self._build_renormalize_steps())
        return steps

    def _build_point_lens_steps(self) -> list[WorkflowStep]:
        """
        Build the step list for a point-lens workflow.

        Returns
        -------
        list of WorkflowStep
        """
        steps = self._build_common_point_lens_steps()
        if self.parallax_point_lens == "grid":
            steps.extend(self._build_parallax_grid_steps())

        return steps

    def _build_binary_lens_steps(self) -> list[WorkflowStep]:
        """
        Build the step list for a binary-lens workflow.

        Returns
        -------
        list of WorkflowStep
        """

        if self._initial_entry_point != "search_for_anomaly":
            # Renormalization is deferred until AFTER the anomaly search:
            # its outlier rejection protects the anomaly window via
            # intermediate_results.anomaly_lc_params, which does not exist
            # yet at the common-steps position -- running it there flagged
            # the planetary anomaly itself as bad data (the points survive
            # into dataset.bad and the exozippy-init excluded_points).
            steps = self._build_common_point_lens_steps(
                include_renormalize=False
            )
        else:
            steps: list[WorkflowStep] = []

        steps.extend(self._build_anomaly_search_steps())
        if (
            self.renormalize_errors
            and self._initial_entry_point != "search_for_anomaly"
        ):
            steps.extend(self._build_renormalize_steps())
        steps.extend(self._build_binary_fit_steps())
        if self.renormalize_errors:
            steps.extend(self._build_check_binary_renorm_steps())
        if self.parallax_binary_lens == True:
            steps.extend(self._build_parallax_binary_lens_steps())

        return steps

    def _build_parallax_binary_lens_steps(self) -> list[WorkflowStep]:
        """
        Build one WorkflowStep per parallax branch.

        Returns
        -------
        list of WorkflowStep
        """
        if self.parallax_binary_lens is False:
            return []

        steps = []
        for key in self._iter_parallax_point_lens_keys():
            branch = key.parallax_branch
            name = f"fit_parallax_{branch.value.lower()}"
            steps.append(
                WorkflowStep(
                    name=name,
                    func=lambda b=branch: self.fit_parallax(branch=b),
                    stage="fit_binary_lens_parallax",
                    description=f"Fit parallax model for branch {branch.value}",
                )
            )
        return steps

    def _build_event_search_steps(self) -> list[WorkflowStep]:
        return [
            WorkflowStep(
                name="run_event_search",
                func=self.run_event_search,
                stage="event_search",
                description="Run EventFinder grid search",
            ),
        ]

    def _build_static_fit_steps(self) -> list[WorkflowStep]:
        """
        Build steps covering the static (no-parallax) fit.

        If a PSPL record already exists in ``all_fit_results`` (e.g. from
        user-supplied ``initial_results``), its params are passed to
        ``fit_static_point_source_point_lens`` as the initial seed.

        Includes an FSPL step when ``self.finite_source_point_lens`` is not False.

        Returns
        -------
        list of WorkflowStep
        """
        if self._initial_entry_point == "fit_static_point_source_point_lens":
            static_pspl_key = FitKey(
                lens_type=LensType.POINT,
                source_type=SourceType.POINT,
                parallax_branch=ParallaxBranch.NONE,
                lens_orb_motion=LensOrbMotion.NONE,
            )
            existing = self.all_fit_results.get(static_pspl_key)
            pspl_seed = existing.params
            steps = []
        else:
            steps = [
                WorkflowStep(
                    name="estimate_point_lens_parameters",
                    func=self.estimate_point_lens_parameters,
                    stage="fit_static_point_lens",
                    description="Estimate point-lens parameters from EF grid result",
                )
            ]
            pspl_seed = None

        steps.append(
            WorkflowStep(
                name="fit_static_point_source_point_lens",
                func=lambda p=pspl_seed: (
                    self.fit_static_point_source_point_lens(initial_params=p)
                ),
                stage="fit_static_point_lens",
                description="Fit static PSPL model",
            ),
        )

        if self.finite_source_point_lens:
            steps.append(
                WorkflowStep(
                    name="fit_static_finite_source_point_lens",
                    func=self.fit_static_finite_source_point_lens,
                    stage="fit_static_point_lens",
                    description="Fit static FSPL model",
                )
            )
        return steps

    def _build_parallax_point_lens_steps(self) -> list[WorkflowStep]:
        """
        Build one WorkflowStep per parallax branch.

        Returns
        -------
        list of WorkflowStep
        """
        if self.parallax_point_lens is False:
            return []

        steps = []
        for key in self._iter_parallax_point_lens_keys():
            branch = key.parallax_branch
            name = f"fit_parallax_{branch.value.lower()}"
            steps.append(
                WorkflowStep(
                    name=name,
                    func=lambda b=branch: self.fit_parallax(branch=b),
                    stage="fit_point_lens_parallax",
                    description=f"Fit parallax model for branch {branch.value}",
                )
            )
        return steps

    def _build_renormalize_steps(
        self,
        stage: str = "renormalize",
    ) -> list[WorkflowStep]:
        """
        Build steps covering error renormalization.

        Parameters
        ----------
        stage : str
            Stage name to assign to the returned steps.  Defaults to
            ``'renormalize'``.  Pass ``'check_binary_renorm'`` when these
            steps are inserted dynamically after the binary fit.

        Returns
        -------
        list of WorkflowStep
        """
        return [
            WorkflowStep(
                name="renormalize_datasets",
                func=self.renormalize_datasets,
                stage=stage,
                description=(
                    "Remove outliers and compute per-dataset error "
                    "rescaling factors"
                ),
            ),
            WorkflowStep(
                name="refit_all",
                func=self.refit_all,
                stage=stage,
                description="Refit all stored fits with updated error normalization",
            ),
        ]

    def _build_anomaly_search_steps(self) -> list[WorkflowStep]:
        """
        Build steps covering the AnomalyFinder grid search.

        Returns
        -------
        list of WorkflowStep
        """
        return [
            # WorkflowStep( # This step doesn't do anything.
            #    name='select_best_point_lens_model',
            #    func=self.select_best_point_lens_model,
            #    stage='search_for_anomaly',
            #    description='Select the best point-lens model for anomaly search',
            # ),
            WorkflowStep(
                name="compute_point_lens_residuals",
                func=self.compute_point_lens_residuals,
                stage="search_for_anomaly",
                description="Compute residuals from best point-lens model",
            ),
            WorkflowStep(
                name="run_anomaly_search",
                func=self.run_anomaly_search,
                stage="search_for_anomaly",
                description="Run AnomalyFinder grid search",
            ),
            WorkflowStep(
                name="get_anomaly_light_curve_parameters",
                func=self.get_anomaly_light_curve_parameters,
                stage="search_for_anomaly",
                description="Measure observable anomaly properties",
            ),
            WorkflowStep(
                name="classify_anomaly",
                func=self.classify_anomaly,
                stage="search_for_anomaly",
                description="Classify the anomaly type",
            ),
        ]

    def _build_binary_fit_steps(self) -> list[WorkflowStep]:
        return [
            WorkflowStep(
                name="estimate_binary_lens_parameters",
                func=self.estimate_binary_lens_parameters,
                stage="fit_binary_lens",
                description="Estimate binary-lens parameters from AF grid result",
            ),
            WorkflowStep(
                name="fit_binary_lens_models",
                func=self.fit_binary_lens_models,
                stage="fit_binary_lens",
                description=(
                    "Fit binary-lens models; may return dynamic follow-up steps"
                ),
            ),
        ]

    def _build_check_binary_renorm_steps(self) -> list[WorkflowStep]:
        return [
            WorkflowStep(
                name="check_needs_renorm",
                func=self.check_needs_renorm,
                stage="check_binary_renorm",
                description="Check whether binary fits require renormalization",
            ),
        ]

    def _build_parallax_grid_steps(self) -> list[WorkflowStep]:
        """
        Build one WorkflowStep per parallax branch for the grid search.

        Always yields two steps (U0_PLUS and U0_MINUS), regardless of the
        number of observing locations.

        Returns
        -------
        list of WorkflowStep
        """
        steps = [
            WorkflowStep(
                name="run_parallax_grids",
                func=self.run_parallax_grids,
                stage="parallax_grids",
                description="Run both parallax grid searches",
            )
        ]
        return steps

    def _step_matches_stop_value(
        self,
        stop_value: str,
        step: WorkflowStep,
    ) -> bool:
        """
        Return True if *step* matches a stop value string.

        Handles both ``'stage'`` and ``'stage:step'`` syntax without any
        mode-specific lookahead logic.

        Parameters
        ----------
        stop_value : str
            A stage name or ``stage:step`` string.
        step : WorkflowStep
            The step to test.

        Returns
        -------
        bool
        """
        if ":" in stop_value:
            stage_name, step_name = stop_value.split(":", 1)
            return step.stage == stage_name and step.name == step_name
        return step.stage == stop_value

    def _matches_stop_point(
        self,
        stop_value: str,
        step: WorkflowStep,
        mode: str,
        remaining_steps: Optional[list[WorkflowStep]] = None,
    ) -> bool:
        if not self._step_matches_stop_value(stop_value, step):
            return False

        if ":" in stop_value:
            return True  # stage:step — exact match, no lookahead needed

        # Stage-only: mode-specific logic
        if mode == "before":
            return not any(s.stage == step.stage for s in self.completed_steps)
        # mode == 'after'
        if remaining_steps is None:
            return True
        return not any(s.stage == step.stage for s in remaining_steps)

    # ------------------------------------------------------------------
    # Step action methods
    # ------------------------------------------------------------------

    def run_event_search(self) -> None:
        """
        Run the EventFinder grid.

        Stores the best grid point in
        ``self.intermediate_results.best_ef_grid_point``.
        """
        ef_grid = EventFinderGridSearch(datasets=self.datasets)
        ef_grid.run()

        if self._output_config is not None and self._output_config.save_plots:
            fig = ef_grid.plot()
            fig.savefig(self._output_config.plot_path("ef_grid"))
            plt.close(fig)
            logger.info(
                "Saved EF grid plot to %s.",
                self._output_config.plot_path("ef_grid"),
            )

        logger.info("Best EF grid point: %s", ef_grid.best)
        self.intermediate_results.best_ef_grid_point = ef_grid.best

    def estimate_point_lens_parameters(self) -> None:
        """
        Estimate point-lens parameters from the EventFinder grid result.

        Stores estimates in ``self.intermediate_results.estimate_point_lens_parameters``.
        """
        est = get_PSPL_params(
            self.intermediate_results.best_ef_grid_point,
            self.datasets,
        )
        logger.info("Estimated point-lens params: %s", est)
        self.intermediate_results.estimate_point_lens_parameters = est

    def fit_static_point_source_point_lens(self, initial_params=None) -> None:
        """
        Fit a static PSPL model.

        Parameters
        ----------
        initial_params : dict, optional
            Starting parameter values.  If None, uses
            ``self.intermediate_results.estimate_point_lens_parameters``.

        Notes
        -----
        Stores the resulting ``FitRecord`` in ``self.all_fit_results``.
        """
        if initial_params is None:
            initial_params = (
                self.intermediate_results.estimate_point_lens_parameters
            )

        fitter = SFitFitter(
            initial_model_params=initial_params,
            datasets=self.datasets,
            **self._get_fitter_kwargs(source_type=SourceType.POINT),
        )
        fitter.run()

        if fitter.success:
            logger.info("Static PSPL: %s", fitter.best)
            logger.info("    sigmas:  %s", list(fitter.results.sigmas))

            key = FitKey(
                lens_type=LensType.POINT,
                source_type=SourceType.POINT,
                parallax_branch=ParallaxBranch.NONE,
                lens_orb_motion=LensOrbMotion.NONE,
            )
            self.all_fit_results.set(
                FitRecord.from_full_result(
                    model_key=key,
                    full_result=MMEXOFASTFitResults(fitter),
                    renorm_factors=self.renorm_factors,
                    fixed=False,
                )
            )
        else:
            logger.warning(
                "Static PSPL fit failed: %s", fitter.get_diagnostic_str()
            )
            raise RuntimeError(
                "Static PSPL fit failed. Cannot proceed to FSPL or parallax fitting."
            )

    def fit_static_finite_source_point_lens(self, initial_params=None) -> None:
        """
        Fit a static FSPL model.

        Parameters
        ----------
        initial_params : dict, optional
            Starting parameter values.  If None, seeds from the PSPL
            result in ``self.all_fit_results`` with ``rho = 0.05`` (giant source)
        Notes
        -----
        Stores the resulting ``FitRecord`` in ``self.all_fit_results``.
        Requires a static PSPL result to already exist.

        Raises
        ------
        RuntimeError
            If no static PSPL fit exists to seed from.
        """
        if initial_params is None:
            pspl_key = FitKey(
                lens_type=LensType.POINT,
                source_type=SourceType.POINT,
                parallax_branch=ParallaxBranch.NONE,
                lens_orb_motion=LensOrbMotion.NONE,
            )
            pspl_record = self.all_fit_results.get(pspl_key)
            if pspl_record is None:
                raise RuntimeError(
                    "A static PSPL fit must exist before fitting FSPL."
                )
            initial_params = dict(pspl_record.params)
            initial_params["rho"] = ParameterEstimator(
                initial_params, limit=self.source_type
            ).get_rho()

        if self._check_FSPL_condition(initial_params):
            mag_methods = self._get_magnification_methods_FSPL(
                initial_params=initial_params
            )
            fitter_kwargs = self._get_fitter_kwargs(
                source_type=SourceType.FINITE
            )
            fitter_kwargs["mag_methods"] = mag_methods
            fitter = SFitFitter(
                initial_model_params=initial_params,
                datasets=self.datasets,
                **fitter_kwargs,
            )
            fitter.run()
            if fitter.success:
                logger.info("Static FSPL: %s", fitter.best)
                logger.info("    sigmas:  %s", list(fitter.results.sigmas))

                key = FitKey(
                    lens_type=LensType.POINT,
                    source_type=SourceType.FINITE,
                    parallax_branch=ParallaxBranch.NONE,
                    lens_orb_motion=LensOrbMotion.NONE,
                )
                self.all_fit_results.set(
                    FitRecord.from_full_result(
                        model_key=key,
                        full_result=MMEXOFASTFitResults(fitter),
                        renorm_factors=self.renorm_factors,
                        fixed=False,
                    )
                )
            else:
                logger.warning(
                    "Static FSPL fit failed: %s", fitter.get_diagnostic_str()
                )
                self.finite_source_point_lens = False

    def _get_magnification_methods_FSPL(self, initial_params=None):
        """
        Get the magnification methods for FSPL model.
        """
        if self.mag_methods is not None:
            for mag_method in self.mag_methods:
                if (
                    not isinstance(mag_method, (float, int))
                    and mag_method != "finite_source_uniform_Gould94"
                ):
                    raise ValueError(
                        "Invalid magnification method for FSPL. Only 'finite_source_uniform_Gould94' is alowed by MulensModel"
                    )
            return self.mag_methods

        if initial_params is None:
            return "finite_source_uniform_Gould94"

        t_0 = initial_params["t_0"]
        t_E = initial_params["t_E"]
        return [
            t_0 - 0.5 * t_E,
            "finite_source_uniform_Gould94",
            t_0 + 0.5 * t_E,
        ]

    def _check_FSPL_condition(self, initial_params):
        """
        Check if the FSPL condition is set and satisfied.
        """
        if isinstance(self.finite_source_point_lens, bool):
            return self.finite_source_point_lens
        elif isinstance(self.finite_source_point_lens, list):
            param_name = self.finite_source_point_lens[0]
            function = self.finite_source_point_lens[1]
            limit = self.finite_source_point_lens[2]
            value = initial_params[param_name]
            if function(value, limit):
                logger.info(
                    "FSPL condition satisfied (%s=%.4f %s %s); enabling FSPL fit.",
                    param_name,
                    value,
                    function.__name__,
                    limit,
                )
                return True
            else:
                logger.info(
                    "FSPL condition not satisfied (%s=%.4f %s %s); disabling FSPL fit.",
                    param_name,
                    value,
                    function.__name__,
                    limit,
                )
                self.finite_source_point_lens = False
                return False
        else:
            raise ValueError(
                "FSPL parameter must be a boolean or a string of the form 'u_0<value', 'u_0>value',"
                + " 't_E<value', or 't_E>value'."
            )

    def fit_parallax(self, branch=None) -> None:
        """
        Fit a parallax model for the given branch.

        Parameters
        ----------
        branch : mmexo.ParallaxBranch, optional
            Which parallax branch to fit.  If None, fits all branches
            appropriate for the current data via
            ``_iter_parallax_point_lens_keys()``.

        Notes
        -----
        Stores each resulting ``FitRecord`` in ``self.all_fit_results``.
        """
        if branch is not None:
            source_type = (
                SourceType.FINITE
                if self.finite_source_point_lens
                else SourceType.POINT
            )
            keys = [
                FitKey(
                    lens_type=LensType.POINT,
                    source_type=source_type,
                    parallax_branch=branch,
                    lens_orb_motion=LensOrbMotion.NONE,
                )
            ]
        else:
            keys = list(self._iter_parallax_point_lens_keys())

        for key in keys:
            seed = self._get_parallax_seed_params(key)
            result = self._do_parallax_fit(seed, source_type=key.source_type)
            if result is not None:
                self.all_fit_results.set(
                    FitRecord.from_full_result(
                        model_key=key,
                        full_result=result,
                        renorm_factors=self.renorm_factors,
                        fixed=False,
                    )
                )
                logger.info(
                    "Parallax %s: chi2=%.2f",
                    key.parallax_branch.value,
                    result.chi2,
                )

    def renormalize_datasets(self) -> None:
        """
        Run outlier rejection and compute per-dataset error rescaling factors.

        Notes
        -----
        Datasets already present in ``self.renorm_factors`` are skipped.
        Combines logic from the old ``_remove_outliers_and_calc_errfacs``
        and ``_apply_error_renormalization``.  Rebuilds ``fix_blend_flux_map``
        and ``fix_source_flux_map`` after replacing dataset objects.

        Two regions are **protected from outlier rejection**: the anomaly
        window ``[t_pl - dt, t_pl + dt]`` (when
        ``self.intermediate_results.anomaly_lc_params`` is set), and every
        point the reference model magnifies above ``PEAK_PROTECTION_MAG``.
        Neither the anomaly signal nor the reference model's mismatch near
        peak should be treated as a systematic and removed; see
        :meth:`_build_protected_mask`.
        """
        event = self._build_renorm_event()
        new_datasets = []

        for i, dataset in enumerate(self.datasets):
            label = dataset.plot_properties["label"]
            if label in self.renorm_factors:
                new_datasets.append(dataset)
                continue
            new_ds, errfac = self._process_single_dataset(i, event)
            new_datasets.append(new_ds)
            self.renorm_factors[label] = errfac

        self.datasets = new_datasets
        self.fix_blend_flux_map = self._map_label_dict_to_datasets(
            self.fix_blend_flux
        )
        self.fix_source_flux_map = self._map_label_dict_to_datasets(
            self.fix_source_flux
        )
        self.event_config = self._build_event_config()

    def _build_renorm_event(self):
        """
        Build and fit the reference event used throughout renormalization.

        Selects the best available model via ``select_best_model``, constructs
        a MulensModel event from the current datasets, and performs an initial
        flux fit.

        Returns
        -------
        MulensModel.Event
        """
        fit = self.select_best_model()
        model = fit.full_result.fitter.get_model()
        logger.info(
            "Renormalizing using: %s", model_key_to_label(fit.model_key)
        )
        event = self.event_config.build(model=model, datasets=self.datasets)
        event.fit_fluxes()
        return event

    def _build_protected_mask(self, dataset, event, index) -> np.ndarray:
        """
        Return a boolean mask of points that must survive outlier rejection.

        Two regions are protected, and the mask is their union:

        * The anomaly window ``[t_pl - dt, t_pl + dt]`` from
          ``intermediate_results.anomaly_lc_params`` (skipped if unset) --
          the reference (point-lens) model cannot describe the planetary
          anomaly, so those points would otherwise be the first to be
          flagged as outliers.
        * Every point where the reference model's magnification exceeds
          ``PEAK_PROTECTION_MAG``. The reference is a point-lens model, and
          wherever it magnifies appreciably the true (binary, finite-source)
          light curve can deviate by far more than the rejection threshold
          at space-photometry precision -- on DC18 event 128 the >3-sigma
          mismatch extended to |t - t_0| ~ 15 d and rejection consumed the
          entire peak. Protection follows the model rather than a fixed
          time window, so it scales with each event.

        Parameters
        ----------
        dataset : MulensModel.MulensData
        event : MulensModel.Event
            Reference event, already flux-fitted.
        index : int
            Index of *dataset* within ``event.datasets``.

        Returns
        -------
        np.ndarray of bool
            Same length as ``dataset.time``.
        """
        label = dataset.plot_properties["label"]
        mask = np.zeros(len(dataset.time), dtype=bool)

        params = self.intermediate_results.anomaly_lc_params
        if params is not None:
            # The anomaly epoch is 't_pl'; 't_0' in the same dict is the
            # underlying point-lens peak. Reading t_0 here centered the
            # protection window on the event peak and left the planetary
            # anomaly itself exposed to outlier rejection.
            t_pl = params.get("t_pl", params.get("t_0"))
            dt = params.get("dt")
            if t_pl is not None and dt is not None:
                t0, t1 = t_pl - dt, t_pl + dt
                logger.info(
                    "Anomaly window protection active: [%.6f, %.6f]", t0, t1
                )
                mask = (dataset.time >= t0) & (dataset.time <= t1)
                n = int(np.sum(mask & dataset.good))
                if n > 0:
                    logger.info(
                        "  %s: %d point(s) inside anomaly window are protected.",
                        label,
                        n,
                    )

        mag = np.asarray(event.fits[index].get_data_magnification(bad=True))
        if mag.ndim > 1:
            # Multi-source: protect where ANY source is magnified.
            mag = mag.max(axis=0)
        peak = mag > self.PEAK_PROTECTION_MAG
        n_peak = int(np.sum(peak & dataset.good & ~mask))
        if n_peak > 0:
            logger.info(
                "  %s: %d additional point(s) with model magnification > "
                "%.2f are protected.",
                label,
                n_peak,
                self.PEAK_PROTECTION_MAG,
            )

        return mask | peak

    def _process_single_dataset(
        self, i: int, event
    ) -> tuple[MulensModel.MulensData, float]:
        """
        Remove outliers from dataset *i*, compute its error rescaling factor,
        and return a rescaled copy.

        ``compute_errfac`` and ``remove_outliers`` are defined as closures so
        they share ``dataset``, ``protected_mask``, and ``n_params`` without
        argument threading.  ``remove_outliers`` mutates ``dataset.bad`` in
        place and has no meaningful use in isolation.

        The scatter estimate in ``compute_errfac`` is anchored to
        unprotected good points only (outside both the anomaly window and
        the magnified peak), preventing the planetary signal and the
        reference model's near-peak mismatch from inflating the error bars
        of the whole dataset.

        Parameters
        ----------
        i : int
            Index into ``event.datasets``.
        event : MulensModel.Event
            Reference event, already flux-fitted.

        Returns
        -------
        tuple[MulensModel.MulensData, float]
            ``(rescaled_dataset, errfac)``
        """
        dataset = event.datasets[i]
        n_params = len(event.model.parameters.as_dict())
        protected_mask = self._build_protected_mask(dataset, event, i)

        def compute_errfac(res: np.ndarray, err: np.ndarray) -> float:
            """Chi2-based scatter estimate using non-anomaly good points only."""
            clean = dataset.good & ~protected_mask
            dof = int(np.sum(clean)) - n_params
            return (
                float(np.sqrt(np.sum((res[clean] / err[clean]) ** 2) / dof))
                if dof > 0
                else 1.0
            )

        def solve_fluxes(mag, flux, err, good):
            """Chi2-minimizing (f_s, f_b) on the good points.

            Weighted linear regression in flux space -- the same problem
            ``event.fit_fluxes()`` solves for a single-source model --
            honoring a fixed source or blend flux when the event was built
            with one.
            """
            w = 1.0 / err[good] ** 2
            a, f = mag[good], flux[good]
            # MulensModel convention: a value of False means "fit this flux"
            # (the dicts are populated with False per dataset), while a
            # NUMBER -- including 0.0 -- means "fix it there". Identity
            # comparison keeps a legitimate fix at 0.0 working.
            fs_fix = event.fix_source_flux.get(dataset)
            fb_fix = event.fix_blend_flux.get(dataset)
            if fs_fix is False:
                fs_fix = None
            if fb_fix is False:
                fb_fix = None
            if fs_fix is not None and fb_fix is not None:
                return float(fs_fix), float(fb_fix)
            if fs_fix is not None:
                return float(fs_fix), float(
                    np.sum(w * (f - fs_fix * a)) / np.sum(w)
                )
            if fb_fix is not None:
                return (
                    float(np.sum(w * a * (f - fb_fix)) / np.sum(w * a**2)),
                    float(fb_fix),
                )
            s_aa = np.sum(w * a**2)
            s_a = np.sum(w * a)
            s_1 = np.sum(w)
            s_af = np.sum(w * a * f)
            s_f = np.sum(w * f)
            det = s_aa * s_1 - s_a**2
            fs = (s_af * s_1 - s_f * s_a) / det
            fb = (s_aa * s_f - s_a * s_af) / det
            return float(fs), float(fb)

        def remove_outliers() -> None:
            """Iteratively flag the worst outlier until none exceed max_sig.

            Semantics are IDENTICAL to the original loop (one worst point
            per iteration, ``errfac``/``max_sig`` re-derived after every
            rejection -- the criterion is self-referential and can have
            multiple fixed points, so batching rejections would change the
            result). What changed is the cost: the reference model is FIXED
            here (only the fluxes are refit; the nonlinear refit happens
            once afterward, in ``refit_all``), so the model magnification
            is computed ONCE and each iteration is a closed-form linear
            flux solve -- instead of ``fit_fluxes``/``get_residuals``
            recomputing the full (binary-lens) magnification for every
            rejected point, which made dense light curves (e.g. Roman DC18
            W149, 38k epochs, hundreds of rejections) take hours.
            Multi-source models report per-source magnifications; the
            closed-form solve below is single-source, so those fall back
            to the original loop.
            """
            event.fit_fluxes()
            mag = event.fits[i].get_data_magnification(bad=True)
            if np.ndim(mag) != 1:
                remove_outliers_slow()
                return
            flux, err = dataset.flux, dataset.err_flux
            bad = dataset.bad.copy()
            while True:
                good = ~bad
                dof = int(np.sum(good)) - n_params
                if dof <= 0:
                    break
                max_sig = max(np.sqrt(2.0) * erfcinv(1.0 / dof), 3.0)
                fs, fb = solve_fluxes(mag, flux, err, good)
                res = flux - (fs * mag + fb)
                clean = good & ~protected_mask
                cdof = int(np.sum(clean)) - n_params
                errfac = (
                    float(
                        np.sqrt(np.sum((res[clean] / err[clean]) ** 2) / cdof)
                    )
                    if cdof > 0
                    else 1.0
                )
                sigma = np.abs(res / (err * errfac))
                candidate = good & ~protected_mask
                if not np.any(sigma[candidate] > max_sig):
                    break
                worst = np.where(candidate)[0][np.argmax(sigma[candidate])]
                bad[worst] = True
            if np.any(bad != dataset.bad):
                dataset.bad = bad

        def remove_outliers_slow() -> None:
            """Original one-worst-point-per-iteration loop (multi-source
            fallback; kept verbatim)."""
            bad_index: Any = -1
            while bad_index is not None:
                event.fit_fluxes()
                dof = int(np.sum(dataset.good)) - n_params
                if dof <= 0:
                    break
                max_sig = max(np.sqrt(2.0) * erfcinv(1.0 / dof), 3.0)
                res, err = event.fits[i].get_residuals(
                    phot_fmt="flux", bad=True
                )
                sigma = np.abs(res / (err * compute_errfac(res, err)))
                candidate_mask = dataset.good & ~protected_mask
                if np.any(sigma[candidate_mask] > max_sig):
                    candidates = np.where(candidate_mask)[0]
                    bad_idx = candidates[[np.argmax(sigma[candidate_mask])]]
                    new_bad = dataset.bad.copy()
                    new_bad[bad_idx] = True
                    dataset.bad = new_bad
                    bad_index = bad_idx
                else:
                    bad_index = None

        remove_outliers()
        event.fit_fluxes()
        res, err = event.fits[i].get_residuals(phot_fmt="flux", bad=True)
        errfac = compute_errfac(res, err)
        logger.info(
            "  %s: errfac=%.3f", dataset.plot_properties["label"], errfac
        )
        return self._recreate_dataset(dataset, errfac), errfac

    def _recreate_dataset(
        self, dataset: MulensModel.MulensData, errfac: float
    ) -> MulensModel.MulensData:
        """
        Return a new ``MulensData`` instance identical to *dataset* but with
        error bars scaled by *errfac*.

        Constructor keyword arguments are copied from *dataset* via
        introspection of ``MulensData.__init__``, excluding ``data_list``,
        ``good``, ``phot_fmt``, and ``file_name``, which are either supplied
        explicitly or are not applicable to an in-memory copy.

        Parameters
        ----------
        dataset : MulensModel.MulensData
        errfac : float

        Returns
        -------
        MulensModel.MulensData
        """
        sig = inspect.signature(MulensModel.MulensData.__init__)
        kwargs = {
            k: getattr(dataset, k)
            for k in sig.parameters
            if k not in ("self", "data_list", "good", "phot_fmt", "file_name")
            and hasattr(dataset, k)
        }
        return MulensModel.MulensData(
            data_list=[dataset.time, dataset.flux, errfac * dataset.err_flux],
            phot_fmt="flux",
            **kwargs,
        )

    def refit_all(self) -> None:
        """
        Refit all stored fits using updated error normalization.
        """
        logger.info("Refitting all stored models...")
        for key, fit_record in self.all_fit_results.items():
            fitter = fit_record.full_result.fitter
            fitter.datasets = self.datasets
            fitter.initial_model_params = fit_record.params
            fitter.run()
            self.all_fit_results.set(
                FitRecord.from_full_result(
                    model_key=key,
                    full_result=MMEXOFASTFitResults(fitter),
                    renorm_factors=self.renorm_factors,
                    fixed=False,
                )
            )
            logger.info(
                "%s: %s",
                model_key_to_label(key),
                fitter.best,
            )
            logger.info("    sigmas:  %s", list(fitter.results.sigmas))

            # TODO: Output plots if save_plots = True

    def _select_best_lens_model(self, lens_type: LensType) -> FitRecord:
        """
        Internal helper: return the best ``FitRecord`` for the given
        ``lens_type``, preferring the parallax model when its chi-squared
        improvement over the best static model exceeds 50.

        Parameters
        ----------
        lens_type : LensType

        Returns
        -------
        FitRecord

        Raises
        ------
        RuntimeError
            If no fits of the requested lens type are available, or if
            multiple incomplete records exist and none have chi-squared.
        """
        # TODO: ADD a THRESHOLDS parameter (dict) that can hold all the thresholds like DELTA_CHI2
        # so the user can control them.
        # TODO: ADD support for choosing between FSPL and PSPL models.
        DELTA_CHI2 = 50.0
        label = lens_type.name.lower()

        lens_fits = [
            rec
            for key, rec in self.all_fit_results.items()
            if key.lens_type == lens_type
        ]

        if not lens_fits:
            raise RuntimeError(f"No {label} fits found in all_fit_results.")

        if self._initial_entry_point == "search_for_anomaly":
            return lens_fits[0]

        complete_fits = [rec for rec in lens_fits if rec.chi2() is not None]

        # If no complete fits exist, return the sole available record
        # (e.g. a user-supplied initial guess without chi2).
        if not complete_fits:
            if len(lens_fits) > 1:
                raise RuntimeError(
                    f"No complete {label} fits found and more than one "
                    "incomplete record exists — cannot select best model."
                )
            return lens_fits[0]

        static_fits = [
            rec
            for key, rec in self.all_fit_results.items()
            if key.lens_type == lens_type
            and key.parallax_branch == ParallaxBranch.NONE
            and rec.chi2() is not None
        ]

        parallax_fits = [
            rec
            for key, rec in self.all_fit_results.items()
            if key.lens_type == lens_type
            and key.parallax_branch != ParallaxBranch.NONE
            and rec.chi2() is not None
        ]

        best_static = (
            min(static_fits, key=lambda r: r.chi2()) if static_fits else None
        )
        best_par = (
            min(parallax_fits, key=lambda r: r.chi2())
            if parallax_fits
            else None
        )

        if best_par is None:
            return best_static
        if best_static is None:
            return best_par
        if best_static.chi2() - best_par.chi2() > DELTA_CHI2:
            return best_par
        return best_static

    def select_best_point_lens_model(self) -> FitRecord:
        """
        Return the best point-lens ``FitRecord`` from ``all_fit_results``.

        Prefers the best parallax model when its chi-squared improvement
        over the best static model exceeds 50; otherwise returns the best
        static model.

        If no complete point-lens fits exist but exactly one point-lens
        record is present (e.g. a user-supplied initial guess), that
        record is returned directly regardless of chi-squared.

        Returns
        -------
        FitRecord

        Raises
        ------
        RuntimeError
            If no point-lens fits are available.
        """
        return self._select_best_lens_model(LensType.POINT)

    def select_best_binary_model(self) -> FitRecord:
        """
        Return the best binary-lens ``FitRecord`` from ``all_fit_results``.

        Prefers the best parallax model when its chi-squared improvement
        over the best static model exceeds 50; otherwise returns the best
        static model.

        If no complete binary fits exist but exactly one binary record is
        present (e.g. a user-supplied initial guess), that record is
        returned directly regardless of chi-squared.

        Returns
        -------
        FitRecord

        Raises
        ------
        RuntimeError
            If no binary fits are available.
        """
        return self._select_best_lens_model(LensType.BINARY)

    def select_best_model(self) -> FitRecord:
        """
        Return the overall best ``FitRecord`` across all lens types.

        Compares the best point-lens model against the best binary model.
        The binary model is preferred only when its chi-squared is lower
        than the point-lens model's by more than 20.  If no binary fits
        are available the best point-lens model is returned unconditionally.

        Returns
        -------
        FitRecord

        Raises
        ------
        RuntimeError
            If no point-lens fits are available.
        """
        DELTA_CHI2_BINARY = 20.0

        best_point = self.select_best_point_lens_model()

        try:
            best_binary = self.select_best_binary_model()
        except RuntimeError:
            return best_point

        # Guard against records that have no chi2 (e.g. initial guesses).
        point_chi2 = best_point.chi2()
        binary_chi2 = best_binary.chi2()

        if binary_chi2 is None:
            return best_point
        if point_chi2 is None:
            return best_binary

        if point_chi2 - binary_chi2 > DELTA_CHI2_BINARY:
            return best_binary
        return best_point

    def compute_point_lens_residuals(self) -> None:
        """
        Compute residuals from the best point-lens model.

        Notes
        -----
        Residuals are stored in ``self.residuals`` as a list of
        ``MulensData`` objects in flux format, for use by
        ``run_anomaly_search()``.
        """
        reference_fit = self.select_best_point_lens_model()
        reference_model = reference_fit.full_result.fitter.get_model()
        logger.info(
            "Calculating residuals relative to %s",
            model_key_to_label(reference_fit.model_key),
        )

        event = self.event_config.build(
            model=reference_model,
            datasets=self.datasets,
        )
        # TODO: There might be a problem for FSPL models
        event.fit_fluxes()

        self.residuals: list = []
        for i, dataset in enumerate(self.datasets):
            res, err = event.fits[i].get_residuals(phot_fmt="flux")
            self.residuals.append(
                MulensModel.MulensData(
                    [dataset.time, res, err],
                    phot_fmt="flux",
                    bandpass=dataset.bandpass,
                    ephemerides_file=dataset.ephemerides_file,
                )
            )

    def run_anomaly_search(self) -> None:
        """
        Run the AnomalyFinder grid.

        Stores the best grid point in
        ``self.intermediate_results.best_af_grid_point``.
        """
        af_grid = AnomalyFinderGridSearch(residuals=self.residuals)
        af_grid.run()
        if self._output_config is not None and self._output_config.save_plots:
            fig = af_grid.plot()

            fig.savefig(self._output_config.plot_path("af_grid"))
            plt.close(fig)
            logger.info(
                "Saved AF grid plot to %s.",
                self._output_config.plot_path("ef_grid"),
            )

        best = af_grid.best
        if best is None:
            # Not a numerical failure: the grid ran to completion and every
            # window was rejected for having too little data to fit.  That is
            # the statement "no anomaly is detectable here", and for a
            # binary-lens fit it is fatal -- every downstream step
            # (get_anomaly_light_curve_parameters, classify_anomaly,
            # estimate_binary_lens_parameters) reads best_af_grid_point and
            # the first of them subscripts it immediately.  Raising a named
            # error here says so once, in the vocabulary of the problem,
            # instead of letting None reach AnomalyPropertyEstimator as a
            # TypeError -- or, before this, letting nanargmax raise
            # "All-NaN slice encountered" from inside a property.
            raise NoAnomalyFoundError(
                "AnomalyFinder found no fittable grid point, so no anomaly "
                "can be characterized and a binary-lens fit cannot be "
                "seeded. No grid window held a coherent deviation from the "
                "point-lens fit -- check_successive requires MORE THAN THREE "
                "consecutive points at >= 2 sigma. This is a DETECTION "
                "threshold, not a sparse-data condition: the fit did not "
                "diverge and the cadence is very unlikely to be the problem. "
                "Consider fit_type='point_lens' for this event."
            )

        logger.info("Best AF grid point: %s", best)
        self.intermediate_results.best_af_grid_point = best

    def get_anomaly_light_curve_parameters(self):
        """
        Estimate anomaly properties from the AnomalyFinder grid
        result.

        Stores estimates in
        ``self.intermediate_results.anomaly_lc_params``.
        """
        best_pspl = self.select_best_point_lens_model()
        estimator = AnomalyPropertyEstimator(
            datasets=self.datasets,
            pspl_params=best_pspl.params,
            af_results=self.intermediate_results.best_af_grid_point,
            model_config=self.model_config,
            event_config=self.event_config,
        )
        params = estimator.get_anomaly_lc_parameters()
        logger.info("Estimated anomaly params: %s", params)
        self.intermediate_results.anomaly_lc_params = params

    def classify_anomaly(self) -> None:
        """
        Use estimated anomaly properties from the AnomalyFinder grid
        result to classify the anomaly.

        Stores estimates in
        ``self.intermediate_results.anomaly_type``.
        """
        classifier = AnomalyClassifier()
        self.intermediate_results.anomaly_type = classifier.classify(
            self.residuals, self.intermediate_results.anomaly_lc_params
        )
        logger.info(
            "Anomaly classified as anomaly_type = %s",
            self.intermediate_results.anomaly_type,
        )
        if self._output_config is not None and self._output_config.save_plots:
            fig = classifier.plot_bell_fits()
            if fig is not None:
                fig.savefig(
                    self._output_config.plot_path("anomaly_classifier_fits")
                )
                fig.show()
                plt.close(fig)
                logger.info(
                    "Saved anomaly classification fits plot to %s.",
                    self._output_config.plot_path("anomaly_classifier_fits"),
                )

    def estimate_binary_lens_parameters(self) -> None:
        """
        Estimate binary-lens parameters from the AnomalyFinder grid
        result.

        Stores estimates in
        ``self.intermediate_results.estimate_binary_lens_parameters``.
        """
        est_params = {}
        estimator_classes = None
        self._anomaly_classification_backward_compatibility()
        # TODO: Consider running all Estimators in all cases
        if self.intermediate_results.anomaly_type == "bump":
            estimator_classes = [
                WideUpperGridSearchEstimator,
                WideLowerGridSearchEstimator,
                CloseUpperGridSearchEstimator,
                CloseLowerGridSearchEstimator,
            ]
            # TODO: Implement checking for large vs. small rho solutions. Maybe add a second estimator?
        elif self.intermediate_results.anomaly_type == "caustic_crossing":
            estimator_classes = [
                WideAxisGridSearchEstimator,
                CloseAxisGridSearchEstimator,
                CloseUpperGridSearchEstimator,
                CloseLowerGridSearchEstimator,
            ]
        elif self.intermediate_results.anomaly_type == "dip":
            estimator_classes = [CloseAxisGridSearchEstimator]
        else:
            logger.info(
                "Binary params estimate not implemented for %s",
                self.intermediate_results.anomaly_type,
            )

        if estimator_classes is not None:
            for estimator_class in estimator_classes:
                estimator = estimator_class(
                    datasets=self.datasets,
                    params=self.intermediate_results.anomaly_lc_params,
                    model_config=self.model_config,
                    event_config=self.event_config,
                )
                # Overrides the class-level default on ParameterEstimator so
                # every BinaryLensParams it builds carries this tolerance.
                estimator.vbbl_accuracy = self.vbbl_accuracy
                estimator.run()

                class_name = estimator_class.__name__
                class_name = class_name.removesuffix("ParameterEstimator")
                class_name = class_name.removesuffix("GridSearchEstimator")

                params = estimator.binary_params
                logger.info(
                    "Estimated binary params (%s): %s",
                    class_name,
                    params.ulens,
                )
                logger.info("mag_methods: %s", params.mag_methods)
                est_params[class_name] = params

                if self.intermediate_results.anomaly_type in [
                    "dip",
                    "bump",
                    "caustic_crossing",
                ]:
                    if (
                        self.intermediate_results.anomaly_lc_params["u_0"]
                        < 0.05
                    ):
                        s_alt = estimator.get_binary_lens_params()
                        s_alt.ulens["s"] = 1.0 / s_alt.ulens["s"]
                    else:
                        s_alt = estimator.alternate_params

                    logger.info("Alternate solution: %s", s_alt.ulens)
                    est_params[class_name + "_alt"] = s_alt

                self.mag_methods = params.mag_methods
                self.mag_methods_parameters = params.mag_methods_parameters

        self.intermediate_results.estimate_binary_lens_parameters = est_params
        if (
            self._output_config is not None
        ) and self._output_config.save_plots:
            self._plot_initial_2L1S_guess()

    def _anomaly_classification_backward_compatibility(self):
        """
        Handle backward compatibility for anomaly classification (old: wide/close -> new: bump/dip).
        """
        if self.intermediate_results.anomaly_type == "close":
            self.intermediate_results.anomaly_type = "dip"
        if self.intermediate_results.anomaly_type == "wide":
            self.intermediate_results.anomaly_type = "bump"

    def fit_binary_lens_models(self) -> Optional[list[WorkflowStep]]:
        """
        Fit binary lens models.

        Returns
        -------
        list of WorkflowStep or None
            Dynamically generated follow-up steps, or None if no
            additional steps are required.
        """
        best_pspl = self.select_best_point_lens_model()
        pspl_chi2 = best_pspl.chi2()
        n_data = sum(np.sum(dataset.good) for dataset in self.datasets)
        logger.info(f"PL chi2: {pspl_chi2:.1f}, N_good: {n_data}")

        # TODO: Implement grid search for high-mag models. See gridsearches.BinaryGridSearch()
        # TODO: Separate models (key/param items) into separate steps.
        # TODO: Check for and implement point source binary lens models.

        for (
            key,
            params,
        ) in self.intermediate_results.estimate_binary_lens_parameters.items():
            model = self.model_config.build(
                parameters=params.ulens,
                magnification_methods=params.mag_methods,
                magnification_methods_parameters=params.mag_methods_parameters,
                default_magnification_method="point_source_point_lens",
            )
            event = self.event_config.build(
                model=model,
                datasets=self.datasets,
            )
            binary_chi2 = event.get_chi2()
            logger.info(f"{key} initial chi2: {binary_chi2:.1f}")
            if (pspl_chi2 - binary_chi2) * n_data / np.min(
                (binary_chi2, pspl_chi2)
            ) < 3.0:
                logger.info(
                    "Binary model does not improve chi2 enough, skipping."
                )
                # TODO: if model is "alt" try seeding from the fitted regular solution.
                continue

            # Do the fit
            anomaly_fitter = AnomalyFitter(
                datasets=self.datasets,
                initial_guess=params.ulens,
                anomaly_lc_params=self.intermediate_results.anomaly_lc_params,
                mag_methods=params.mag_methods,
                mag_methods_parameters=params.mag_methods_parameters,
                model_config=self.model_config,
                event_config=self.event_config,
                pool=self.pool,
            )
            logger.debug(f"initial sigmas: {anomaly_fitter.sigmas}")
            msg = anomaly_fitter.run()
            if msg is not None:
                logger.warning(msg)
                continue

            results = MMEXOFASTFitResults(anomaly_fitter)
            logger.info(f"Fitted params ({key}): {results.best}")
            logger.info(
                f"       sigmas ({key}): {results.get_sigmas_from_results()}"
            )

            # Save the results to all results
            fit_key = FitKey(
                lens_type=LensType.BINARY,
                source_type=SourceType.FINITE,  # TODO: In the future, want to allow for rho=0 fits.
                parallax_branch=ParallaxBranch.NONE,
                lens_orb_motion=LensOrbMotion.NONE,
                binary_model_type=key.replace("Planet", "").replace(
                    "Binary", ""
                ),
            )
            self.all_fit_results.set(
                FitRecord.from_full_result(
                    model_key=fit_key,
                    full_result=results,
                    renorm_factors=self.renorm_factors,
                    fixed=False,
                )
            )
            if (
                self._output_config is not None
            ) and self._output_config.save_plots:
                self._plot_event(
                    anomaly_fitter.get_event(),
                    suptitle=f"{key}: {anomaly_fitter.best['chi2']:.1f}\n{anomaly_fitter.get_event().model.parameters}",
                )
                path = self._output_config.plot_path(f"_{key}")
                plt.savefig(path)

        return None

    def check_needs_renorm(self) -> Optional[list[WorkflowStep]]:
        """
        Check whether renormalization is needed after binary fits.

        Returns
        -------
        list of WorkflowStep or None
            Dynamically generated renormalization steps with stage
            ``'check_binary_renorm'``, or None if renormalization is
            not required.
        """
        if self._needs_renormalization():
            logger.info(
                "Renormalization required after binary fit; inserting steps."
            )
            return self._build_renormalize_steps(stage="check_binary_renorm")

        return None

    def run_parallax_grids(self, branch=None) -> None:
        """
        Run a parallax grid search for the given branch.

        Parameters
        ----------
        branch : mmexo.ParallaxBranch, optional
            Which parallax branch to search.  If None, searches both
            U0_PLUS and U0_MINUS.
        """
        branches = (
            [ParallaxBranch.U0_PLUS, ParallaxBranch.U0_MINUS]
            if branch is None
            else [branch]
        )

        reference_fit = self.select_best_point_lens_model()
        static_params = (
            reference_fit.full_result.fitter.get_model().parameters.parameters
        )
        source_type = reference_fit.model_key.source_type

        grids: dict = {}
        for par_branch in branches:
            logger.info("Running parallax grid for %s.", par_branch.value)
            grid = ParallaxGridSearch(
                static_params,
                datasets=self.datasets,
                grid_params=self.PARALLAX_GRID_PARAMS_COARSE,
                fitter_kwargs=self._get_fitter_kwargs(source_type=source_type),
                skip_optimization=False,
                verbose=False,
            )
            grid.run(refine=True)
            grids[par_branch] = grid

            if (
                self._output_config is not None
                and self._output_config.save_grid_results
            ):
                path = self._output_config.grid_path(
                    f"piE_grid_{par_branch.value.lower()}"
                )
                grid.save_grid_points(path)
                logger.info("Saved grid results to %s.", path)

        if self._output_config is not None and self._output_config.save_plots:
            self._plot_piE_grid_search(grids)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _do_parallax_fit(
        self,
        params,
        source_type=None,
    ) -> MMEXOFASTFitResults:
        """
        Shared implementation invoked by ``fit_parallax()`` and
        ``run_parallax_grids()``.

        Parameters
        ----------
        params : dict
            Starting parameter dict for the parallax model.
        source_type : mmexo.SourceType, optional
            Source type for the fit.  If None, inferred from *params*:
            FINITE when ``'rho'`` or ``'t_star'`` are present, otherwise
            POINT.

        Returns
        -------
        mmexo.MMEXOFASTFitResults
            Fit results from the optimizer.
        """
        if source_type is None:
            source_type = (
                SourceType.FINITE
                if ("rho" in params or "t_star" in params)
                else SourceType.POINT
            )
        fitter_kwargs = self._get_fitter_kwargs(source_type=source_type)
        if source_type == SourceType.FINITE:
            mag_methods = self._get_magnification_methods_FSPL(
                initial_params=params
            )
            fitter_kwargs["mag_methods"] = mag_methods
        fitter = SFitFitter(
            initial_model_params=params,
            datasets=self.datasets,
            **fitter_kwargs,
        )
        try:
            fitter.run()
        except Exception as e:
            logger.info(
                "Parallax fit failed:\n{0}: {1}".format(type(e).__name__, e)
            )
            return None
        if fitter.success:
            logger.info("Parallax fit: %s", fitter.best)
            logger.info("      sigmas: %s", list(fitter.results.sigmas))
        else:
            logger.warning(" fit failed: %s", fitter.get_diagnostic_str())
            return None

        return MMEXOFASTFitResults(fitter)

    def _get_parallax_seed_params(self, key: FitKey) -> dict:
        """
        Return starting parameters for a parallax fit key.

        Priority
        --------
        1. Existing result for the same key.
        2. Another parallax branch result, sign-transformed via
           BRANCH_SIGNS.
        3. Static PSPL/FSPL with ``pi_E_N = pi_E_E = 0``.

        Parameters
        ----------
        key : mmexo.FitKey
            The parallax ``FitKey`` being started.

        Returns
        -------
        dict
            Initial parameter dict.

        Raises
        ------
        RuntimeError
            If no static point-lens result is available as a fallback.
        """
        BRANCH_SIGNS = {
            ParallaxBranch.U0_PLUS: (+1, +1),
            ParallaxBranch.U0_MINUS: (-1, -1),
            ParallaxBranch.U0_PP: (+1, +1),
            ParallaxBranch.U0_MM: (-1, -1),
            ParallaxBranch.U0_PM: (+1, -1),
            ParallaxBranch.U0_MP: (-1, +1),
        }

        # 1. Exact match
        existing = self.all_fit_results.get(key)
        if existing is not None:
            return dict(existing.params)

        # 2. Sign-transform from another parallax branch
        # "s" = sign, "tgt" = target
        su0_tgt, spi_tgt = BRANCH_SIGNS[key.parallax_branch]
        for other_key, other_rec in self.all_fit_results.items():
            if (
                other_key.lens_type == key.lens_type
                and other_key.source_type == key.source_type
                and other_key.parallax_branch in BRANCH_SIGNS
                and other_key.parallax_branch != key.parallax_branch
                and (
                    (other_rec.sigmas is None)
                    or (
                        np.abs(
                            other_rec.sigmas["pi_E_E"]
                            / other_rec.params["pi_E_E"]
                        )
                        < 0.5
                    )
                )
            ):
                # don't bother if the parallax isn't well constrained.
                su0_src, spi_src = BRANCH_SIGNS[other_key.parallax_branch]
                base = dict(other_rec.params)
                if "u_0" in base:
                    base["u_0"] *= su0_tgt / su0_src
                if "pi_E_N" in base:
                    base["pi_E_N"] *= spi_tgt / spi_src
                logger.debug(
                    "Seeding %s from %s (sign-transformed).",
                    key.parallax_branch.value,
                    other_key.parallax_branch.value,
                )
                return base

        # 3. Static point-lens fallback
        static_key = FitKey(
            lens_type=LensType.POINT,
            source_type=key.source_type,
            parallax_branch=ParallaxBranch.NONE,
            lens_orb_motion=LensOrbMotion.NONE,
        )
        static_rec = self.all_fit_results.get(static_key)
        if static_rec is None:
            raise RuntimeError(
                "A static point-lens fit must exist before fitting "
                "parallax branches."
            )
        base = dict(static_rec.params)
        base["pi_E_N"] = 0.0
        base["pi_E_E"] = 0.0
        logger.debug(
            "Seeding %s from static model with pi_E = 0.",
            key.parallax_branch.value,
        )
        return base

    def _needs_renormalization(self) -> bool:
        """
        Return True if any dataset's chi-squared per degree of freedom
        deviates from 1 by more than ``RENORM_THRESHOLD``.

        Uses ``select_best_model`` as the reference, consistent with
        ``renormalize_datasets``.  Both methods scale errors relative to
        ``chi2 / (n_good - n_params) = 1`` per dataset.

        Returns
        -------
        bool
        """
        if self.RENORM_THRESHOLD is None:
            return False

        try:
            reference_fit = self.select_best_model()
        except RuntimeError:
            return False

        event = reference_fit.full_result.fitter.get_event()
        event.fit_fluxes()
        n_params = len(event.model.parameters.as_dict())

        for i, dataset in enumerate(event.datasets):
            n_good = int(np.sum(dataset.good))
            dof = n_good - n_params
            if dof <= 0:
                continue
            chi2 = event.get_chi2_for_dataset(i)
            if np.abs(chi2 / dof - 1.0) > self.RENORM_THRESHOLD:
                # TODO: This should return True, but that doesn't actually cause any datasets to be re-renormalized.
                # Need to implement correct behavior for re-renormalization.
                # return True
                logger.warning(
                    "A dataset needs re-renormalization, but this feature is disabled."
                )
                return False

        return False

    def _infer_entry_point_from_initial_results(self) -> Optional[str]:
        """
        Determine the workflow entry point from user-supplied
        ``initial_results``.

        Rules
        -----
        - ``fit_type='point_lens'`` with PSPL params supplied →
          ``'fit_static_point_source_point_lens'`` (``estimate_point_lens_parameters`` is skipped; supplied params
          used as seed).
        - ``fit_type='binary_lens'`` with any PSPL params supplied →
          ``search_for_anomaly`` (all point-lens stages
          skipped; the supplied model is used directly).

        Returns
        -------
        str or None
            Step name to start from, or None if no shortcut applies.
        """
        has_pspl = any(
            key.lens_type == LensType.POINT
            and key.parallax_branch == ParallaxBranch.NONE
            for key in self.all_fit_results.keys()
        )

        if not has_pspl:
            return None

        if self.fit_type == "point_lens":
            return "fit_static_point_source_point_lens"
        if self.fit_type == "binary_lens":
            return "search_for_anomaly"

        return None

    def _save_restart_state(self) -> None:
        """
        Serialize current state to a restart pickle file.

        Notes
        -----
        Only writes to disk when ``self._restart_path`` is set, which
        happens automatically when ``restart_file`` is passed to
        ``__init__``.
        """
        if not getattr(self, "_restart_path", None):
            return

        restart_data = {
            "config": self._get_config(),
            "state": self._get_state(),
        }

        # debugging:
        # step = self.completed_steps[-1]
        # logger.info(f'DEBUG save_restart_state called after: {step.stage}:{step.name}')

        with open(self._restart_path, "wb") as f:
            pickle.dump(restart_data, f)

        logger.debug("Restart state saved to %s.", self._restart_path)

    def _get_config(self) -> dict:
        """
        Return the current configuration as a plain dict.

        Returns
        -------
        dict
        """
        return {key: getattr(self, key, None) for key in self.CONFIG_KEYS}

    def _iter_parallax_point_lens_keys(self) -> Iterable[FitKey]:
        """
        Yield FitKeys for all parallax branches appropriate for the
        current data.

        Uses U0_PLUS and U0_MINUS for single-location data and
        U0_PP, U0_MM, U0_PM, U0_MP for multi-location data.

        Yields
        ------
        mmexo.FitKey
            Parallax model keys.
        """
        n_loc = len(
            {getattr(ds, "ephemerides_file", None) for ds in self.datasets}
        )
        if n_loc <= 1:
            branches = [
                ParallaxBranch.U0_PLUS,
                ParallaxBranch.U0_MINUS,
            ]
        else:
            branches = [
                ParallaxBranch.U0_PP,
                ParallaxBranch.U0_MM,
                ParallaxBranch.U0_PM,
                ParallaxBranch.U0_MP,
            ]

        source_type = (
            SourceType.FINITE
            if self.finite_source_point_lens
            else SourceType.POINT
        )
        for branch in branches:
            yield FitKey(
                lens_type=LensType.POINT,
                source_type=source_type,
                parallax_branch=branch,
                lens_orb_motion=LensOrbMotion.NONE,
            )

    def _plot_piE_grid_search(self, grids: dict) -> None:
        """
        Create a two-panel piE grid search figure and save to disk.

        Parameters
        ----------
        grids : dict
            Mapping of ``mmexo.ParallaxBranch`` to
            ``ParallaxGridSearch`` objects.
        """
        # logger.info('Plotting piE grids: %s.', grids.keys())
        all_chi2 = [
            r["chi2_grid"]
            for grid in grids.values()
            for r in grid.results_history
        ]
        min_chi2 = np.nanmin(all_chi2)

        fig = plt.figure(figsize=(8, 6))
        gs = gridspec.GridSpec(
            1,
            3,
            figure=fig,
            width_ratios=[1, 1, 0.05],
            wspace=0.3,
        )
        axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]
        cax = fig.add_subplot(gs[2])

        branches = [
            ParallaxBranch.U0_PLUS,
            ParallaxBranch.U0_MINUS,
        ]
        scatter = None
        for i, (ax, par_branch) in enumerate(zip(axes, branches)):
            if par_branch not in grids:
                continue
            scatter = grids[par_branch].plot_grid_points(
                ax=ax, min_chi2=min_chi2
            )
            ax.set_xlabel(r"$\pi_{\rm E,E}$")
            ax.set_ylabel(r"$\pi_{\rm E,N}$")
            ax.set_title(par_branch.value)
            ax.invert_xaxis()
            ax.set_aspect("equal")
            ax.minorticks_on()
            if i == 1:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)

        if scatter is not None:
            fig.colorbar(
                scatter,
                cax=cax,
                label=(r"$\sigma$ (min $\chi^2$ = " + f"{min_chi2:.2f})"),
            )

        path = self._output_config.plot_path("piE_grid")
        fig.savefig(path)
        plt.close(fig)
        logger.info("Saved piE grid plot to %s.", path)

    def _get_event_t_range(self, event, n_tE=5):
        params = event.model.parameters.parameters
        start = params["t_0"] - n_tE * params["t_E"]
        stop = params["t_0"] + n_tE * params["t_E"]
        return [start, stop]

    def _get_planet_t_range(self, event, n_tE=5):
        model = event.model
        if model.methods is not None and (model.n_lenses > 1):
            if model.methods is dict:
                raise NotImplementedError(
                    "Plotting for Binary Source models not implemented, yet."
                )
                # probably want to loop over the sources and find min/max values of hexadecapole
            else:
                # mag_methods is now a single VBBL window spanning the whole
                # event (see BinaryLensParams.set_mag_method), so the method
                # list no longer brackets the anomaly the way the old
                # hexadecapole windows did. Prefer the anomaly finder window.
                if self.intermediate_results.best_af_grid_point is not None:
                    return self._get_af_grid_point_t_range()

                return [model.methods[0], model.methods[-1]]

        elif self.intermediate_results.best_af_grid_point is not None:
            return self._get_af_grid_point_t_range()
        else:
            return self._get_event_t_range(event, n_tE=n_tE)

    def _get_af_grid_point_t_range(self, n_teff=3):
        """Time range around the anomaly found by the anomaly finder grid."""
        point = self.intermediate_results.best_af_grid_point
        return [
            point["t_0"] - n_teff * point["t_eff"],
            point["t_0"] + n_teff * point["t_eff"],
        ]

    def _plot_planet_window(self):
        # TODO: Consider updating to use anomaly_lc_params? how different is best_af_grid_point from anomaly_lc_params?
        if self.intermediate_results.best_af_grid_point is not None:
            plt.axvline(
                self.intermediate_results.best_af_grid_point["t_0"]
                - 2450000.0,
                color="black",
                linestyle=":",
            )
            plt.axvline(
                self.intermediate_results.best_af_grid_point["t_0"]
                - self.intermediate_results.best_af_grid_point["t_eff"]
                - 2450000.0,
                color="black",
                linestyle="--",
            )
            plt.axvline(
                self.intermediate_results.best_af_grid_point["t_0"]
                + self.intermediate_results.best_af_grid_point["t_eff"]
                - 2450000.0,
                color="black",
                linestyle="--",
            )

    def _get_anomaly_source_plane_region(self, event, planet_t_range):
        traj = event.model.get_trajectory(planet_t_range)
        # for i in [0, -1]:
        #    print(planet_t_range[i], traj.times[i], traj.x[i], traj.y[i])

        xlim = np.min(traj.x), np.max(traj.x)
        ylim = np.min(traj.y), np.max(traj.y)
        # print(xlim, ylim)
        delta = np.max((xlim[1] - xlim[0], ylim[1] - ylim[0]))
        xlim = np.mean(xlim) + 0.5 * np.array([-delta, delta])
        ylim = np.mean(ylim) + 0.5 * np.array([-delta, delta])
        return xlim, ylim

    def _plot_event(self, event, n_tE=5, suptitle=None):
        # TODO: ADD automatic ylim
        # TODO: ADD residuals panels
        if suptitle is None:
            suptitle = "{0}".format(event.model.parameters)

        if event.model.n_lenses == 1:
            panels = 2
        else:
            panels = 3

        plt.figure(figsize=(5 * panels, 6))
        plt.suptitle(suptitle)
        plt.subplot(1, panels, 1)
        event.plot_data(show_bad=True, subtract_2450000=True)
        t_range = self._get_event_t_range(event, n_tE=n_tE)
        event.plot_model(
            t_range=t_range, subtract_2450000=True, color="black", zorder=10
        )
        if (
            event.model.n_lenses > 1
        ):  # TODO: Change to anomaly_lc_params exists
            self._plot_planet_window()

        plt.xlim(np.array(t_range) - 2450000.0)
        plt.minorticks_on()

        plt.subplot(1, panels, 2)
        event.plot_data(show_bad=True, subtract_2450000=True)
        if event.model.n_lenses > 1:
            planet_t_range = self._get_planet_t_range(event)
        else:
            # TODO: use anomaly_lc_params if exists
            planet_t_range = self._get_event_t_range(event, n_tE=0.5)

        event.plot_model(
            t_range=planet_t_range,
            color="black",
            subtract_2450000=True,
            zorder=10,
        )
        self._plot_planet_window()
        plt.xlim(np.array(planet_t_range) - 2450000.0)
        plt.minorticks_on()

        if panels > 2:
            plt.subplot(1, panels, 3)
            event.plot_trajectory(
                t_range=planet_t_range, caustics=True, zorder=10
            )
            # TODO: add scaled source to plot to indicate size.
            plt.gca().set_aspect("equal")
            xlim, ylim = self._get_anomaly_source_plane_region(
                event, planet_t_range
            )
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.minorticks_on()

        plt.tight_layout()

    def _plot_initial_2L1S_guess(self):
        # print(self.intermediate_results.estimate_binary_lens_parameters)
        for (
            key,
            params,
        ) in self.intermediate_results.estimate_binary_lens_parameters.items():
            print(key)
            print(params.ulens)
            print(params.mag_methods)
            model = self.model_config.build(
                parameters=params.ulens,
                magnification_methods=params.mag_methods,
                magnification_methods_parameters=params.mag_methods_parameters,
                default_magnification_method="point_source_point_lens",
            )
            event = self.event_config.build(
                model=model,
                datasets=self.datasets,
            )
            self._plot_event(
                event,
                suptitle=f"{key}: {event.get_chi2():.1f}\n{model.parameters}",
            )
            path = self._output_config.plot_path(f"af_{key}")
            plt.savefig(path)

    def _plot_best_fit_event(self):
        # Get the best fit
        complete_fits = [
            rec
            for key, rec in self.all_fit_results.items()
            if rec.full_result is not None
        ]
        best_fit = (
            min(complete_fits, key=lambda r: r.chi2())
            if complete_fits
            else None
        )
        if best_fit.model_key.lens_type == LensType.POINT:
            best_fit = self.select_best_point_lens_model()

        event = best_fit.full_result.fitter.get_event()

        # plot the light curve
        # event.plot(trajectory=False)
        self._plot_event(event)
        path = self._output_config.plot_path("lc")
        plt.savefig(path)
        logger.info("Saved light curve plot to %s.", path)

    # ------------------------------------------------------------------
    # Dataset helpers
    # ------------------------------------------------------------------

    def _create_mulensdata_objects(
        self,
        files,
        saved_datasets=None,
    ) -> list:
        """
        Create MulensData objects from file paths, reusing saved
        datasets when labels match.

        Parameters
        ----------
        files : str or list of str
            File path(s) to load.
        saved_datasets : list or None
            Previously saved datasets; any whose label matches a file
            basename are reused rather than re-loaded.

        Returns
        -------
        list of MulensModel.MulensData

        Raises
        ------
        FileNotFoundError
            If a requested file does not exist on disk.
        """
        if isinstance(files, str):
            files = [files]

        saved_by_label: dict = {}
        if saved_datasets:
            for ds in saved_datasets:
                label = ds.plot_properties.get("label")
                if label:
                    saved_by_label[label] = ds

        datasets = []
        for filename in files:
            label = os.path.basename(filename)
            if label in saved_by_label:
                datasets.append(saved_by_label[label])
            else:
                if not os.path.exists(filename):
                    raise FileNotFoundError(
                        f"Data file does not exist: {filename}"
                    )
                kwargs = get_kwargs(filename)
                datasets.append(
                    MulensModel.MulensData(file_name=filename, **kwargs)
                )

        return datasets

    def _validate_dataset_labels(self) -> None:
        """
        Validate that all user-provided datasets have labels set.

        For datasets with a ``file_name`` but no label, sets the label
        to the file basename.

        Raises
        ------
        ValueError
            If any dataset has neither ``file_name`` nor a label.
        """
        for i, dataset in enumerate(self.datasets):
            label = dataset.plot_properties.get("label")
            if not label:
                if getattr(dataset, "file_name", None):
                    dataset.plot_properties["label"] = os.path.basename(
                        dataset.file_name
                    )
                else:
                    raise ValueError(
                        f"Dataset at index {i} has no label in "
                        "plot_properties['label'] and was not loaded "
                        "from a file.  Set "
                        "plot_properties['label'] to a unique string "
                        "before passing to MMEXOFASTFitter."
                    )

    def _map_label_dict_to_datasets(self, label_dict) -> dict:
        """
        Convert a ``{label: value}`` dict to a
        ``{MulensData: value}`` dict.

        Parameters
        ----------
        label_dict : dict or None
            Keys are dataset label strings; values are booleans or
            floats.  If None, all datasets default to False.

        Returns
        -------
        dict
            Keys are MulensData objects; values from *label_dict*.
        """
        if label_dict is None:
            return {ds: False for ds in self.datasets}

        result = {}
        for ds in self.datasets:
            label = ds.plot_properties.get("label")
            result[ds] = label_dict.get(label, False) if label else False
        return result

    def _get_fitter_kwargs(self, source_type=None) -> dict:
        """
        Bundle fitter options for passing to ``SFitFitter``.

        Parameters
        ----------
        source_type : mmexo.SourceType, optional
            When ``SourceType.POINT``, ``mag_methods`` is suppressed
            (not needed for point-source magnification).

        Returns
        -------
        dict
            Keyword arguments ready to unpack into a fitter constructor.
        """
        return {
            "model_config": self.model_config,
            "event_config": self.event_config,
            "mag_methods": (
                None if source_type == SourceType.POINT else self.mag_methods
            ),
            "mag_methods_parameters": (
                None
                if source_type == SourceType.POINT
                else self.mag_methods_parameters
            ),
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _check_dataset_labels_unique(self) -> None:
        """
        Raise if any two datasets share the same label.

        Raises
        ------
        ValueError
            If duplicate dataset labels are found, or if any label is
            None.
        """
        labels = [ds.plot_properties.get("label") for ds in self.datasets]

        if None in labels:
            raise ValueError(
                "Some datasets do not have labels set in "
                "plot_properties['label'].  All datasets must have "
                "unique labels."
            )

        duplicates = [
            label for label in set(labels) if labels.count(label) > 1
        ]
        if duplicates:
            raise ValueError(
                f"Duplicate dataset labels found: {duplicates}.  "
                "All datasets must have unique labels in "
                "plot_properties['label']."
            )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def make_ulens_table(
        self,
        table_type: Optional[str] = "ascii",
        models=None,
    ) -> str:
        """
        Return a formatted table summarizing microlensing fit results.

        Parameters
        ----------
        table_type : str or None
            ``'ascii'`` (default) or ``'latex'``.
        models : list or None
            - None: include all models in ``self.all_fit_results``.
            - list of str: model label strings.
            - list of ``mmexo.FitKey``: explicit selection.

        Returns
        -------
        str
            Table in the requested format.

        Raises
        ------
        ValueError
            If a requested model label or key is not found.
        NotImplementedError
            If *table_type* is not ``'ascii'`` or ``'latex'``.
        """

        def _order_df(df: pd.DataFrame) -> pd.DataFrame:
            """
            Order parameters in a human-friendly way (ulens params
            first, then fluxes).

            Parameters
            ----------
            df : pd.DataFrame
                DataFrame to order.

            Returns
            -------
            pd.DataFrame
                Ordered DataFrame.
            """

            def _get_ordered_ulens_keys(n_sources: int = 1) -> list[str]:
                """
                Return the default ordering of microlensing parameters.

                Parameters
                ----------
                n_sources : int
                    Number of sources.

                Returns
                -------
                list of str
                """
                basic_keys = [
                    "t_0",
                    "u_0",
                    "t_E",
                    "rho",
                    "log_rho",
                    "t_star",
                ]
                additional_keys = [
                    "pi_E_N",
                    "pi_E_E",
                    "t_0_par",
                    "s",
                    "log_s",
                    "q",
                    "log_q",
                    "alpha",
                    "convergence_K",
                    "shear_G",
                    "ds_dt",
                    "dalpha_dt",
                    "s_z",
                    "ds_z_dt",
                    "t_0_kep",
                    "x_caustic_in",
                    "x_caustic_out",
                    "t_caustic_in",
                    "t_caustic_out",
                    "xi_period",
                    "xi_semimajor_axis",
                    "xi_inclination",
                    "xi_Omega_node",
                    "xi_argument_of_latitude_reference",
                    "xi_eccentricity",
                    "xi_omega_periapsis",
                    "q_source",
                    "t_0_xi",
                ]
                if n_sources > 1:
                    ordered: list[str] = []
                    for param_head in basic_keys:
                        if param_head == "t_E":
                            ordered.append(param_head)
                        else:
                            for idx in range(n_sources):
                                ordered.append(f"{param_head}_{idx + 1}")
                else:
                    ordered = list(basic_keys)
                ordered.extend(additional_keys)
                return ["chi2", "N_data"] + ordered

            def _get_ordered_flux_keys() -> list[str]:
                """
                Return ordered flux parameter names for all datasets.

                Returns
                -------
                list of str
                """
                flux_keys: list[str] = []
                for i, dataset in enumerate(self.datasets):
                    if "label" in dataset.plot_properties:
                        obs, band = get_telescope_band_from_filename(
                            dataset.plot_properties["label"]
                        )
                    else:
                        obs, band = i, None
                    flux_keys.append(f"{band}_S_{obs}")
                    flux_keys.append(f"{band}_B_{obs}")
                return flux_keys

            desired_order = (
                _get_ordered_ulens_keys() + _get_ordered_flux_keys()
            )
            order_map = {name: idx for idx, name in enumerate(desired_order)}
            df["sort_key"] = df["parameter_names"].map(order_map)
            df["orig_pos"] = range(len(df))
            df["sort_key"] = df["sort_key"].fillna(len(desired_order))
            df = (
                df.sort_values(["sort_key", "orig_pos"])
                .reset_index()
                .drop(columns=["index", "sort_key", "orig_pos"])
            )
            return df

        if table_type is None:
            table_type = "ascii"

        pm_symbol = r"$\pm$" if table_type == "latex" else "+/-"

        # Resolve models to (label, FitRecord) pairs
        pairs: list[tuple[str, FitRecord]] = []
        if models is None:
            for key, record in self.all_fit_results.items():
                label = model_key_to_label(key)
                pairs.append((label, record))
        else:
            for m in models:
                key = m if isinstance(m, FitKey) else label_to_model_key(m)
                record = self.all_fit_results.get(key)
                if record is None:
                    raise ValueError(f"No FitRecord found for model {m!r}.")
                pairs.append((model_key_to_label(key), record))

        results_table: Optional[pd.DataFrame] = None
        for label, record in pairs:
            new_col = record.to_dataframe()
            new_col = _format_results_column(new_col, pm_symbol)
            new_col = new_col.rename(
                columns={
                    "values": label,
                    "sigmas": f"sig [{label}]",
                    "sigma_minus": f"sig- [{label}]",
                    "sigma_plus": f"sig+ [{label}]",
                }
            )
            results_table = (
                new_col
                if results_table is None
                else results_table.merge(
                    new_col, on="parameter_names", how="outer"
                )
            )

        if results_table is None:
            return ""

        results_table = _order_df(results_table)

        if table_type == "latex":

            def _fmt_latex(name: str) -> str:
                if name == "chi2":
                    return r"$\chi^2$"
                parts = name.split("_")
                if len(parts) == 1:
                    return f"${name}$"
                first = parts[0]
                rest = ", ".join(parts[1:])
                return f"${first}" + "_{" + rest + "}$"

            results_table["parameter_names"] = results_table[
                "parameter_names"
            ].apply(_fmt_latex)
            return results_table.to_latex(index=False, escape=False)

        if table_type == "ascii":
            with pd.option_context(
                "display.max_rows",
                None,
                "display.max_columns",
                None,
                "display.width",
                None,
            ):
                return results_table.to_string(index=False)

        raise NotImplementedError(
            f"table_type {table_type!r} is not implemented."
        )

    def _exozippy_jd_offset(self) -> float:
        """
        JD offset needed to put the fitted epochs on a full JD scale.

        Microlensing data are commonly published in reduced form,
        ``HJD' = HJD - 2450000``, and MMEXOFAST fits in whatever system the
        input files use. EXOZIPPy expects full JD, so report the offset
        needed to get there: ``2450000`` when the data look reduced, and
        ``0`` when they are already full JD.

        Returns
        -------
        float
            Offset added to every epoch parameter by
            :meth:`initialize_exozippy`.
        """
        times = [
            dataset.time.max()
            for dataset in (self.datasets or [])
            if len(dataset.time) > 0
        ]
        if len(times) == 0:
            return 0.0

        return 0.0 if max(times) >= 2450000.0 else 2450000.0

    def _exozippy_excluded_points(self, jd_offset: float) -> dict:
        """
        Per-dataset record of the points excluded from the fit.

        Reported as indices rather than an ``n_data``-length boolean mask.
        Rejection is iterative and stops once nothing exceeds
        ``max(sqrt(2) * erfcinv(1/dof), 3)``, so roughly 0.3% of points are
        dropped; JSON has no packed boolean form, so a full mask would cost
        about 7 bytes per point of mostly ``false`` -- some 270 kB for a
        38568-epoch Data Challenge light curve, against a few hundred bytes
        for the indices.

        Times are given alongside the indices, and are the safer identifier:
        indices are positions in the photometry as loaded, in file order,
        which requires the consumer to skip the same header and comment lines
        MulensModel does, whereas a time matches regardless.

        This is every excluded point, not only those
        :meth:`renormalize_datasets` rejected -- it includes anything flagged
        bad in the input. That union is what EXOZIPPy needs in order to fit
        the same points, and it is empty when nothing was excluded.

        Parameters
        ----------
        jd_offset : float
            Applied to the reported times, so they share the time system of
            the epochs in ``'fits'``.

        Returns
        -------
        dict
            Keyed by dataset label, as ``'errfacs'`` is. Each value has
            ``'n_data'``, ``'indices'`` and ``'times'``.
        """
        excluded = {}
        for dataset in self.datasets or []:
            bad = np.asarray(dataset.bad, dtype=bool)
            indices = np.where(bad)[0]
            excluded[dataset.plot_properties["label"]] = {
                "n_data": int(bad.size),
                "indices": [int(index) for index in indices],
                "times": [
                    float(time) + jd_offset for time in dataset.time[indices]
                ],
            }

        return excluded

    @staticmethod
    def _shift_epochs(params: dict, offset: float) -> dict:
        """
        Copy ``params`` with every epoch parameter shifted by ``offset``.

        Only epochs move. Durations (``t_E``, ``t_star``) and everything
        else are invariant under a change of time origin, as are the
        sigmas, so shifting them would be wrong rather than merely
        redundant.

        Parameters
        ----------
        params : dict
            Model parameters. Not modified.
        offset : float
            Offset to add, from :meth:`_exozippy_jd_offset`.

        Returns
        -------
        dict
            A copy, shifted.
        """
        shifted = dict(params)
        if offset == 0.0:
            return shifted

        for name in EPOCH_PARAMETERS:
            if name in shifted:
                shifted[name] = shifted[name] + offset

        return shifted

    def initialize_exozippy(self) -> dict:
        """
        Return best-fit microlensing parameters for initializing
        EXOZIPPy fitting.

        Returns
        -------
        dict
            With keys:

            ``'fits'``
                List of ``{'parameters': dict, 'sigmas': dict}``.
            ``'errfacs'``
                Per-dataset error renormalization factors
                (``self.renorm_factors``).
            ``'mag_methods'``
                Magnification methods in MulensModel convention.
            ``'coords'``
                Event coordinates as a sexagesimal string, or None if
                none were supplied.
            ``'jd_offset'``
                Offset already added to every epoch in ``'fits'`` to put
                them on a full JD scale; see
                :meth:`_exozippy_jd_offset`. Subtract it to recover the
                epochs as fitted.
            ``'excluded_points'``
                Per-dataset indices and times of the points excluded from
                the fit, chiefly the outliers rejected during
                renormalization; see :meth:`_exozippy_excluded_points`.

        Raises
        ------
        NotImplementedError
            If ``fit_type`` is not ``'point_lens'``.
        """
        jd_offset = self._exozippy_jd_offset()
        coords = str(self.coords) if self.coords is not None else None
        excluded = self._exozippy_excluded_points(jd_offset)
        if self.fit_type == "point_lens":
            fits = []
            for key in self._iter_parallax_point_lens_keys():
                record = self.all_fit_results.get(key)
                if record is not None:
                    fits.append(
                        {
                            "parameters": self._shift_epochs(
                                record.params, jd_offset
                            ),
                            "sigmas": record.sigmas,
                        }
                    )

            return {
                "fits": fits,
                "errfacs": self.renorm_factors,
                "mag_methods": self.mag_methods,
                "coords": coords,
                "jd_offset": jd_offset,
                "excluded_points": excluded,
            }
        if self.fit_type == "binary_lens":
            fits = []

            binary_lens_fits = [
                rec
                for key, rec in self.all_fit_results.items()
                if key.lens_type == LensType.BINARY
            ]
            if len(binary_lens_fits) > 0:
                # Use real fits if they exist
                for binary_fit in binary_lens_fits:
                    fits.append(
                        {
                            "parameters": self._shift_epochs(
                                binary_fit.params, jd_offset
                            ),
                            "sigmas": binary_fit.sigmas,
                        }
                    )
            elif (
                self.intermediate_results.estimate_binary_lens_parameters
                is not None
            ):
                # otherwise, return the initial estimates
                for (
                    key,
                    params,
                ) in self.intermediate_results.estimate_binary_lens_parameters.items():
                    fits.append(
                        {
                            "parameters": self._shift_epochs(
                                params.ulens, jd_offset
                            )
                        }
                    )

        else:
            raise NotImplementedError(
                f"initialize_exozippy not implemented for {self.fit_type}."
            )

        return {
            "fits": fits,
            "errfacs": self.renorm_factors,
            "mag_methods": self.mag_methods,
            "coords": coords,
            "jd_offset": jd_offset,
            "excluded_points": excluded,
        }

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"MMEXOFASTFitter("
            f"fit_type={self.fit_type!r}, "
            f"completed_steps={len(self.completed_steps)}, "
            f"planned_steps={len(self.planned_steps)}, "
            f"n_fits={len(self.all_fit_results)})"
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False  # Don't suppress exceptions

    def close(self) -> None:
        """
        Remove and close any logging handlers added by this fitter.

        Call this when the fitter is no longer needed to prevent handler
        accumulation when multiple fitter instances are created in one
        process.
        """
        _mod_logger = logging.getLogger(__name__)
        for handler in self._log_handlers:
            handler.close()
            _mod_logger.removeHandler(handler)
        self._log_handlers.clear()
