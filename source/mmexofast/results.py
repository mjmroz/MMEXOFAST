from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any, Dict, Optional

import MulensModel
import numpy as np
import pandas as pd

from mmexofast.fit_types import (
    FitKey,
    LensType,
    label_to_model_key,
    model_key_to_label,
)
from mmexofast.observatories import get_telescope_band_from_filename


# ============================================================================
# FitResults wrappers
# ============================================================================
class BaseFitResults(ABC):
    """
    Abstract base class for fit results wrappers.

    Defines the interface that all fit results classes must implement so that
    ``FitRecord`` can consume them interchangeably, regardless of the
    underlying fitter (e.g. SFit, emcee).

    Concrete subclasses must implement:
        - ``get_params_from_results()``
        - ``get_sigmas_from_results()``
        - ``format_results_as_df()``

    Parameters
    ----------
    fitter : object
        The fitter object whose results are being wrapped. Must expose
        ``best``, ``parameters_to_fit``, and ``datasets`` attributes.

    Attributes
    ----------
    fitter : object
        The wrapped fitter object.
    """

    def __init__(self, fitter):
        self.fitter = fitter

    # -----------------------------------------------------------------------
    # Abstract interface — must be implemented by subclasses
    # -----------------------------------------------------------------------

    @abstractmethod
    def get_params_from_results(self) -> dict:
        """
        Return the best-fit model parameters as a dict.

        Returns a dictionary mapping linear-space parameter names to their
        best-fit values, suitable for use as input to
        ``MulensModel.Model()``. Must not include ``'chi2'``.

        Returns
        -------
        dict
            Parameter name -> best-fit value.
        """

    @abstractmethod
    def get_sigmas_from_results(self) -> dict:
        """
        Return 1-sigma uncertainties as a dict.

        Returns a dictionary mapping parameter names to their 1-sigma
        uncertainties. For asymmetric uncertainties (e.g. from emcee),
        returns the mean of the upper and lower uncertainties.

        Returns
        -------
        dict
            Parameter name -> 1-sigma uncertainty.
        """

    @abstractmethod
    def format_results_as_df(self) -> pd.DataFrame:
        """
        Return fit results as a pandas DataFrame.

        The DataFrame must contain at minimum the columns
        ``'parameter_names'`` and ``'values'``. Sigma columns vary by
        subclass: ``MMEXOFASTFitResults`` produces ``'sigmas'``;
        ``EmceeFitResults`` produces ``'sigma_minus'`` and
        ``'sigma_plus'``.

        Returns
        -------
        pd.DataFrame
        """

    # -----------------------------------------------------------------------
    # Concrete shared properties
    # -----------------------------------------------------------------------

    @property
    def datasets(self):
        """list : Datasets from the fitter."""
        return self.fitter.datasets

    @property
    def best(self):
        """dict : Best-fit parameters including chi2."""
        return self.fitter.best

    @property
    def parameters_to_fit(self):
        """list of str : Names of the parameters sampled by the fitter."""
        return self.fitter.parameters_to_fit

    @property
    def all_model_parameters(self):
        """dict_keys : All parameter names in best, including fixed ones."""
        return self.fitter.best.keys()

    @property
    def chi2(self):
        """float or None : Best-fit chi2, or None if not available."""
        return self.fitter.best.get("chi2")


class MMEXOFASTFitResults(BaseFitResults):
    """
    Wrapper for results from either an SFit minimizer or an emcee MCMC run.

    Exposes the ``BaseFitResults`` interface so that ``FitRecord`` can
    consume both SFit and emcee results identically.

    The type of fitter is detected automatically via duck typing in
    ``_is_emcee()``: emcee results expose ``sigma_minus``; sfit results
    do not.

    Assumes ``fitter`` exposes ``.best``, ``.results``,
    ``.parameters_to_fit``, and ``.datasets``. For emcee fitters,
    ``fitter.results`` must be an ``EmceeFitResults`` instance (from
    ``fitters.py``) set by ``EmceeLCFitter.run()``.

    Parameters
    ----------
    fitter : object
        The fitter object after the fit has completed. For sfit, must
        expose ``best``, ``results``, ``parameters_to_fit``, and
        ``datasets``. For emcee, must additionally expose
        ``best_theta``, ``_event``, ``initialize_event()``, and
        ``get_parameter_name()``.

    Notes
    -----
    ``format_results_as_df()`` produces different sigma columns depending
    on the fitter type:

    - sfit: single ``'sigmas'`` column (symmetric uncertainties).
    - emcee: ``'sigma_minus'`` and ``'sigma_plus'`` columns (asymmetric).
      Fitted parameter values are p50 (median of post-burn-in chain).
    """

    def __init__(self, fitter):
        super().__init__(fitter)

    # -----------------------------------------------------------------------
    # Detection
    # -----------------------------------------------------------------------

    def _is_emcee(self) -> bool:
        """
        Return True if the fitter's results object is from an emcee run.

        Detected via duck typing: emcee results expose ``sigma_minus``,
        which sfit results do not. This avoids a cross-module import of
        ``EmceeFitResults`` from ``fitters.py``.

        Returns
        -------
        bool
        """
        return hasattr(self.fitter.results, "sigma_minus")

    # -----------------------------------------------------------------------
    # BaseFitResults interface
    # -----------------------------------------------------------------------

    def get_params_from_results(self) -> Dict[str, float]:
        """
        Return a dict with just the best-fit microlensing parameters and
        values, i.e., something appropriate for using as input to
        ``MulensModel.Model()``.
        """
        params = {key: value for key, value in self.best.items()}
        params.pop("chi2", None)
        return params

    def get_sigmas_from_results(self) -> Dict[str, float]:
        """
        Return a dict mapping parameter name -> 1-sigma uncertainty.

        For both sfit and emcee, reads ``fitter.results.sigmas``. For
        emcee, this is the symmetric mean of ``sigma_minus`` and
        ``sigma_plus``, computed inside ``EmceeFitResults``.
        """
        sigmas = {}
        for param, sigma in zip(self.parameters_to_fit, self.results.sigmas):
            sigmas[param] = sigma
        return sigmas

    def format_results_as_df(self) -> pd.DataFrame:
        """
        Build a summary DataFrame (fitted params, fixed params, flux params).

        Branches on whether the fitter used emcee or sfit, detected via
        ``_is_emcee()``. The two paths produce different sigma columns:

        - sfit: single ``'sigmas'`` column (symmetric).
        - emcee: ``'sigma_minus'`` and ``'sigma_plus'`` columns (asymmetric).
          Fitted parameter values are p50 (median of post-burn-in chain).

        Returns
        -------
        pd.DataFrame
        """
        if self._is_emcee():
            df_fitted = self._get_df_fitted_parameters_emcee()
            df_fixed = self._get_df_fixed_parameters_emcee()
            df_flux = self._get_df_flux_parameters_emcee()
        else:
            df_fitted = self._get_df_fitted_parameters_sfit()
            df_fixed = self._get_df_fixed_parameters_sfit()
            df_flux = self._get_df_flux_parameters_sfit()

        df_ulens = pd.concat((df_fitted, df_fixed))
        return pd.concat((df_ulens, df_flux), ignore_index=True)

    # -----------------------------------------------------------------------
    # sfit private helpers
    # -----------------------------------------------------------------------

    def _get_df_fitted_parameters_sfit(self) -> pd.DataFrame:
        """
        Build the fitted parameters section of the DataFrame for sfit results.

        Uses ``results.x`` for values and ``results.sigmas`` for symmetric
        uncertainties.

        Returns
        -------
        pd.DataFrame
            Columns: ``'parameter_names'``, ``'values'``, ``'sigmas'``.
        """
        parameters = list(self.parameters_to_fit)
        values = list(self.results.x[0 : len(parameters)])
        sigmas = list(self.results.sigmas[0 : len(parameters)])

        return pd.DataFrame(
            {
                "parameter_names": parameters,
                "values": values,
                "sigmas": sigmas,
            }
        )

    def _get_df_fixed_parameters_sfit(self) -> pd.DataFrame:
        """
        Build the fixed parameters and N_data section of the DataFrame
        for sfit results.

        Fixed parameters are those present in ``best`` but absent from
        ``parameters_to_fit``. ``N_data`` (total good data points across
        all datasets) is appended last.

        Returns
        -------
        pd.DataFrame
            Columns: ``'parameter_names'``, ``'values'``.
        """
        fixed_parameters = [
            p
            for p in self.all_model_parameters
            if p not in self.parameters_to_fit
        ]
        values = [self.best[param] for param in fixed_parameters]
        fixed_parameters.append("N_data")
        values.append(
            np.sum([np.sum(dataset.good) for dataset in self.datasets])
        )

        return pd.DataFrame(
            {
                "parameter_names": fixed_parameters,
                "values": values,
            }
        )

    def _get_df_flux_parameters_sfit(self) -> pd.DataFrame:
        """
        Build the flux parameters section of the DataFrame for sfit results.

        Reads source and blend fluxes directly from ``results.x`` using
        the sfit index layout: model parameters occupy the first
        ``len(parameters_to_fit)`` indices, followed by source and blend
        flux pairs for each dataset.

        Returns
        -------
        pd.DataFrame
            Columns: ``'parameter_names'``, ``'values'``, ``'sigmas'``.
        """
        parameters = []
        values = []
        sigmas = []

        for i, dataset in enumerate(self.datasets):
            obs, band = get_telescope_band_from_filename(
                dataset.plot_properties["label"]
            )
            parameters.append(f"{band}_S_{obs}")
            parameters.append(f"{band}_B_{obs}")

            obs_index = len(self.parameters_to_fit) + 2 * i
            for index in range(2):
                flux = self.results.x[obs_index + index]
                if flux > 0:
                    err_flux = self.results.sigmas[obs_index + index]
                    mag, err_mag = (
                        MulensModel.utils.Utils.get_mag_and_err_from_flux(
                            flux, err_flux
                        )
                    )
                else:
                    # TODO: Maybe it would be better to give the blend ratio instead of neg flux
                    mag = "neg flux"
                    err_mag = np.nan

                values.append(mag)
                sigmas.append(err_mag)

        return pd.DataFrame(
            {
                "parameter_names": parameters,
                "values": values,
                "sigmas": sigmas,
            }
        )

    # -----------------------------------------------------------------------
    # emcee private helpers
    # -----------------------------------------------------------------------

    def _get_df_fitted_parameters_emcee(self) -> pd.DataFrame:
        """
        Build the fitted parameters section of the DataFrame for emcee results.

        Uses p50 (median of post-burn-in chain) as values. ``sigma_minus``
        and ``sigma_plus`` are read from ``fitter.results`` and are both
        stored as positive numbers.

        Returns
        -------
        pd.DataFrame
            Columns: ``'parameter_names'``, ``'values'``,
            ``'sigma_minus'``, ``'sigma_plus'``.
        """
        p = self.fitter.results.percentiles
        return pd.DataFrame(
            {
                "parameter_names": list(self.parameters_to_fit),
                "values": list(p[1]),
                "sigma_minus": list(self.fitter.results.sigma_minus),
                "sigma_plus": list(self.fitter.results.sigma_plus),
            }
        )

    def _get_df_fixed_parameters_emcee(self) -> pd.DataFrame:
        """
        Build the fixed parameters and N_data section of the DataFrame
        for emcee results.

        Fixed parameters are those present in ``best`` but absent from
        the linear-mapped ``parameters_to_fit``. ``chi2`` is included
        here. ``N_data`` (total good data points across all datasets)
        is appended last. All sigma columns are NaN.

        Returns
        -------
        pd.DataFrame
            Columns: ``'parameter_names'``, ``'values'``,
            ``'sigma_minus'``, ``'sigma_plus'``.
        """
        linear_params_to_fit = {
            self.fitter.get_parameter_name(p) for p in self.parameters_to_fit
        }
        fixed_parameters = [
            p
            for p in self.all_model_parameters
            if p not in linear_params_to_fit
        ]
        values = [self.best[p] for p in fixed_parameters]

        fixed_parameters.append("N_data")
        values.append(
            int(np.sum([np.sum(dataset.good) for dataset in self.datasets]))
        )

        n = len(fixed_parameters)
        return pd.DataFrame(
            {
                "parameter_names": fixed_parameters,
                "values": values,
                "sigma_minus": [np.nan] * n,
                "sigma_plus": [np.nan] * n,
            }
        )

    def _get_df_flux_parameters_emcee(self) -> pd.DataFrame:
        """
        Build the flux parameters section of the DataFrame for emcee results.

        Delegates event setup to ``fitter.get_best_fit_event()``, which
        sets model parameters to ``best_theta`` and fits fluxes.

        Source and blend fluxes are converted to magnitudes via
        ``MulensModel.utils.Utils.get_mag_and_err_from_flux``. Negative
        fluxes are reported as ``'neg flux'``. All sigma columns are NaN
        since flux uncertainties are not available from the emcee chain.

        Returns
        -------
        pd.DataFrame
            Columns: ``'parameter_names'``, ``'values'``,
            ``'sigma_minus'``, ``'sigma_plus'``.
        """
        event = self.fitter.get_best_fit_event()
        source_fluxes = event.source_fluxes
        blend_fluxes = event.blend_fluxes

        parameters = []
        values = []

        for i, dataset in enumerate(self.datasets):
            obs, band = get_telescope_band_from_filename(
                dataset.plot_properties["label"]
            )
            if len(source_fluxes[i]) == 1:
                parameters.append(f"{band}_S_{obs}")
            else:
                for j in range(len(source_fluxes[i])):
                    parameters.append(f"{band}_S{j}_{obs}")

            parameters.append(f"{band}_B_{obs}")

            for flux in list(source_fluxes[i]) + [blend_fluxes[i]]:
                flux_scalar = float(np.squeeze(flux))
                if flux_scalar > 0:
                    mag, _ = MulensModel.utils.Utils.get_mag_and_err_from_flux(
                        flux_scalar, 0.0
                    )
                else:
                    mag = "neg flux"

                values.append(mag)

        n = len(parameters)
        return pd.DataFrame(
            {
                "parameter_names": parameters,
                "values": values,
                "sigma_minus": [np.nan] * n,
                "sigma_plus": [np.nan] * n,
            }
        )

    # -----------------------------------------------------------------------
    # Property
    # -----------------------------------------------------------------------

    @property
    def results(self):
        """object : Full results object from the fitter."""
        return self.fitter.results


# ============================================================================
# FitRecord and AllFitResults
# ============================================================================


@dataclass
class FitRecord:
    """
    Container for a fit result from MMEXOFAST.

    Stores model parameters, uncertainties, and associated fit metadata for a
    single model configuration (lens type, source type, parallax branch, etc.).
    Optionally retains the full fit result object for downstream analysis.

    Attributes
    ----------
    model_key : FitKey
        Key identifying the model configuration (lens type, source type, etc.).
    params : dict
        Dictionary mapping parameter names to fitted values.
    sigmas : dict, optional
        Dictionary mapping parameter names to 1-sigma uncertainties.
        None if uncertainties were not computed.
    renorm_factors : dict, optional
        Dictionary of renormalization/systematics factors applied.
        None if no renormalization was needed.
    full_result : object, optional
        Complete fit results object from MMEXOFAST.
        None if only summary data is retained.
    fixed : bool
        Whether the fit was performed with fixed parameters.
    is_complete : bool
        Whether the fit completed successfully.

    """

    model_key: FitKey
    params: dict
    sigmas: dict = None
    renorm_factors: dict = None
    full_result: object = None
    fixed: bool = False
    is_complete: bool = False

    @classmethod
    def from_full_result(
        cls, model_key, full_result, renorm_factors=None, fixed=False
    ):
        """
        Construct a FitRecord from a full MMEXOFASTFitResults object.

        Parameters
        ----------
        model_key : FitKey
            Key identifying the model configuration.
        full_result : MMEXOFASTFitResults
            Complete fit results from MMEXOFAST.
        renorm_factors : dict, optional
            Dictionary of renormalization factors. Default is None.
        fixed : bool, optional
            Whether the fit used fixed parameters. Default is False.

        Returns
        -------
        FitRecord
            New FitRecord instance populated from the full result.

        """
        params = full_result.get_params_from_results()
        try:
            sigmas = full_result.get_sigmas_from_results()
        except Exception:
            sigmas = None

        return cls(
            model_key=model_key,
            params=params,
            sigmas=sigmas,
            renorm_factors=renorm_factors,
            full_result=full_result,
            fixed=fixed,
            is_complete=True,
        )

    def to_dataframe(self):
        """
        Export fit results as a pandas DataFrame.

        If ``full_result`` is available, delegates to its
        ``format_results_as_df()`` method. Otherwise returns a minimal
        DataFrame constructed from ``params`` and ``sigmas``.

        The column structure depends on the type of ``full_result``:

        - ``MMEXOFASTFitResults``: columns are ``'parameter_names'``,
          ``'values'``, ``'sigmas'``. Uncertainties are symmetric.
        - ``EmceeFitResults``: columns are ``'parameter_names'``,
          ``'values'``, ``'sigma_minus'``, ``'sigma_plus'``. Uncertainties
          are asymmetric. ``sigma_minus = p50 - p16`` and
          ``sigma_plus = p84 - p50``, both stored as positive numbers.
          Fixed parameters, ``N_data``, and flux parameters have
          ``NaN`` for both sigma columns.
        - Minimal fallback (no ``full_result``): columns are
          ``'parameter_names'``, ``'values'``, ``'sigmas'``.

        Returns
        -------
        pd.DataFrame
            DataFrame with ``'parameter_names'`` and ``'values'`` columns
            at minimum. Sigma column(s) vary by ``full_result`` type as
            described above.

        See Also
        --------
        MMEXOFASTFitResults.format_results_as_df
        EmceeFitResults.format_results_as_df
        """
        if self.full_result is not None:
            return self.full_result.format_results_as_df()
        return self._minimal_dataframe()

    def _minimal_dataframe(self):
        """
        Construct a minimal DataFrame from params and sigmas only.

        Used when full_result is unavailable. Returns basic parameter values
        and uncertainties without additional fit metadata (e.g., fluxes, N_data).

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: 'parameter_names', 'values', 'sigmas'.

        """
        param_names = list(self.params.keys())
        values = [self.params[name] for name in param_names]
        if self.sigmas is not None:
            sigmas = [self.sigmas.get(name, None) for name in param_names]
        else:
            sigmas = [None] * len(param_names)

        return pd.DataFrame(
            {
                "parameter_names": param_names,
                "values": values,
                "sigmas": sigmas,
            }
        )

    def __repr__(self):
        """
        Return a compact string representation of the FitRecord.

        Displays model configuration, parameter count, and fit status.
        Parameter dictionaries are truncated for readability if they exceed
        5 items.

        Returns
        -------
        str
            String representation of the FitRecord.

        """
        has_full = self.full_result is not None
        has_sigmas = self.sigmas is not None
        has_renorm = self.renorm_factors is not None
        n_params = len(self.params) if self.params is not None else 0

        def _short_dict(d, max_items=5):
            """Truncate dictionary representation for display."""
            if not d:
                return "{}"
            items = list(d.items())
            if len(items) > max_items:
                head = ", ".join(f"{k}={v}" for k, v in items[:max_items])
                return "{" + head + ", ...}"
            return "{" + ", ".join(f"{k}={v}" for k, v in items) + "}"

        params_repr = _short_dict(self.params)
        sigmas_repr = _short_dict(self.sigmas)

        return (
            f"<FitRecord("
            f"lens={self.model_key.lens_type.value}, "
            f"source={self.model_key.source_type.value}, "
            f"parallax={self.model_key.parallax_branch.value}, "
            f"motion={self.model_key.lens_orb_motion.value}; "
            f"params={params_repr}, sigmas={sigmas_repr}; "
            f"full={has_full}, fixed={self.fixed}, complete={self.is_complete}, "
            f"renorm={has_renorm}, n_params={n_params}"
            f")>"
        )

    def chi2(self):
        """
        Extract chi-squared value from the fit result.

        Returns the best-fit chi-squared statistic if full_result is available,
        otherwise returns None.

        Returns        -------
        float or None
            Chi-squared value, or None if full_result is unavailable.

        """
        if self.full_result is None:
            return None

        return self.full_result.chi2


@dataclass
class GridSearchResult:
    """
    Results of a grid search, intended to be optionally persisted to disk.

    Attributes
    ----------
    name : str
        Short name of the grid search, e.g. 'EF', 'AF', 'PAR'.
    param_names : tuple[str, ...]
        Names of the grid parameters, e.g. ('s', 'q'), ('pi_E_N', 'pi_E_E').
    grid_points : np.ndarray
        Array of shape (N_points, n_params) containing grid coordinates.
    chi2 : np.ndarray
        Array of shape (N_points,) with chi^2 (or other scalar merit) values.
    metadata : dict
        Arbitrary extra info (datasets used, dates, config settings, etc.).
    best_index : int
        Index into grid_points / chi2 of the best point.
    """

    name: str
    param_names: tuple[str, ...]
    grid_points: np.ndarray
    chi2: np.ndarray
    metadata: Dict[str, Any]
    best_index: int


class AllFitResults(MutableMapping):
    """
    Central registry for all fit results, keyed by FitKey.
    """

    def __init__(self):
        self._records: Dict[FitKey, FitRecord] = {}

    # --- Required MutableMapping methods ---
    def __getitem__(self, key_or_label: str | FitKey) -> FitRecord:
        key = self._normalize_key(key_or_label)
        return self._records[key]

    def __setitem__(
        self, key_or_label: str | FitKey, record: FitRecord
    ) -> None:
        key = self._normalize_key(key_or_label)
        self._records[key] = record

    def __delitem__(self, key_or_label: str | FitKey) -> None:
        key = self._normalize_key(key_or_label)
        del self._records[key]

    def __iter__(self):
        return iter(self._records)

    def __len__(self):
        return len(self._records)

    # --- internal helper ---
    def _normalize_key(self, key_or_label: str | FitKey) -> FitKey:
        if isinstance(key_or_label, FitKey):
            return key_or_label
        return label_to_model_key(key_or_label)

    # --- custom convenience methods ---
    def get(self, key_or_label: str | FitKey) -> Optional[FitRecord]:
        key = self._normalize_key(key_or_label)
        return self._records.get(key)

    def set(self, record: FitRecord) -> None:
        self._records[record.model_key] = record

    def has(self, key_or_label: str | FitKey) -> bool:
        key = self._normalize_key(key_or_label)
        return key in self._records

    def keys(self, labels: bool = False):
        if labels:
            return [model_key_to_label(k) for k in self._records.keys()]
        return list(self._records.keys())

    def items(self, labels: bool = False):
        if labels:
            return [
                (model_key_to_label(k), r) for k, r in self._records.items()
            ]
        return list(self._records.items())

    def __repr__(self) -> str:
        if not self._records:
            return "<AllFitResults: (empty)>"

        lines = ["<AllFitResults:"]
        for key, record in self._records.items():
            label = model_key_to_label(key)
            lines.append(f"  {label!r}: {record}")
        lines.append(">")
        return "\n".join(lines)

    def iter_point_lens_records(self):
        """Yield (key, record) pairs for all point-lens models (PSPL/FSPL)."""
        for key, record in self._records.items():
            if key.lens_type == LensType.POINT:
                yield key, record


@dataclass
class IntermediateResults:
    """
    Stores intermediate, non-fit results produced during workflow execution.

    These are results that are needed by subsequent steps but are not
    fit results stored in AllFitResults. All fields default to None and
    are populated by their corresponding workflow step actions.

    Fields
    ------
    best_ef_grid_point : dict or None
        Best grid point from the EventFinder grid search.
        Set by: run_event_search
        Format: {'t_0': float, 't_eff': float, 'j': int, 'chi2': float}

    best_af_grid_point : dict or None
        Best grid point from the AnomalyFinder grid search.
        Set by: run_anomaly_search
        Format: TBD

    estimate_point_lens_parameters : dict or None
        Estimated point-lens parameters from the EF grid result.
        Set by: estimate_point_lens_parameters
        Format: {'t_0': float, 'u_0': float, 't_E': float}

    estimate_binary_lens_parameters : dict of dicts or None
        Estimated binary lens parameters from the AF grid result. Some anomaly
        types may have multiple possible solutions.
        Set by: estimate_binary_lens_parameters
        Format: {solution: {'t_0': float, 'u_0': float, 't_E': float,
                 'rho': float, 'q': float, 's': float, 'alpha': float}, ...}

    anomaly_lc_params : dict or None
        PSPL properties + observed anomaly properties.
        Set by: get_anomaly_light_curve_parameters
        Format: {'t_0': float, 'u_0': float, 't_E': float,
                  'dmag': float, 'dt': float, 't_pl': float}

    anomaly_type : str or None
       Type of anomaly. Allowed types are given in VALID_ANOMALY_TYPES

    """

    VALID_ANOMALY_TYPES = {"high_mag", "bump", "dip", "caustic_crossing"}

    best_ef_grid_point: Optional[dict] = None
    best_af_grid_point: Optional[dict] = None
    estimate_point_lens_parameters: Optional[dict] = None
    estimate_binary_lens_parameters: Optional[dict] = None
    anomaly_lc_params: Optional[dict] = None

    def __init__(self):
        self._anomaly_type: Optional[str] = None

    @property
    def anomaly_type(self) -> Optional[str]:
        return self._anomaly_type

    @anomaly_type.setter
    def anomaly_type(self, value: Optional[str]) -> None:
        if value is not None and value not in self.VALID_ANOMALY_TYPES:
            raise ValueError(
                f"{value!r} is not a valid anomaly_type. "
                f"Must be one of {self.VALID_ANOMALY_TYPES}."
            )
        self._anomaly_type = value

    def __getstate__(self):
        state = self.__dict__.copy()
        state["anomaly_type"] = self.anomaly_type
        state.pop("_anomaly_type", None)
        return state

    def __setstate__(self, state):
        anomaly_type = state.pop("anomaly_type", None)
        self.__dict__.update(state)
        self._anomaly_type = None
        self.anomaly_type = anomaly_type
