import logging
import os
from multiprocessing import Pool, cpu_count

import emcee
import matplotlib.pyplot as plt
import MulensModel
import numpy as np
import sfit_minimizer as sfit
from scipy.optimize import curve_fit, minimize

from .estimate_params import WideAxisEnsembleInitializer
from .mulens_object_config import EventConfig, ModelConfig

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------- #
# Multiprocessing support for the emcee fitters.
#
# emcee's pool path pickles its log_prob_fn into the workers on EVERY
# pool.map task chunk. With a bound method (self.ln_prob) that means
# re-serializing the whole fitter -- datasets included -- twice per emcee
# step, and each unpickled copy arrives with _event of None (see
# __getstate__) and rebuilds its Event per chunk. In practice the parent
# process saturates one CPU pickling while every worker idles, which is
# SLOWER than no pool at all. The standard fix: ship the fitter to each
# worker ONCE via the Pool initializer, and hand emcee a tiny module-level
# function that closes over nothing -- only the theta vectors travel per
# call. Each worker keeps its lazily-initialized Event for the whole run.
# --------------------------------------------------------------------- #

_POOL_FITTER = None
logger = logging.getLogger(__name__)


def _pool_worker_init(fitter):
    """Pool initializer: install the (once-pickled) fitter in this worker."""
    global _POOL_FITTER
    _POOL_FITTER = fitter


def _pool_ln_prob(theta):
    """Module-level ln_prob proxy (picklable by reference, unlike a bound
    method); evaluates against the worker's resident fitter."""
    return _POOL_FITTER.ln_prob(theta)


class EmceeFitResults:
    """
    Results from a completed emcee MCMC run.

    Stores the full sampler and best-fit parameter vector, and lazily
    computes percentiles and uncertainties from the post-burn-in chain.

    Parameters
    ----------
    sampler : emcee.EnsembleSampler
        The full sampler object from a completed emcee run.
    x : array-like
        Best-fit (max-likelihood) parameter values from the
        post-burn-in chain.
    emcee_settings : dict
        Settings dict from the fitter. Must contain 'n_burn' and
        'n_dim'. Other keys ('n_walkers', 'n_steps', etc.) are
        stored but not used internally.
    parameters_to_fit : list of str
        Names of the parameters sampled by emcee, in the same order
        as the chain columns.

    Attributes
    ----------
    sampler : emcee.EnsembleSampler
        The full sampler object.
    x : np.ndarray, shape (n_dim,)
        Best-fit (max-likelihood) parameter values.
    emcee_settings : dict
        Settings dict from the fitter.
    parameters_to_fit : list of str
        Names of the parameters sampled by emcee.

    Notes
    -----
    ``sigma_minus = p50 - p16`` and ``sigma_plus = p84 - p50`` are both
    stored as positive numbers. The minus sign is a display concern only.

    Percentiles are computed lazily on first access and cached.
    """

    def __init__(self, sampler, x, emcee_settings, parameters_to_fit):
        self.sampler = sampler
        self.x = np.array(x)
        self.emcee_settings = emcee_settings
        self.parameters_to_fit = parameters_to_fit
        self._percentiles = None

    @property
    def percentiles(self):
        """
        np.ndarray, shape (3, n_dim) : 16th, 50th, and 84th percentiles
        of the post-burn-in chain, one column per parameter in
        ``parameters_to_fit``. Rows are p16, p50, p84. Computed once
        and cached.
        """
        if self._percentiles is None:
            n_burn = self.emcee_settings["n_burn"]
            n_dim = self.emcee_settings["n_dim"]
            samples = self.sampler.chain[:, n_burn:, :].reshape((-1, n_dim))
            self._percentiles = np.percentile(samples, [16, 50, 84], axis=0)
        return self._percentiles

    @property
    def sigma_minus(self):
        """
        np.ndarray, shape (n_dim,) : p50 - p16 for each parameter,
        stored as a positive number.
        """
        return self.percentiles[1] - self.percentiles[0]

    @property
    def sigma_plus(self):
        """
        np.ndarray, shape (n_dim,) : p84 - p50 for each parameter,
        stored as a positive number.
        """
        return self.percentiles[2] - self.percentiles[1]

    @property
    def sigmas(self):
        """
        np.ndarray, shape (n_dim,) : Symmetric 1-sigma uncertainties,
        computed as (sigma_minus + sigma_plus) / 2.
        """
        return (self.sigma_minus + self.sigma_plus) / 2

    def __repr__(self):
        """
        Return a compact string representation of EmceeFitResults.

        Returns
        -------
        str
            String representation showing parameters, x values, and
            symmetric sigmas.
        """
        return (
            f"EmceeFitResults(\n"
            f"  parameters_to_fit={self.parameters_to_fit},\n"
            f"  x={self.x},\n"
            f"  sigmas={self.sigmas},\n"
            f"  n_burn={self.emcee_settings['n_burn']},\n"
            f"  n_steps={self.emcee_settings['n_steps']},\n"
            f"  n_walkers={self.emcee_settings['n_walkers']},\n"
            f")"
        )


class MulensFitter:
    """
    Parent class for microlensing model fitters.

    Parameters
    ----------
    datasets : list
        List of MulensModel.MulensData objects.
    initial_model_params : dict
        Initial parameters of the model.
    parameters_to_fit : list, optional
        Parameters to be fitted. If None, all keys in initial_model_params are fitted.
    sigmas : dict, optional
        Dict mapping parameter names to step sizes. Parameters
        not in the dict default to ``None``.
    mag_methods : list, optional
        Magnification methods specification; see
        MulensModel.Model.set_magnification_methods. Passed directly to
        ``get_model()`` since it varies per model type.
    mag_methods_parameters : dict, optional
        Parameters for those methods, e.g.
        ``{'VBBL': {'accuracy': 0.01}}``; see
        MulensModel.Model.set_magnification_methods_parameters. Like
        ``mag_methods`` it varies per model type. Usually supplied as
        ``BinaryLensParams.mag_methods_parameters``.
    model_config : ModelConfig, optional
        Configuration for Model construction (coords, limb darkening, etc.).
        If None, a default ``ModelConfig`` is used.
    event_config : EventConfig, optional
        Configuration for Event construction (coords, flux fixing, etc.).
        If None, a default ``EventConfig`` is used.
    verbose : bool, optional
        If True, print progress information. Default is False.
    pool : multiprocessing.Pool, optional
        Pool for parallel computation.
    """

    def __init__(
        self,
        datasets=None,
        initial_model_params=None,
        parameters_to_fit=None,
        sigmas=None,
        mag_methods=None,
        mag_methods_parameters=None,
        model_config=None,
        event_config=None,
        verbose=False,
        pool=None,
    ):
        self._initial_model = None
        self._best = None
        self._results = None

        self.datasets = datasets

        self.initial_model_params = initial_model_params
        self.parameters_to_fit = parameters_to_fit
        self.sigmas = sigmas

        self.mag_methods = mag_methods
        self.mag_methods_parameters = mag_methods_parameters
        self.model_config = (
            model_config if model_config is not None else ModelConfig()
        )
        self.event_config = (
            event_config if event_config is not None else EventConfig()
        )

        self.verbose = verbose
        self.pool = pool

    def run(self):
        """
        Run the fitter. Implemented by subclasses.

        Notes
        -----
        This method is not formally declared as abstract but is intended to be
        overridden by subclasses. Consider using ``abc.abstractmethod`` to enforce
        this, consistent with ``set_event_parameters()`` and
        ``make_starting_vector()`` in ``AnomalyFitter``.
        """
        pass

    def get_model(self):
        """
        Create a MulensModel.Model with best-fit parameters, or initial
        parameters if no fit has been run yet.

        Magnification methods and limb darkening coefficients are applied
        via ``model_config``, which is built from the constructor arguments
        at initialization time.

        Returns
        -------
        MulensModel.Model
            Configured model with best-fit or initial parameters.

        See Also
        --------
        get_event : Creates a MulensModel.Event using this model.
        """
        if self.best is not None:
            params = dict(self.best)
            params.pop("chi2", None)
        else:
            params = self.initial_model_params

        return self.model_config.build(
            parameters=params,
            magnification_methods=self.mag_methods,
            magnification_methods_parameters=self.mag_methods_parameters,
        )

    def get_event(self):
        """
        Create a MulensModel.Event using the current datasets and model.

        Coordinates and flux-fixing are applied via ``event_config``, which
        is built from the constructor arguments at initialization time.

        Returns
        -------
        MulensModel.Event
            Event constructed from the current fitter state.

        See Also
        --------
        get_model : Creates the MulensModel.Model used by this event.
        """
        return self.event_config.build(
            model=self.get_model(),
            datasets=self.datasets,
        )

    def get_diagnostic_str(self):
        """
        Build a diagnostic string summarising the current event fit.

        Calls ``fit_fluxes()`` on the event before building the string. The
        returned string includes the model parameters, and for each dataset: the
        label, number of good data points, chi2, source flux(es), and blend flux.

        Returns
        -------
        str
            Formatted string containing event and dataset fit information.

        Notes
        -----
        Despite the name, this method does not print anything. Use
        ``print(get_diagnostic_str())`` to print the output.
        """
        event = self.get_event()
        event.fit_fluxes()
        msg = f"\n---- Event Info ----\nModel:\n{event.model}\n\nDatasets:"
        msg += "\n{0:20} {1:>4} {2:>12} {3} {4}".format(
            "Label", "N_good", "chi2", "f_source", "f_blend"
        )
        for i, dataset in enumerate(event.datasets):
            msg += "\n{0:20} {1:4} {2:12.2f} {3} {4}".format(
                dataset.plot_properties["label"],
                np.sum(dataset.good),
                event.get_chi2_for_dataset(i),
                event.fits[i].source_fluxes,
                event.fits[i].blend_flux,
            )

        msg += "\n--------------------\n"
        return msg

    @property
    def best(self):
        """
        Best-fit model parameters and chi2.

        Returns
        -------
        dict or None
            Dictionary of best-fit model parameter names and values, with an
            additional ``'chi2'`` key. Returns None if no fit has been run yet.
        """
        return self._best

    @best.setter
    def best(self, params_dict):
        self._best = params_dict

    @property
    def results(self):
        """
        Full results object from the fitter.

        Returns
        -------
        dict or None
            Full results from the fitting routine. The structure depends on the
            subclass: for example, ``SFitFitter`` stores the result object returned
            by ``sfit.minimize()``. Returns None if no fit has been run yet.

        Notes
        -----
        For the best-fit model parameters specifically, use ``best`` instead.
        """
        return self._results

    @results.setter
    def results(self, value):
        self._results = value

    @property
    def initial_model_params(self):
        """
        Initial model parameters used as the starting point for the fit.

        Parameters
        ----------
        params_dict : dict or None
            Dictionary of model parameter names and values. Must be a dict or
            None; raises ValueError otherwise.

        Returns
        -------
        dict or None
            Dictionary of model parameter names and values.

        Raises
        ------
        ValueError
            If set with a value that is neither None nor a dict.
        """
        return self._initial_model_params

    @initial_model_params.setter
    def initial_model_params(self, params_dict):
        if (params_dict is not None) and (not isinstance(params_dict, dict)):
            raise ValueError(
                "initial_model must be set with either *None* or *dict*."
            )

        self._initial_model_params = params_dict

    @property
    def success(self):
        """
        Boolean indicating whether the fit was successful.

        Returns
        -------
        bool
            True if the fit was successful, False otherwise.
        Notes
        -----
        The definition of "success" may vary between subclasses. For example,
        ``SFitFitter`` considers a fit successful if ``sfit.minimize()`` returns not None.

        """
        if self.results is None:
            return False
        else:
            return True

    @property
    def parameters_to_fit(self):
        """
        List of model parameters to be fitted.

        If not explicitly set, defaults to all keys in ``initial_model_params``.

        Parameters
        ----------
        params_dict : list, tuple, or None
            Names of parameters to fit. Must be a list, tuple, or None; raises
            ValueError otherwise. If None, all keys in ``initial_model_params``
            will be fitted.

        Returns
        -------
        list
            Names of parameters to be fitted.

        Raises
        ------
        ValueError
            If set with a value that is neither None, a list, nor a tuple.
        """
        if self._parameters_to_fit is None:
            self._parameters_to_fit = list(self.initial_model_params.keys())

        return self._parameters_to_fit

    @parameters_to_fit.setter
    def parameters_to_fit(self, params_dict):
        if (params_dict is not None) and (
            not isinstance(params_dict, (list, tuple))
        ):
            raise ValueError(
                "parameters_to_fit must be set with either *None* or *list* or *tuple*."
            )

        self._parameters_to_fit = params_dict


class SFitFitter(MulensFitter):
    """
    Fit a point lens model to the data using the SFit method.

    Wraps ``sfit.minimize()`` with a ``PointLensSFitFunction``. First attempts
    the fit with an adaptive step size; if unsuccessful, retries with a fixed
    step size of 0.001 and a maximum of 10000 iterations.

    All parameters are inherited from ``MulensFitter``.

    See Also
    --------
    MulensFitter : Parent class defining all constructor parameters.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self):
        """
        Fit the point lens model using the SFit method.

        Constructs an initial guess vector from ``initial_model_params`` and the
        source and blend fluxes from an initial ``fit_fluxes()`` call. First
        attempts minimization with an adaptive step size; if unsuccessful, retries
        with a fixed step size of 0.001 and a maximum of 10000 iterations.

        On completion, sets ``results`` to the full ``sfit.minimize()`` result
        object and ``best`` to the best-fit model parameters plus ``'chi2'``.

        Notes
        -----
        ``event.fits[i].source_flux`` should be ``event.fits[i].source_fluxes``
        for MMv3, where source fluxes are returned as an array. The way the value
        is subsequently appended to ``initial_guess`` may also need revisiting.
        See also ``get_diagnostic_str()``, which correctly uses ``source_fluxes``.
        """
        event = self.get_event()
        event.fit_fluxes()

        my_func = sfit.mm_funcs.PointLensSFitFunction(
            event, self.parameters_to_fit
        )

        initial_guess = [
            self.initial_model_params[key] for key in self.parameters_to_fit
        ]
        for i in range(len(self.datasets)):
            initial_guess.append(event.fits[i].source_flux)
            initial_guess.append(event.fits[i].blend_flux)

        try:
            result = sfit.minimize(
                my_func,
                x0=initial_guess,
                tol=1e-5,
                options={"step": "adaptive"},
                verbose=self.verbose,
            )
        except Exception as e:
            logger.warning(
                f"Error during sfit.minimize with adaptive step: {e}"
            )
            result = None
        if self.verbose:
            print(result)

        if result is None or not result.success:
            try:
                result = sfit.minimize(
                    my_func,
                    x0=initial_guess,
                    tol=1e-5,
                    max_iter=10000,
                    options={"step": 0.001},
                    verbose=self.verbose,
                )
            except Exception as e:
                logger.warning(
                    f"Error during sfit.minimize with fixed step: {e}"
                )
                result = None

            if self.verbose:
                print(result)

        self.results = result
        best = my_func.event.model.parameters.parameters
        best["chi2"] = my_func.event.get_chi2()
        self.best = best


class EmceeLCFitter(MulensFitter):
    """
    Pure emcee mechanism for fitting microlensing light curves.

    Extends :class:`MulensFitter` with MCMC sampling via emcee.  The starting
    ensemble is built by Gaussian perturbation of ``initial_guess`` scaled by
    ``sigmas``.  All domain-specific policy — parameter estimation, ensemble
    initialisation, log-space defaults — lives in subclasses.

    Parameters
    ----------
    initial_guess : dict, optional
        Starting parameter values keyed by parameter name.  Keys may carry a
        ``log_`` prefix (e.g. ``'log_rho'``) to indicate sampling in log10
        space; :meth:`initialize_event` converts them back to linear space.
        If ``parameters_to_fit`` is not supplied it defaults to
        ``list(initial_guess.keys())``.
    emcee_settings : dict, optional
        Settings for the emcee sampler.  Missing keys are filled from
        :attr:`default_emcee_settings`.  Valid keys: ``'n_walkers'``,
        ``'n_burn'``, ``'n_steps'``, ``'acceptance_fraction'``, and
        optionally ``'temperature'``.  ``'n_dim'`` is added automatically
        from ``len(parameters_to_fit)`` once that list is finalised.

    Class Attributes
    ----------------
    default_emcee_settings : dict
        ``{'n_walkers': 40, 'n_burn': 500, 'n_steps': 1000,
        'acceptance_fraction': 0.1}``

    Notes
    -----
    After ``event.get_chi2()`` has been evaluated the ``MulensModel.Event``
    object cannot be pickled.  :meth:`__getstate__` resets ``_event`` to
    ``None`` before serialisation; call :meth:`initialize_event` again after
    unpickling if needed.

    Subclasses that finalise ``parameters_to_fit`` *after*
    ``super().__init__()`` are responsible for updating ``'n_dim'`` in
    :attr:`emcee_settings` themselves.

    All other parameters (``datasets``, ``sigmas``, ``model_config``,
    ``event_config``, ``pool``, …) are inherited from :class:`MulensFitter`.
    """

    default_emcee_settings = {
        "n_walkers": 40,
        "n_burn": 500,
        "n_steps": 1000,
        "acceptance_fraction": 0.1,
    }

    # TODO: Implement all the fancy diagnostic plots from Radek's Example 16.
    def __init__(self, initial_guess=None, emcee_settings=None, **kwargs):
        super().__init__(**kwargs)
        self.initial_guess = initial_guess
        self._event = None

        # Default parameters_to_fit to initial_guess key
        # provided via kwargs → MulensFitter.__init__.
        if self._parameters_to_fit is None and initial_guess is not None:
            self.parameters_to_fit = list(initial_guess.keys())

        # Merge user-supplied settings on top of class defaults.
        settings = dict(self.default_emcee_settings)
        if emcee_settings is not None:
            settings.update(emcee_settings)
        self.emcee_settings = settings

        # Register n_dim now if parameters_to_fit is already known.
        # Subclasses that finalise parameters_to_fit *after* super().__init__()
        # must update 'n_dim' themselves.
        if self._parameters_to_fit is not None:
            self.emcee_settings.setdefault(
                "n_dim", len(self._parameters_to_fit)
            )

    # ------------------------------------------------------------------ #
    # Pickling support                                                     #
    # ------------------------------------------------------------------ #

    def __getstate__(self):
        """
        Return the object state for pickling with ``_event`` set to ``None``.

        ``MulensModel.Event`` objects cannot be pickled after ``get_chi2()``
        has been called.  Call :meth:`initialize_event` after unpickling to
        rebuild the event.

        Returns
        -------
        dict
            ``__dict__`` with ``_event`` replaced by ``None``.
        """
        state = self.__dict__.copy()
        state["_event"] = None
        return state

    def __setstate__(self, state):
        """
        Restore the object state after unpickling.

        Parameters
        ----------
        state : dict
            State dictionary produced by :meth:`__getstate__`.
        """
        self.__dict__.update(state)

    # ------------------------------------------------------------------ #
    # Parameter helpers                                                    #
    # ------------------------------------------------------------------ #

    def get_parameter_name(self, parameter):
        """
        Strip the ``log_`` prefix from a parameter name if present.

        Parameters
        ----------
        parameter : str
            Parameter name, e.g. ``'log_rho'`` or ``'t_E'``.

        Returns
        -------
        str
            Name with ``log_`` prefix removed, e.g. ``'rho'`` or ``'t_E'``.
        """
        if parameter.startswith("log_"):
            return parameter[4:]
        return parameter

    def make_emcee_vector_from_ModelParameters(self, parameters):
        """
        Convert a ``MulensModel.ModelParameters`` object to an emcee vector.

        Parameters with a ``log_`` prefix are converted to log10 space.

        Parameters
        ----------
        parameters : MulensModel.ModelParameters
            Model parameters to convert.

        Returns
        -------
        list
            Vector of length ``len(parameters_to_fit)``.

        See Also
        --------
        event.setter : Performs the inverse conversion.
        """
        vector = []
        for parameter in self.parameters_to_fit:
            key = self.get_parameter_name(parameter)
            value = getattr(parameters, key)
            if key != parameter:  # log_ prefix
                value = np.log10(value)
            vector.append(value)
        return vector

    # ------------------------------------------------------------------ #
    # Event initialisation                                                 #
    # ------------------------------------------------------------------ #

    def initialize_event(self):
        """
        Initialise the ``MulensModel.Event`` from :attr:`initial_guess`.

        Parameters with a ``log_`` prefix are converted from log10 to linear
        space before the model is constructed.  Coordinates and flux-fixing
        are applied via :attr:`model_config` and :attr:`event_config`.

        Raises
        ------
        AttributeError
            If :attr:`initial_guess` is ``None``.
        """
        if self.initial_guess is None:
            raise AttributeError(
                "initial_guess must be set before calling initialize_event()."
            )

        params = {}
        for key, value in self.initial_guess.items():
            actual_key = self.get_parameter_name(key)
            params[actual_key] = 10.0**value if actual_key != key else value

        model = self.model_config.build(
            parameters=params,
            magnification_methods=self.mag_methods,
            magnification_methods_parameters=self.mag_methods_parameters,
        )
        self._event = self.event_config.build(
            model=model,
            datasets=self.datasets,
        )

    def get_best_fit_event(self):
        """
        Return the event evaluated at the best-fit (max-likelihood) parameters
        with fluxes fitted.

        Initialises the event from ``initial_guess`` if it has not yet been
        created. Sets the model parameters to ``best_theta`` and calls
        ``fit_fluxes()`` before returning.

        Returns
        -------
        MulensModel.Event
            Event at the max-likelihood parameters with fluxes fitted.
        """
        if self._event is None:
            self.initialize_event()
        self.event = self.best_theta
        self._event.fit_fluxes()
        return self._event

    # ------------------------------------------------------------------ #
    # Starting ensemble                                                    #
    # ------------------------------------------------------------------ #

    def make_starting_vector(self):
        """
        Build the emcee starting ensemble by Gaussian perturbation of
        :attr:`initial_guess`.

        Walker 0 is the UNPERTURBED :attr:`initial_guess`; every other walker
        adds independent ``N(0, sigma)`` noise to each component.  Parameters
        absent from :attr:`sigmas` (or when :attr:`sigmas` is ``None``) are
        left at their nominal value for every walker.

        Keeping the seed itself in the ensemble matters because the caller
        treats this fit as a POLISH of a solution estimate.  When every
        walker was perturbed, the seed was not in the ensemble at all, so a
        run whose walkers all drifted downhill could return a point far worse
        than the one it was given -- measured on the 2018 Data Challenge
        event 128, where a seed at chi2 37988 came back at 147227.  With
        walker 0 at the seed, the seed is a live member the stretch move can
        be compared against, and :meth:`run` additionally guarantees the
        reported best is never worse than the starting ensemble.

        Returns
        -------
        list of list
            ``n_walkers`` parameter vectors of length ``n_dim``.
        """
        n_walkers = self.emcee_settings["n_walkers"]
        starting_vector = []
        for i_walker in range(n_walkers):
            walker = []
            for emcee_param in self.parameters_to_fit:
                param = self.get_parameter_name(emcee_param)
                if param == emcee_param:
                    value = self.initial_guess[param]
                else:
                    value = np.log10(self.initial_guess[param])

                sigma = (self.sigmas or {}).get(emcee_param)

                if sigma is not None and i_walker > 0:
                    value = value + np.random.normal(0.0, sigma)

                walker.append(value)
            starting_vector.append(walker)
        return starting_vector

    # ------------------------------------------------------------------ #
    # Event property                                                       #
    # ------------------------------------------------------------------ #

    @property
    def event(self):
        """
        The current ``MulensModel.Event`` object.

        Returns ``None`` until :meth:`initialize_event` has been called
        (or after unpickling).
        """
        return self._event

    @event.setter
    def event(self, theta):
        """
        Update model parameters from an emcee parameter vector.

        Parameters with a ``log_`` prefix are converted from log10 to linear
        space before assignment.

        Parameters
        ----------
        theta : array-like
            Vector of length ``len(parameters_to_fit)``.

        Raises
        ------
        AttributeError
            If the event has not yet been initialised.

        See Also
        --------
        make_emcee_vector_from_ModelParameters : Performs the inverse
            conversion.
        """
        if self._event is None:
            # Lazily (re)build rather than raise: __getstate__ drops _event
            # (it cannot be pickled), so a multiprocessing pool worker --
            # emcee pickles ln_prob's bound fitter into the workers -- arrives
            # here with _event of None on its first likelihood call. Each
            # worker builds its own Event once and reuses it afterward.
            self.initialize_event()
        for parameter, value in zip(self.parameters_to_fit, theta):
            key = self.get_parameter_name(parameter)
            if key != parameter:  # log_ prefix
                value = 10.0**value
            self._event.model.parameters.__setattr__(key, value)

    # ------------------------------------------------------------------ #
    # Likelihood / prior / probability                                     #
    # ------------------------------------------------------------------ #

    def ln_like(self, theta):
        """
        Log-likelihood for the emcee sampler (``-0.5 * chi2``).

        If ``'temperature'`` is present in :attr:`emcee_settings`, chi2 is
        divided by the square of the temperature (simulated annealing).

        Parameters
        ----------
        theta : array-like

        Returns
        -------
        float
            Log-likelihood, or ``-np.inf`` if evaluation failed.

        Notes
        -----
        The bare ``except`` clause silently catches all exceptions; consider
        catching specific exceptions to ease debugging.
        """
        self.event = theta
        try:
            chi2 = self.event.get_chi2()
            if "temperature" in self.emcee_settings:
                chi2 /= self.emcee_settings["temperature"] ** 2
        except Exception:
            return -np.inf
        return -0.5 * chi2

    def ln_prior(self, theta):
        """
        Log-prior: flat with hard lower bounds on positive-definite parameters.

        Rejects models where ``t_E``, ``rho``, ``q``, or ``s`` are
        non-positive.

        Parameters
        ----------
        theta : array-like

        Returns
        -------
        float
            ``0.0`` if all priors are satisfied; ``np.inf`` otherwise.

        Notes
        -----
        Returns ``np.inf`` (rather than ``-np.inf``) for rejected models;
        this is handled correctly by :meth:`ln_prob`.
        """
        for key, value in zip(self.parameters_to_fit, theta):
            if key in ("t_E", "rho", "q", "s") and value <= 0.0:
                return np.inf
        return 0.0

    def ln_prob(self, theta):
        """
        Log-probability: prior + likelihood.

        Returns ``-np.inf`` when the prior is violated, when the likelihood
        cannot be evaluated, or when the likelihood is NaN (e.g. negative
        source fluxes).

        Parameters
        ----------
        theta : array-like

        Returns
        -------
        float

        See Also
        --------
        ln_prior : Log-prior function.
        ln_like : Log-likelihood function.
        """
        ln_prior_ = self.ln_prior(theta)
        if not np.isfinite(ln_prior_):
            return -np.inf
        ln_like_ = self.ln_like(theta)
        if np.isnan(ln_like_):
            return -np.inf
        return ln_prior_ + ln_like_

    # ------------------------------------------------------------------ #
    # Runner                                                               #
    # ------------------------------------------------------------------ #

    def run(self, verbose=False):
        """
        Fit the model using emcee MCMC sampling.

        Calls :meth:`make_starting_vector` to build the ensemble (subclass
        implementations may initialise the event as a side-effect).  If the
        event is still uninitialised after that call, :meth:`initialize_event`
        is invoked.  Sampling aborts early if the mean acceptance fraction
        drops below ``emcee_settings['acceptance_fraction']``.

        On completion, sets :attr:`best` to the highest-probability point the
        fit visited anywhere -- full chain including burn-in, plus the
        starting ensemble -- together with its ``'chi2'``.  The result is
        therefore never worse than the point the fitter was seeded with.
        :attr:`results` still derives percentiles and sigmas from the
        post-burn-in chain alone.

        Parameters
        ----------
        verbose : bool, optional
            If ``True``, prints 16th/50th/84th-percentile summaries and logs
            the acceptance fraction every 100 steps.  Default ``False``.

        Returns
        -------
        str or None
            If None, the run complected successfully. Otherwise, str
            explains what went wrong.
        """
        starting_vector = self.make_starting_vector()

        # Kept so the best-fit contract below is inspectable after the run:
        # the reported best is the argmax over the full chain AND this
        # ensemble, and a caller (or a test) cannot check that without it.
        self.starting_vector_used = starting_vector

        # Subclass make_starting_vector() may initialise the event as a
        # side-effect; only call initialize_event() when that has not happened.
        if self._event is None:
            self.initialize_event()

        if self.pool:
            # `pool` as an int caps the process count (essential on shared
            # cluster nodes, where cpu_count() sees the whole node rather
            # than the job's slot grant); any other truthy value keeps the
            # historical one-process-per-CPU behavior. Note emcee's stretch
            # move evaluates half the ensemble per batch, so more than
            # n_walkers/2 processes cannot help.
            n_proc = (
                self.pool
                if isinstance(self.pool, int)
                and not isinstance(self.pool, bool)
                else cpu_count()
            )
            n_proc = min(n_proc, max(1, self.emcee_settings["n_walkers"] // 2))
            print("{0} CPUs ({1} pool processes)".format(cpu_count(), n_proc))
            os.environ["OMP_NUM_THREADS"] = "1"
            # initializer ships the fitter to each worker exactly once;
            # _pool_ln_prob is what emcee pickles per call (a no-op by
            # reference). See the module-level note above.
            pool = Pool(
                processes=n_proc,
                initializer=_pool_worker_init,
                initargs=(self,),
            )
            self.sampler = emcee.EnsembleSampler(
                self.emcee_settings["n_walkers"],
                self.emcee_settings["n_dim"],
                _pool_ln_prob,
                pool=pool,
            )
        else:
            self.sampler = emcee.EnsembleSampler(
                self.emcee_settings["n_walkers"],
                self.emcee_settings["n_dim"],
                self.ln_prob,
            )

        try:
            mean_af = 1.0
            for _ in self.sampler.sample(
                starting_vector, iterations=self.emcee_settings["n_steps"]
            ):
                # TODO: ADD option to periodically output results.
                if (self.sampler.iteration % 100) != 0:
                    continue
                if self.emcee_settings.get("acceptance_fraction") is None:
                    continue

                mean_af = np.mean(self.sampler.acceptance_fraction)

                if verbose:
                    print(
                        self.sampler.iteration,
                        "{0:.3f}".format(mean_af),
                        self.sampler.acceptance_fraction,
                    )

                if mean_af < self.emcee_settings["acceptance_fraction"]:
                    break

        finally:
            if self.pool:
                pool.close()
                pool.join()

        n_burn = self.emcee_settings["n_burn"]
        n_dim = self.emcee_settings["n_dim"]
        samples = self.sampler.chain[:, n_burn:, :].reshape((-1, n_dim))

        if len(samples) == 0:
            msg = (
                "No samples remain after burn-in. "
                + f"Sampler ran {self.sampler.iteration} iterations but "
                + f"n_burn={n_burn}."
            )
            if mean_af < self.emcee_settings["acceptance_fraction"]:
                msg += f"Acceptance fraction {mean_af:.3f} too low! Minimum set to:"
                msg += f" {self.emcee_settings['acceptance_fraction']}"

            return msg

        if verbose:
            percentiles = np.percentile(samples, [16, 50, 84], axis=0)
            print("Fitted parameters:")
            for i in range(n_dim):
                med = percentiles[1, i]
                print(
                    "${:.5f}^{{+{:.5f}}}_{{-{:.5f}}}$ &".format(
                        med, percentiles[2, i] - med, med - percentiles[0, i]
                    )
                )

        # Best-fit point: the highest-probability point ANYWHERE this fit has
        # been -- the full chain (burn-in included) and the starting ensemble
        # itself -- not just the post-burn-in samples.
        #
        # Selecting over `[:, n_burn:]` alone made the returned best-fit
        # unbounded from below with respect to the input: burn-in is where a
        # polish does most of its improving, and if the walkers subsequently
        # wandered to a worse region and stayed there, the reported point was
        # whatever they had settled on.  On 2018 Data Challenge event 128 that
        # turned a seed at chi2 37988 into a "fit" at 147227.  For a routine
        # whose job is to refine a solution estimate, coming back worse than
        # the input is never the right answer.
        #
        # Including the starting ensemble is what makes that a guarantee
        # rather than a strong tendency: walker 0 is the unperturbed seed (see
        # make_starting_vector), but emcee's stretch move may Metropolis-accept
        # a step away from it, and the initial state is not itself stored in
        # the chain.  n_walkers extra likelihood calls against n_walkers *
        # n_steps in the run is a rounding error.
        #
        # Percentiles/sigmas still come from the post-burn-in chain only
        # (EmceeFitResults) -- those describe the posterior, and burn-in has
        # no business in them.  This is a point estimate, which is different.
        n_walkers_started = len(starting_vector)
        start_theta = np.asarray(starting_vector, dtype=float)
        start_prob = np.array(
            [self.ln_prob(theta) for theta in start_theta], dtype=float
        )
        cand_theta = np.vstack(
            [self.sampler.chain.reshape((-1, n_dim)), start_theta]
        )
        cand_prob = np.concatenate(
            [self.sampler.lnprobability.reshape((-1)), start_prob]
        )
        cand_prob = np.where(np.isfinite(cand_prob), cand_prob, -np.inf)
        best_index = int(np.argmax(cand_prob))
        self.best_theta = cand_theta[best_index]
        if best_index >= len(cand_prob) - n_walkers_started:
            logger.info(
                "emcee polish did not improve on its starting ensemble; "
                "returning the best starting point (walker %d).",
                best_index - (len(cand_prob) - n_walkers_started),
            )
        self.event = self.best_theta

        self.best = self._event.model.parameters.parameters
        self.best["chi2"] = self._event.get_chi2()

        self.results = EmceeFitResults(
            sampler=self.sampler,
            x=self.best_theta,
            emcee_settings=self.emcee_settings,
            parameters_to_fit=self.parameters_to_fit,
        )

class LSFitter(EmceeLCFitter):
    """
    Fit a microlensing light curve using the Levenberg-Marquardt algorithm.

    Extends :class:`EmceeLCFitter` with a :meth:`run` method that uses
    ``scipy.optimize.minimize()`` to fit the model to the data.  The
    starting point is taken from :attr:`initial_guess`, and the fitted
    parameters are stored in :attr:`best` along with the corresponding
    ``'chi2'``.

    All other parameters (``datasets``, ``sigmas``, ``model_config``,
    ``event_config``, ``pool``, …) are inherited from :class:`EmceeLCFitter`
    and :class:`MulensFitter`.
    """
    def __init__(self, bounds=None, **kwargs):
        super().__init__(**kwargs)
        self.bounds = bounds
        self.parse_bounds()

    def parse_bounds(self):
        """
        Parse the bounds provided by the user and set default bounds for parameters.

        If no bounds are provided, default bounds are set for each parameter.
        """
        if isinstance(self.bounds, dict):
            lower_bounds = []
            upper_bounds = []
            for param in self.parameters_to_fit:
                if param in self.bounds:
                    lower_bounds.append(self.bounds[param][0])
                    upper_bounds.append(self.bounds[param][1])
                else:
                    # Set default bounds if not provided
                    lower_bounds.append(-np.inf)
                    upper_bounds.append(np.inf)
            self.bounds = (lower_bounds, upper_bounds)

        if not isinstance(self.bounds, tuple) or len(self.bounds) != 2:
            raise ValueError("Bounds must be a tuple of two lists (lower_bounds, upper_bounds).")
        if len(self.bounds[0]) != len(self.parameters_to_fit) or len(self.bounds[1]) != len(self.parameters_to_fit):
            raise ValueError("Bounds must have the same length as parameters_to_fit.")

    def run(self, verbose=False):
        """
        Fit the model using the scipy.optimize.minimize algorithm.

        The negative log probability is minimized, starting from
        :attr:`initial_guess`. The fitted parameters are stored
        in :attr:`best` along with the corresponding ``'chi2'``.

        Parameters
        ----------
        verbose : bool, optional
            If ``True``, prints the optimization result summary. Default is
            ``False``.
        """
        if self._event is None:
            self.initialize_event()

        initial_vector = self.make_emcee_vector_from_ModelParameters(
            self._event.model.parameters
        )
        print(f"initial vector: {initial_vector}")
        result = minimize(
            lambda theta: -self.ln_prob(theta),
            x0=initial_vector,
            method="L-BFGS-B",
            bounds=list(zip(*self.bounds)),
            options={
                "maxiter": 100,
                "ftol": 1e-4,
            })
        self.best_theta = np.asarray(result.x)
        self.event = self.best_theta

        self.best = self._event.model.parameters.parameters
        self.best["chi2"] = self._event.get_chi2()
        self.results = dict(
            best=self.best,
            chi2=self.best["chi2"],
            parameters_to_fit=self.parameters_to_fit,
        )
        print(f"Best-fit parameters: {self.results['best']}")

        return result


class AnomalyFitter(EmceeLCFitter):
    """
    Emcee fitter with automatic sigma estimation for microlensing anomaly fitting.

    Extends :class:`EmceeLCFitter` by adding a :attr:`default_parameters_to_fit`
    and automatic :attr:`sigmas` computation when sigmas are not explicitly
    provided.

    Unlike :class:`EmceeLCFitter`, ``initial_guess`` is required and
    ``parameters_to_fit`` defaults to :attr:`default_parameters_to_fit` rather
    than ``initial_guess.keys()``.  Users providing a custom ``initial_guess``
    must therefore include values for every parameter in
    :attr:`default_parameters_to_fit`, or supply ``parameters_to_fit``
    explicitly to restrict or reorder the fitted set.

    Sigma estimation follows a three-tier priority:

    1. **Explicit** — if ``sigmas`` is provided it is used unchanged.

    2. **From** ``anomaly_lc_params`` — if ``sigmas`` is not provided but
       ``anomaly_lc_params`` is, then for ``t_0`` and ``t_E``::

           sigma_t0 = sigma_tE = min(0.01 * anomaly_lc_params['dt'], default_sigma[param])

       All other parameters fall through to tier 3.

    3. **Defaults from** ``initial_guess``::

           t_0   :  0.00001
           u_0   :  0.001 * |u_0|
           t_E   :  0.001 * |t_E|
           alpha :  0.0001
           log_X :  0.0001    # equivalent to sigma_X ~ 0.02% of X

       Parameters not matching any recognised pattern are omitted from
       ``sigmas``; the corresponding walkers are not perturbed in that
       dimension by :meth:`EmceeLCFitter.make_starting_vector`.

    Class Attributes
    ----------------
    default_parameters_to_fit : list
        ``['t_0', 'u_0', 't_E', 'log_rho', 'log_s', 'log_q', 'alpha']``

    Parameters
    ----------
    initial_guess : dict
        Starting parameter values keyed by parameter name.  Required.  Must
        contain a value for every parameter in :attr:`default_parameters_to_fit`
        (or every parameter in ``parameters_to_fit`` if that is supplied
        explicitly).  Keys may carry a ``log_`` prefix for parameters sampled
        in log10 space.
    anomaly_lc_params : dict, optional
        Parameters describing the anomaly light curve, as returned by
        ``AnomalyPropertyEstimator.get_anomaly_lc_parameters()``.  Used only
        to derive sigmas for ``t_0`` and ``t_E`` via the ``'dt'`` key.  Has
        no effect on ``initial_guess`` and no effect when ``sigmas`` is already
        provided explicitly.
    emcee_settings : dict, optional
        Passed through to :class:`EmceeLCFitter`.

    Notes
    -----
    :attr:`sigmas` are never derived from ``anomaly_lc_params`` directly.
    The typical source for PSPL parameter sigmas is the result of a prior PSPL
    fit (e.g. ``pspl_results.sigmas``), passed as explicit ``sigmas``.

    Subclasses that build ``initial_guess`` lazily (e.g.
    :class:`AnomalyFitterEnsembleInitialization`) should pass
    ``initial_guess=None`` to suppress automatic sigma computation, handling
    it themselves once ``initial_guess`` is available.
p_tru = [230, -0.4, 2.0, 0.1]

    All other parameters (``datasets``, ``sigmas``, ``model_config``,
    ``event_config``, ``pool``, …) are inherited from :class:`EmceeLCFitter`
    and :class:`MulensFitter`.

    See Also
    --------
    EmceeLCFitter : Parent class providing the full emcee mechanism.
    AnomalyFitterEnsembleInitialization : Subclass using
        ``WideAxisEnsembleInitializer`` for the starting ensemble.
    """

    default_parameters_to_fit = [
        "t_0",
        "u_0",
        "t_E",
        "log_rho",
        "log_s",
        "log_q",
        "alpha",
    ]

    def __init__(
        self,
        initial_guess,
        anomaly_lc_params=None,
        emcee_settings=None,
        **kwargs,
    ):
        super().__init__(
            initial_guess=initial_guess,
            emcee_settings=emcee_settings,
            **kwargs,
        )
        if self.mag_methods is None:
            raise ValueError("mag_methods must be provided for AnomalyFitter.")

        self.anomaly_lc_params = anomaly_lc_params

        # Use default_parameters_to_fit when not explicitly provided.
        # This overrides EmceeLCFitter's default of list(initial_guess.keys()),
        # so initial_guess must contain values for all default parameters.
        if "parameters_to_fit" not in kwargs:
            self.parameters_to_fit = list(self.default_parameters_to_fit)
            # n_dim must reflect the updated parameters_to_fit.  Direct
            # assignment rather than setdefault ensures a stale value from
            # EmceeLCFitter (set from initial_guess.keys()) is overwritten.
            self.emcee_settings["n_dim"] = len(self.parameters_to_fit)

        # Compute sigmas only when not explicitly provided and initial_guess
        # is available.  The guard on initial_guess allows subclasses that
        # build initial_guess lazily to pass None and defer sigma computation.
        if self.sigmas is None and self.initial_guess is not None:
            self.sigmas = self._compute_sigmas()

    def _compute_sigmas(self):
        """
        Compute sigmas following the three-tier priority.

        Starts from :meth:`_default_sigmas`, then overrides ``t_0`` and
        ``t_E`` from ``anomaly_lc_params['dt']`` if ``anomaly_lc_params`` is
        set.

        Returns
        -------
        dict
            Sigma for each recognised parameter in :attr:`parameters_to_fit`.
        """
        sigmas = self._default_sigmas()

        if self.anomaly_lc_params is not None:
            dt = self.anomaly_lc_params["dt"]
            if "t_0" in self.parameters_to_fit:
                sigmas["t_0"] = min(sigmas["t_0"], 0.01 * dt)
            if "t_E" in self.parameters_to_fit:
                sigmas["t_E"] = min(sigmas["t_E"], 0.01 * dt)

        return sigmas

    def _default_sigmas(self):
        """
        Compute default sigmas from :attr:`initial_guess`.

        For ``log_X`` parameters the step size is derived so that the
        equivalent linear-space step equals 0.01% of the parameter value:

        .. math::

            \\sigma_{\\log X} = \\frac{\\sigma_X}{X \\ln 10}
                              = \\frac{0.0001}{\\ln 10}

        Parameters not matching any recognised pattern are omitted.

        Returns
        -------
        dict
            Default sigma for each recognised parameter in
            :attr:`parameters_to_fit`.
        """
        sigmas = {}
        for param in self.parameters_to_fit:
            if param == "t_0":
                sigmas[param] = 0.00001
            elif param == "u_0":
                sigmas[param] = 0.001 * abs(self.initial_guess["u_0"])
            elif param == "t_E":
                sigmas[param] = 0.001 * abs(self.initial_guess["t_E"])
            elif param == "alpha":
                sigmas[param] = 0.0001
            elif param.startswith("log_"):
                sigmas[param] = 0.0001

        return sigmas


# TODO: ADD child class of AnomalyFitter that uses dxsi parameter.


class WideAxisEnsembleInitialization(AnomalyFitter):
    """
    Anomaly fitter using ``WideAxisEnsembleInitializer`` for the starting
    ensemble.

    Extends :class:`AnomalyFitter` for the wide-planet geometry.  Overrides
    :meth:`make_starting_vector` to drive ``WideAxisEnsembleInitializer`` and
    :meth:`initialize_event` to use the resulting ``initial_model`` and
    ``mag_methods``.

    :attr:`default_parameters_to_fit` is inherited from :class:`AnomalyFitter`.
    ``parameters_to_fit`` and ``'n_dim'`` are finalised in the parent
    ``__init__``; this class adds no further changes to either.

    ``initial_guess`` is built lazily as a side-effect of the first call to
    :meth:`make_starting_vector`; ``None`` is passed to the parent to suppress
    automatic sigma computation.  Sigmas passed explicitly (typically PSPL fit
    sigmas) are forwarded to ``WideAxisEnsembleInitializer`` for PSPL
    perturbation; when none are provided they default to ``{}`` (no
    perturbation).

    Parameters
    ----------
    anomaly_lc_params : dict, optional
        Passed to ``WideAxisEnsembleInitializer``.  Also inspected by the
        parent :meth:`AnomalyFitter._compute_sigmas` — but since
        ``initial_guess`` is ``None`` at construction time, automatic sigma
        computation is suppressed and ``anomaly_lc_params`` is used only by
        the initialiser itself.
    emcee_settings : dict, optional
        Passed through to :class:`AnomalyFitter`.

    Notes
    -----
    ``initial_model`` and ``mag_methods`` are populated lazily as side-effects
    of the first call to :meth:`make_starting_vector` (triggered by
    :meth:`~EmceeLCFitter.run` or by accessing either property directly).

    All other parameters are inherited from :class:`AnomalyFitter` and
    :class:`MulensFitter`.

    See Also
    --------
    AnomalyFitter : Parent class providing default parameters and sigma tiers.
    EmceeLCFitter : Grandparent providing the full emcee run loop.
    WideAxisEnsembleInitializer : Builds the starting ensemble.
    """

    def __init__(self, anomaly_lc_params=None, emcee_settings=None, **kwargs):
        # Pass initial_guess=None to suppress automatic sigma computation;
        # initial_guess is built lazily inside make_starting_vector().
        super().__init__(
            initial_guess=None,
            anomaly_lc_params=anomaly_lc_params,
            emcee_settings=emcee_settings,
            **kwargs,
        )

        # Sigmas are forwarded to WideAxisEnsembleInitializer for PSPL
        # perturbation.  Default to {} (no perturbation) when not provided;
        # automatic sigma computation was suppressed above because
        # initial_guess=None.
        if self.sigmas is None:
            self.sigmas = {}

        self._initializer = None
        self._starting_vector = None
        self._initial_model = None
        self._mag_methods = None

    # ------------------------------------------------------------------ #
    # Lazy properties populated by make_starting_vector()                 #
    # ------------------------------------------------------------------ #

    @property
    def initial_model(self):
        """
        Initial model parameter dict; populated by :meth:`make_starting_vector`.

        Accessing this property before :meth:`make_starting_vector` has been
        called triggers full ensemble initialisation as a side-effect.
        """
        if self._initial_model is None:
            self.make_starting_vector()
        return self._initial_model

    @initial_model.setter
    def initial_model(self, value):
        self._initial_model = value

    @property
    def mag_methods(self):
        """
        Magnification-methods list; populated by :meth:`make_starting_vector`.

        Accessing this property before :meth:`make_starting_vector` has been
        called triggers full ensemble initialisation as a side-effect.
        """
        if self._mag_methods is None:
            self.make_starting_vector()
        return self._mag_methods

    @mag_methods.setter
    def mag_methods(self, value):
        self._mag_methods = value

    # ------------------------------------------------------------------ #
    # Overridden starting ensemble                                         #
    # ------------------------------------------------------------------ #

    def make_starting_vector(self):
        """
        Build the starting ensemble using ``WideAxisEnsembleInitializer``.

        Runs ``n_walkers`` ``WideAxisGridSearchEstimators`` with perturbed
        PSPL parameters, sorts results by chi2, and converts the best
        ``n_walkers`` rows to emcee parameter vectors.  Sets
        :attr:`initial_model` and :attr:`mag_methods` as side-effects and
        initialises the event if not already created.  Subsequent calls
        return the cached result without re-running the initialiser.

        Returns
        -------
        list of list
            ``n_walkers`` parameter vectors of length ``n_dim``.
        """
        if self._starting_vector is not None:
            return self._starting_vector

        self._initializer = WideAxisEnsembleInitializer(
            datasets=self.datasets,
            anomaly_params=self.anomaly_lc_params,
            sigmas=self.sigmas,
            n_estimators=self.emcee_settings["n_walkers"],
            pspl_chi2=getattr(self, "pspl_chi2", None),
        )

        # Store via setters so that the lazy properties resolve on subsequent
        # accesses without triggering another initialiser run.
        self.initial_model = self._initializer.initial_model
        self.mag_methods = self._initializer.mag_methods

        if self._event is None:
            self.initialize_event()

        df = self._initializer.results.sort_values("chi2")
        top_rows = df.head(self.emcee_settings["n_walkers"])

        starting_vector = []
        for _, row in top_rows.iterrows():
            params = {
                k: row[k]
                for k in ["t_0", "u_0", "t_E", "s", "q", "rho", "alpha"]
            }
            vector = self.make_emcee_vector_from_ModelParameters(
                MulensModel.ModelParameters(params)
            )
            starting_vector.append(vector)

        self._starting_vector = starting_vector
        return starting_vector

    # ------------------------------------------------------------------ #
    # Overridden event initialisation                                      #
    # ------------------------------------------------------------------ #

    def initialize_event(self):
        """
        Initialise the ``MulensModel.Event`` from :attr:`initial_model` and
        :attr:`mag_methods`.

        Both are set as side-effects of :meth:`make_starting_vector`.
        ``default_magnification_method`` is set to
        ``'point_source_point_lens'`` to cover regions outside the explicitly
        specified method windows.

        Raises
        ------
        AttributeError
            If :attr:`initial_model` or :attr:`mag_methods` has not been set
            (i.e. :meth:`make_starting_vector` has not been called yet).
        """
        if self._initial_model is None:
            raise AttributeError(
                "initial_model is not set. Call make_starting_vector() first."
            )
        if self._mag_methods is None:
            raise AttributeError(
                "mag_methods is not set. Call make_starting_vector() first."
            )

        model = self.model_config.build(
            parameters=self._initial_model,
            magnification_methods=self._mag_methods,
            default_magnification_method="point_source_point_lens",
        )
        self._event = self.event_config.build(
            model=model,
            datasets=self.datasets,
        )


class BellTemplateFitter:
    """
    Fit residuals with a simple multi-bell template.

    The bell centres are tied to the anomaly timing estimate and spread across
    the requested time span. The fit solves for one amplitude per bell, a
    shared width, and a constant offset.
    """

    def __init__(self, residuals, lc_parameters, n_bells=2, fit_centers=False):
        if isinstance(residuals, MulensModel.MulensData):
            residuals = [residuals]

        self.residuals = residuals
        self.lc_parameters = lc_parameters
        self.n_bells = max(int(n_bells), 1)
        self.fit_centers = bool(fit_centers)
        self.best = None
        self._fit_parameters = self._get_fit_parameters()

    def _bell_centers(self, t_pl, dt):
        if self.n_bells == 1:
            return np.array([t_pl], dtype=float)
        offsets = np.linspace(-0.5 * dt, 0.5 * dt, self.n_bells)
        return t_pl + offsets

    def _center_bounds(self, t_pl, dt):
        # Keep each center in its own equal-width share of the dt span.
        edges = np.linspace(t_pl - 0.5 * dt, t_pl + 0.5 * dt, self.n_bells + 1)
        return edges[:-1], edges[1:]

    @staticmethod
    def _template(
        time,
        centers=None,
        width=None,
        offset=None,
        amplitudes=None,
        dt=None,
        t_pl=None,
        **kwargs,
    ):
        width = max(float(width), 1.0e-6)
        model = np.full_like(time, offset, dtype=float)
        for amplitude, center in zip(amplitudes, centers):
            model += amplitude * np.exp(-0.5 * ((time - center) / width) ** 2)
        model[time < t_pl - 0.5 * dt] = 0
        model[time > t_pl + 0.5 * dt] = 0
        return model

    def _stack_pspl_residuals(self):
        times = []
        fluxes = []
        errors = []

        for dataset in self.residuals:
            good = getattr(dataset, "good", None)
            if good is None:
                good = np.ones(len(dataset.time), dtype=bool)

            times.append(np.asarray(dataset.time)[good])
            fluxes.append(np.asarray(dataset.flux)[good])
            errors.append(np.asarray(dataset.err_flux)[good])

        time = np.hstack(times)
        flux = np.hstack(fluxes)
        err = np.hstack(errors)
        order = np.argsort(time)
        return time[order], flux[order], err[order]

    @staticmethod
    def _fit_window(time, flux, err, t_pl, dt):
        half_width = 5 * max(float(dt), 1.0e-6)
        window = (time >= t_pl - half_width) & (time <= t_pl + half_width)
        if not np.any(window):
            return time, flux, err
        return time[window], flux[window], err[window]

    def _set_time_parameters(self, time):
        """Set initial guess for the fit parameters based on the residuals and lightcurve parameters."""
        t_pl = float(self.lc_parameters.get("t_pl", np.median(time)))
        dt_0 = float(abs(self.lc_parameters.get("dt", np.ptp(time) / 6.0)))
        dt_0 = max(dt_0, 1.0e-6)
        return t_pl, dt_0

    def _set_shape_parameters(self, t_pl, dt_0):
        """Set initial guess for the fit parameters based on the residuals and lightcurve parameters."""
        centers = self._bell_centers(t_pl, dt_0)
        baseline = 0.0
        amplitudes = [
            max(
                0.001,
                float(
                    self._flux[np.argmin(np.abs(self._time - center))]
                    - baseline
                ),
            )
            for center in centers
        ]
        width = (
            dt_0 / self.n_bells / self.n_bells if len(self._time) > 1 else dt_0
        )
        return centers, amplitudes, width, baseline

    def _set_initial_guess(self, amplitudes, centers, width, baseline, dt_0):
        """Set initial guess for the fit parameters based on the residuals and lightcurve parameters."""
        p0 = amplitudes
        if self.fit_centers:
            p0 += [float(value) for value in centers]
        p0 += [width, baseline, dt_0]
        # print(f"Initial guess: {p0}")
        return p0

    def _set_bounds(self, t_pl, dt_0):
        """Set bounds for the fit parameters"""

        lower_bounds = [0.0] * self.n_bells
        upper_bounds = [np.inf] * self.n_bells

        if self.fit_centers:
            center_low, center_high = self._center_bounds(t_pl, dt_0)
            lower_bounds += [float(value) - 1.0e-6 for value in center_low]
            upper_bounds += [float(value) + 1.0e-6 for value in center_high]
            lower_bounds += [1.0e-6, -np.inf, 0.5 * dt_0]
            upper_bounds += [dt_0 / 2, np.inf, 5.0 * dt_0]
        else:
            lower_bounds += [1.0e-6, -np.inf, 0.5 * dt_0]
            upper_bounds += [
                dt_0 / self.n_bells / self.n_bells,
                np.inf,
                5.0 * dt_0,
            ]
        bounds = (lower_bounds, upper_bounds)
        # print(f"Bounds: {bounds}")
        return bounds

    def _set_results(self, popt, pcov, centers, model_flux):
        """Store the best-fit parameters and chi-squared value."""
        chi2 = float(np.sum(((self._flux - model_flux) / self._err) ** 2))

        index = self.n_bells
        fit_centers = centers
        if self.fit_centers:
            fit_centers = np.asarray(
                popt[index : index + self.n_bells], dtype=float
            )
            index += self.n_bells

        self.best = {
            "amplitudes": [float(value) for value in popt[: self.n_bells]],
            "width": float(popt[index]),
            "offset": float(popt[index + 1]),
            "centers": [float(center) for center in fit_centers],
            "dt": float(popt[index + 2]),
            "chi2": chi2,
            "covariance": pcov,
        }
        for bell_index, center in enumerate(self.best["centers"], start=1):
            self.best[f"t_{bell_index}"] = float(center)
            self.best[f"amp_{bell_index}"] = float(popt[bell_index - 1])

    def _check_bounds(self, p0, bounds):
        """Check if the initial guess is within the bounds."""
        lower_bounds, upper_bounds = bounds
        for i, (param, low, high) in enumerate(
            zip(p0, lower_bounds, upper_bounds)
        ):
            if not (low <= param <= high):
                raise ValueError(

                    f"Initial guess for anomaly classifier parameter {self._fit_parameters[i]} ({param}) is out of bounds ({low}, {high})."
                )

    def _get_fit_parameters(self):
        """Get the names of the fit parameters based on the number of bells and whether centers are fitted."""
        fit_parameters = [
            f"amplitude_{str(i + 1)}" for i in range(self.n_bells)
        ]
        if self.fit_centers:
            fit_parameters += [f"t_{i + 1}" for i in range(self.n_bells)]
        fit_parameters += ["width", "offset", "dt"]
        return fit_parameters

    def run(self):
        """Run the fit and store the best-fit parameters."""
        time, flux, err = self._stack_pspl_residuals()
        t_pl, dt_0 = self._set_time_parameters(time)
        self.t_pl = t_pl
        self._time, self._flux, self._err = self._fit_window(
            time, flux, err, t_pl, dt_0
        )

        centers, amplitudes, width, baseline = self._set_shape_parameters(
            t_pl, dt_0
        )
        p0 = self._set_initial_guess(
            amplitudes, centers, width, baseline, dt_0
        )
        bounds = self._set_bounds(t_pl, dt_0)
        self._check_bounds(p0, bounds)
        # print(f"Initial guess: {p0}")
        # print(f"Bounds: {bounds}")

        def model(x, *params):
            fit_amplitudes = params[: self.n_bells]
            index = self.n_bells
            fit_centers = centers
            if self.fit_centers:
                fit_centers = params[index : index + self.n_bells]
                index += self.n_bells
            fit_width = params[index]
            fit_offset = params[index + 1]
            fit_dt = params[index + 2]
            return self._template(
                x,
                fit_centers,
                fit_width,
                fit_offset,
                fit_amplitudes,
                fit_dt,
                t_pl,
            )

        popt, pcov = curve_fit(
            model,
            self._time,
            self._flux,
            p0=p0,
            sigma=self._err,
            absolute_sigma=True,
            bounds=bounds,
            maxfev=50000,
        )

        model_flux = model(self._time, *popt)
        self._set_results(popt, pcov, centers, model_flux)
        return self.best

    def plot_fit(
        self,
        best=None,
        fig=None,
        ax_fit=None,
        ax_resid=None,
        show=False,
        t_range=None,
        data_color="k",
        model_color="C1",
        vline_color="C2",
        residual_color="C1",
    ):
        if self.best is None:
            self.run()

        time, flux, err = self._stack_pspl_residuals()
        # Build model_flux by passing the expected keyword arguments to _template
        model_flux = self._template(
            time,
            centers=self.best["centers"],
            width=self.best["width"],
            offset=self.best["offset"],
            amplitudes=self.best["amplitudes"],
            dt=self.best["dt"],
            t_pl=self.t_pl,
        )
        fit_residuals = flux - model_flux

        if ax_fit is None or ax_resid is None:
            fig, (ax_fit, ax_resid) = plt.subplots(
                2,
                1,
                figsize=(8, 6),
                sharex=True,
                gridspec_kw={"height_ratios": [3, 1]},
            )
        elif fig is None:
            fig = ax_fit.figure

        ax_fit.errorbar(
            time,
            flux,
            yerr=err,
            fmt=".",
            color=data_color,
            alpha=0.6,
            label="data",
        )
        ax_fit.plot(
            time,
            model_flux,
            color=model_color,
            lw=2,
            label=f"{self.n_bells}-bell fit",
        )
        for center in self.best["centers"]:
            ax_fit.axvline(center, color=vline_color, ls="--", alpha=0.7)
            ax_fit.axvline(
                center - self.best["width"],
                color=vline_color,
                ls=":",
                alpha=0.5,
            )
            ax_fit.axvline(
                center + self.best["width"],
                color=vline_color,
                ls=":",
                alpha=0.5,
            )
        ax_fit.set_ylabel("Residual flux")
        if t_range is not None:
            ax_fit.set_xlim(t_range)
        ax_fit.legend()

        ax_resid.errorbar(
            time,
            fit_residuals,
            yerr=err,
            fmt=".",
            color=residual_color,
            alpha=0.6,
            label=f"{self.n_bells}-bell residuals",
        )
        ax_resid.axhline(0.0, color="0.3", ls="--", lw=1)
        ax_resid.set_xlabel("Time")
        ax_resid.set_ylabel("Data - fit")
        if t_range is not None:
            ax_resid.set_xlim(t_range)
        ax_resid.legend()

        if show:
            plt.show()

        return fig, ax_fit, ax_resid
