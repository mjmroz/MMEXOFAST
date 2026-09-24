# test_workflow.py

import glob
import os
import pickle
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import mmexofast as mmexo
from mmexofast import MMEXOFASTFitter, WorkflowStep, fit_types
from mmexofast.config import DATA_PATH
from mmexofast.fitters import MulensFitter
from mmexofast.mulens_object_config import EventConfig, ModelConfig
from mmexofast.workflow_step import StepStatus

# OB05390
OB05390_FILES = sorted(
    glob.glob(os.path.join(DATA_PATH, "OB05390", "n200*.txt"))
)

with open(os.path.join(DATA_PATH, "OB05390", "coords.txt")) as f:
    OB05390_COORDS = f.read().strip()

BINARY_FIT_KEY = fit_types.FitKey(
    lens_type=fit_types.LensType.BINARY,
    source_type=fit_types.SourceType.FINITE,
    parallax_branch=fit_types.ParallaxBranch.NONE,
    lens_orb_motion=fit_types.LensOrbMotion.NONE,
    locations_used=None,
)

BINARY_PARAMS = {
    "t_0": 2453582.7281740606,
    "u_0": 0.355227507989543,
    "t_E": 11.106795114521415,
    "rho": 0.024632765186197645,
    "q": 7.524529162733864e-05,
    "s": 1.6044784697939465,
    "alpha": 157.9506556145345,
}

BEST_EF_GRID_POINT = {
    "t_0": 2456836.080383359,
    "t_eff": 23.67696884508345,
    "j": 2,
    "chi2": -137842.8089725696,
}

# OB140939
GROUND_DATA_FILES = [
    os.path.join(DATA_PATH, "OB140939", "n20100310.I.OGLE.OB140939.txt")
]

COORDS = "17:47:12.25 -21:22:58.7"

STATIC_PSPL_KEY = fit_types.FitKey(
    lens_type=fit_types.LensType.POINT,
    source_type=fit_types.SourceType.POINT,
    parallax_branch=fit_types.ParallaxBranch.NONE,
    lens_orb_motion=fit_types.LensOrbMotion.NONE,
    locations_used=None,
)

STATIC_PSPL_PARAMS = {
    "t_0": 2456836.0,
    "u_0": 1.012,
    "t_E": 21.48,
}

INITIAL_RESULTS = {
    "PSPL static": {
        "params": STATIC_PSPL_PARAMS,
    }
}

# ---------------------------------------------------------------------------
# Per-stage step blocks
#
# Each stage's steps are defined once. The combined workflow lists below are
# built by concatenation, so adding or renaming a step in one place
# propagates everywhere automatically.
# ---------------------------------------------------------------------------

_STEPS_EVENT_SEARCH = [
    ("run_event_search", "event_search"),
]
_STEPS_STATIC_PL = [
    ("estimate_point_lens_parameters", "fit_static_point_lens"),
    ("fit_static_point_source_point_lens", "fit_static_point_lens"),
]
_STEPS_PL_PARALLAX = [
    ("fit_parallax_u0+", "fit_point_lens_parallax"),
    ("fit_parallax_u0-", "fit_point_lens_parallax"),
]
_STEPS_RENORM = [
    ("renormalize_datasets", "renormalize"),
    ("refit_all", "renormalize"),
]
_STEPS_SEARCH_ANOMALY = [
    ("compute_point_lens_residuals", "search_for_anomaly"),
    ("run_anomaly_search", "search_for_anomaly"),
    ("get_anomaly_light_curve_parameters", "search_for_anomaly"),
    ("classify_anomaly", "search_for_anomaly"),
]
_STEPS_FIT_BINARY = [
    ("estimate_binary_lens_parameters", "fit_binary_lens"),
    ("fit_binary_lens_models", "fit_binary_lens"),
]
_STEPS_CHECK_BINARY_RENORM = [
    ("check_needs_renorm", "check_binary_renorm"),
]
_STEPS_PARALLAX_GRIDS = [
    ("run_parallax_grids", "parallax_grids"),
]

# Combined workflow lists, each built entirely from the blocks above.
EXPECTED_STEPS = _STEPS_EVENT_SEARCH + _STEPS_STATIC_PL + _STEPS_PL_PARALLAX
EXPECTED_STEPS_PL_RENORM = (
    _STEPS_EVENT_SEARCH
    + _STEPS_STATIC_PL
    + _STEPS_PL_PARALLAX
    + _STEPS_RENORM
    + _STEPS_PARALLAX_GRIDS
)
# Renormalization runs AFTER the anomaly search in the binary workflow, so
# its outlier rejection can protect the anomaly window (anomaly_lc_params
# does not exist earlier; the old renorm-first order flagged the planetary
# anomaly itself as bad data).
EXPECTED_STEPS_BINARY = (
    _STEPS_EVENT_SEARCH
    + _STEPS_STATIC_PL
    + _STEPS_PL_PARALLAX
    + _STEPS_SEARCH_ANOMALY
    + _STEPS_RENORM
    + _STEPS_FIT_BINARY
    + _STEPS_CHECK_BINARY_RENORM
)

# ---------------------------------------------------------------------------
# Name-based slice helper
#
# Replaces magic-index slicing (e.g. EXPECTED_STEPS_BINARY[:11]) with an
# intent-revealing name. A wrong index silently tests the wrong scenario;
# a wrong name raises ValueError immediately.
# ---------------------------------------------------------------------------


def steps_through(steps, step_name):
    """
    Return a prefix of `steps` ending at (and including) the first step
    whose name matches `step_name`.

    Parameters
    ----------
    steps : list of (name, stage) tuples
    step_name : str

    Returns
    -------
    list of (name, stage) tuples

    Raises
    ------
    ValueError
        If step_name is not present in steps.
    """
    for i, (name, _stage) in enumerate(steps):
        if name == step_name:
            return steps[: i + 1]
    raise ValueError(f"{step_name!r} not found in steps list")


# ---------------------------------------------------------------------------
# Step-to-method mapping and shared patching helper
#
# Maps each workflow step name to the MMEXOFASTFitter method it invokes.
# fit_parallax_u0+ and fit_parallax_u0- share one underlying method so one
# patch covers both.
#
# When a new step is added to any EXPECTED_STEPS_* constant, add its entry
# here too. patch_fitter_methods will raise AssertionError immediately if a
# step name is missing, making the omission obvious.
# ---------------------------------------------------------------------------

_STEP_TO_METHOD = {
    "run_event_search": "run_event_search",
    "estimate_point_lens_parameters": "estimate_point_lens_parameters",
    "fit_static_point_source_point_lens": "fit_static_point_source_point_lens",
    "fit_static_finite_source_point_lens": "fit_static_finite_source_point_lens",
    "fit_parallax_u0+": "fit_parallax",  # shared method
    "fit_parallax_u0-": "fit_parallax",  # shared method
    "renormalize_datasets": "renormalize_datasets",
    "refit_all": "refit_all",
    "select_best_point_lens_model": "select_best_point_lens_model",
    "compute_point_lens_residuals": "compute_point_lens_residuals",
    "run_anomaly_search": "run_anomaly_search",
    "get_anomaly_light_curve_parameters": "get_anomaly_light_curve_parameters",
    "classify_anomaly": "classify_anomaly",
    "estimate_binary_lens_parameters": "estimate_binary_lens_parameters",
    "fit_binary_lens_models": "fit_binary_lens_models",
    "check_needs_renorm": "check_needs_renorm",
    "run_parallax_grids": "run_parallax_grids",
}

# Methods whose no-op return value must be something other than None.
_METHOD_RETURN_VALUES = {
    "run_event_search": {},
    "estimate_point_lens_parameters": {},
}


def patch_fitter_methods(test_case, fitter, expected_steps):
    """
    Patch the fitter methods invoked by expected_steps with no-ops.

    Each step name is looked up in _STEP_TO_METHOD to find the underlying
    fitter method. Steps that share a method (fit_parallax_u0+/u0-) produce
    one patch whose mock is accessible under both step-name keys.

    Parameters
    ----------
    test_case : unittest.TestCase
        Used for the coverage guard assertion.
    fitter : MMEXOFASTFitter
        Instance whose methods are patched.
    expected_steps : list of (name, stage) tuples
        Full workflow for this test class; determines which methods are
        patched and what the guard checks against.

    Returns
    -------
    ExitStack
        Active context manager. Attribute .mocks is a dict keyed by step
        name, mapping to the corresponding MagicMock.

    Raises
    ------
    AssertionError
        If any step name in expected_steps has no entry in _STEP_TO_METHOD.
        Add the missing entry to _STEP_TO_METHOD to resolve.
    """
    # Guard first — fail clearly before patching anything.
    # If this fires, a step was added to an EXPECTED_STEPS_* constant
    # without a corresponding entry in _STEP_TO_METHOD.
    unknown = {name for name, _ in expected_steps} - set(_STEP_TO_METHOD)
    test_case.assertFalse(
        unknown,
        f"Steps missing from _STEP_TO_METHOD: {unknown}. "
        f"Add an entry to _STEP_TO_METHOD when adding a new workflow step.",
    )

    stack = ExitStack()
    method_mocks = {}  # keyed by method name; prevents double-patching
    mocks = {}  # keyed by step name for caller inspection

    for step_name, _ in expected_steps:
        method_name = _STEP_TO_METHOD[step_name]
        if method_name not in method_mocks:
            rv = _METHOD_RETURN_VALUES.get(method_name, None)
            method_mocks[method_name] = stack.enter_context(
                patch.object(fitter, method_name, return_value=rv)
            )
        mocks[step_name] = method_mocks[method_name]

    stack.mocks = mocks
    return stack


def _make_noop_steps(expected_steps):
    """
    Build a WorkflowStep list from (name, stage) tuples with no-op actions.
    Used to pre-populate completed_steps in resume tests.
    """
    return [
        WorkflowStep(
            name=name,
            stage=stage,
            func=MagicMock(return_value=None),
            description=f"No-op for {name}",
        )
        for name, stage in expected_steps
    ]


class TestPointLensWorkflow(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = self.tmp_dir.name

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _make_fitter(self, **kwargs):
        defaults = dict(
            files=GROUND_DATA_FILES,
            coords=COORDS,
            fit_type="point_lens",
            renormalize_errors=False,
        )
        defaults.update(kwargs)
        return MMEXOFASTFitter(**defaults)

    def _patch_fit_methods(self, fitter):
        return patch_fitter_methods(self, fitter, EXPECTED_STEPS)

    # --- dry run ---

    def test_dry_run_planned_steps(self):
        """
        Dry run for ground-only point-lens fit with renormalize_errors=False
        produces the expected step queue with correct names and stages.
        """
        fitter = self._make_fitter(dry_run=True)
        fitter.fit()

        actual = [(step.name, step.stage) for step in fitter.planned_steps]
        self.assertEqual(actual, EXPECTED_STEPS)

    def test_finite_source_point_lens_adds_fspl_step(self):
        """
        Enabling finite_source_point_lens adds the static FSPL step to the
        point-lens workflow.
        """
        fitter = self._make_fitter(dry_run=True, finite_source_point_lens=True)
        fitter.fit()

        actual = [(step.name, step.stage) for step in fitter.planned_steps]
        self.assertIn(
            ("fit_static_finite_source_point_lens", "fit_static_point_lens"),
            actual,
        )

    def test_source_type_controls_fspl_rho_seed(self):
        """
        source_type is forwarded into the FSPL rho estimator when the
        finite-source fit is built.
        """
        fitter = self._make_fitter(
            finite_source_point_lens=True,
            source_type="dwarf",
        )

        pspl_key = fit_types.FitKey(
            lens_type=fit_types.LensType.POINT,
            source_type=fit_types.SourceType.POINT,
            parallax_branch=fit_types.ParallaxBranch.NONE,
            lens_orb_motion=fit_types.LensOrbMotion.NONE,
        )
        fitter.all_fit_results.set(
            mmexo.FitRecord(
                model_key=pspl_key,
                params=dict(STATIC_PSPL_PARAMS),
            )
        )

        captured = {}

        def fake_get_rho(self):
            captured["limit"] = self.limit
            return 0.001 if self.limit == "dwarf" else 0.05

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    fitter, "_check_FSPL_condition", return_value=True
                )
            )
            rho_patch = stack.enter_context(
                patch(
                    "mmexofast.estimate_params.ParameterEstimator.get_rho",
                    fake_get_rho,
                )
            )
            fitter_patch = stack.enter_context(
                patch("mmexofast.mmexofast.SFitFitter")
            )
            fitter.fit_static_finite_source_point_lens()

        self.assertEqual(captured["limit"], "dwarf")
        self.assertEqual(
            fitter_patch.call_args.kwargs["initial_model_params"]["rho"],
            0.001,
        )

    def test_unmet_fspl_condition_falls_back_to_point_source_parallax(self):
        """
        When the FSPL condition is not met, later parallax fitting should use
        point-source keys instead of leaving the workflow in an FSPL state.
        """
        fitter = self._make_fitter(finite_source_point_lens="u_0<0.1")

        pspl_key = fit_types.FitKey(
            lens_type=fit_types.LensType.POINT,
            source_type=fit_types.SourceType.POINT,
            parallax_branch=fit_types.ParallaxBranch.NONE,
            lens_orb_motion=fit_types.LensOrbMotion.NONE,
        )
        fitter.all_fit_results.set(
            mmexo.FitRecord(
                model_key=pspl_key,
                params={**STATIC_PSPL_PARAMS, "u_0": 0.2},
            )
        )

        fitter.fit_static_finite_source_point_lens(
            initial_params={**STATIC_PSPL_PARAMS, "u_0": 0.2}
        )

        self.assertFalse(fitter.finite_source_point_lens)

        with patch.object(
            fitter, "_do_parallax_fit", return_value=None
        ) as mock:
            fitter.fit_parallax(branch=fit_types.ParallaxBranch.U0_PLUS)

        self.assertEqual(
            mock.call_args.kwargs["source_type"],
            fit_types.SourceType.POINT,
        )

    def test_binary_lens_parallax_uses_finite_source_keys(self):
        """
        Binary-lens parallax fits should keep using finite-source keys while
        the point-lens finite-source branch remains enabled.
        """
        fitter = self._make_fitter(
            finite_source_point_lens=True,
            parallax_binary_lens=True,
        )

        fspl_key = fit_types.FitKey(
            lens_type=fit_types.LensType.POINT,
            source_type=fit_types.SourceType.FINITE,
            parallax_branch=fit_types.ParallaxBranch.NONE,
            lens_orb_motion=fit_types.LensOrbMotion.NONE,
        )
        fitter.all_fit_results.set(
            mmexo.FitRecord(
                model_key=fspl_key,
                params={**STATIC_PSPL_PARAMS, "u_0": 0.2, "rho": 0.01},
            )
        )

        with patch.object(
            fitter, "_do_parallax_fit", return_value=None
        ) as mock:
            fitter.fit_parallax(branch=fit_types.ParallaxBranch.U0_PLUS)

        self.assertEqual(
            mock.call_args.kwargs["source_type"],
            fit_types.SourceType.FINITE,
        )

    # --- stop_before stage:step ---

    def test_stop_before_first_step_of_stage(self):
        """
        stop_before='fit_static_point_lens:estimate_point_lens_parameters' executes only
        steps before estimate_point_lens_parameters (i.e. the event_search stage only).
        """
        fitter = self._make_fitter(
            stop_before="fit_static_point_lens:estimate_point_lens_parameters"
        )

        with self._patch_fit_methods(fitter):
            fitter.fit()

        expected = _STEPS_EVENT_SEARCH
        actual = [(step.name, step.stage) for step in fitter.completed_steps]
        self.assertEqual(actual, expected)

    def test_stop_before_second_step_of_stage(self):
        """
        stop_before='fit_static_point_lens:fit_static_point_source_point_lens' executes steps through
        estimate_point_lens_parameters but not fit_static_point_source_point_lens.
        """
        fitter = self._make_fitter(
            stop_before="fit_static_point_lens:fit_static_point_source_point_lens"
        )

        with self._patch_fit_methods(fitter):
            fitter.fit()

        expected = steps_through(
            EXPECTED_STEPS, "estimate_point_lens_parameters"
        )
        actual = [(step.name, step.stage) for step in fitter.completed_steps]
        self.assertEqual(actual, expected)

    # --- stop_after stage:step ---

    def test_stop_after_first_step_of_stage(self):
        """
        stop_after='fit_static_point_lens:estimate_point_lens_parameters' executes steps
        through and including estimate_point_lens_parameters.
        """
        fitter = self._make_fitter(
            stop_after="fit_static_point_lens:estimate_point_lens_parameters"
        )

        with self._patch_fit_methods(fitter):
            fitter.fit()

        expected = steps_through(
            EXPECTED_STEPS, "estimate_point_lens_parameters"
        )
        actual = [(step.name, step.stage) for step in fitter.completed_steps]
        self.assertEqual(actual, expected)

    def test_stop_after_second_step_of_stage(self):
        """
        stop_after='fit_static_point_lens:fit_static_point_source_point_lens' executes steps through
        and including fit_static_point_source_point_lens.
        """
        fitter = self._make_fitter(
            stop_after="fit_static_point_lens:fit_static_point_source_point_lens"
        )

        with self._patch_fit_methods(fitter):
            fitter.fit()

        expected = steps_through(
            EXPECTED_STEPS, "fit_static_point_source_point_lens"
        )
        actual = [(step.name, step.stage) for step in fitter.completed_steps]
        self.assertEqual(actual, expected)

    # --- stop_before stage-only ---

    def test_stop_before_stage_halts_before_first_step(self):
        """
        stop_before='fit_static_point_lens' halts before the first step
        of that stage.
        """
        fitter = self._make_fitter(stop_before="fit_static_point_lens")

        with self._patch_fit_methods(fitter):
            fitter.fit()

        expected = _STEPS_EVENT_SEARCH
        actual = [(step.name, step.stage) for step in fitter.completed_steps]
        self.assertEqual(actual, expected)

    # --- stop_after stage-only ---

    def test_stop_after_stage_halts_after_last_step(self):
        """
        stop_after='fit_static_point_lens' halts after the last step
        of that stage (i.e. after fit_static_point_source_point_lens).
        """
        fitter = self._make_fitter(stop_after="fit_static_point_lens")

        with self._patch_fit_methods(fitter):
            fitter.fit()

        expected = steps_through(
            EXPECTED_STEPS, "fit_static_point_source_point_lens"
        )
        actual = [(step.name, step.stage) for step in fitter.completed_steps]
        self.assertEqual(actual, expected)

    # --- resume after stop ---

    def test_resume_after_stop_before_planned_steps(self):
        """
        When completed_steps contains only run_event_search, planned_steps
        starts at estimate_point_lens_parameters.
        """
        fitter = self._make_fitter(dry_run=True)
        fitter.completed_steps = _make_noop_steps(
            steps_through(EXPECTED_STEPS, "run_event_search")
        )
        fitter.fit()

        expected = _STEPS_STATIC_PL + _STEPS_PL_PARALLAX
        actual = [(step.name, step.stage) for step in fitter.planned_steps]
        self.assertEqual(actual, expected)

    def test_resume_after_stop_after_planned_steps(self):
        """
        When completed_steps contains run_event_search and estimate_point_lens_parameters,
        planned_steps starts at fit_static_point_source_point_lens.
        """
        fitter = self._make_fitter(dry_run=True)
        fitter.completed_steps = _make_noop_steps(
            steps_through(EXPECTED_STEPS, "estimate_point_lens_parameters")
        )
        fitter.fit()

        expected = _STEPS_STATIC_PL[1:] + _STEPS_PL_PARALLAX
        actual = [(step.name, step.stage) for step in fitter.planned_steps]
        self.assertEqual(actual, expected)


class TestPointLensRenormWorkflow(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = self.tmp_dir.name

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _make_fitter(self, **kwargs):
        defaults = dict(
            files=GROUND_DATA_FILES,
            coords=COORDS,
            fit_type="point_lens",
            renormalize_errors=True,
            parallax_point_lens='grid'
        )
        defaults.update(kwargs)
        return MMEXOFASTFitter(**defaults)

    def _patch_fit_methods(self, fitter):
        return patch_fitter_methods(self, fitter, EXPECTED_STEPS_PL_RENORM)

    def test_dry_run_planned_steps(self):
        """
        Dry run for ground-only point-lens fit with renormalize_errors=True
        and parallax_point_lens='grid' produces the expected step queue.
        """
        fitter = self._make_fitter(dry_run=True)
        fitter.fit()

        actual = [(step.name, step.stage) for step in fitter.planned_steps]
        self.assertEqual(actual, EXPECTED_STEPS_PL_RENORM)

    def test_stop_before_renormalize_stage(self):
        """
        stop_before='renormalize' halts before renormalize_datasets.
        """
        fitter = self._make_fitter(stop_before="renormalize")

        with self._patch_fit_methods(fitter):
            fitter.fit()

        expected = steps_through(EXPECTED_STEPS_PL_RENORM, "fit_parallax_u0-")
        actual = [(step.name, step.stage) for step in fitter.completed_steps]
        self.assertEqual(actual, expected)

    def test_stop_after_renormalize_stage(self):
        """
        stop_after='renormalize' halts after refit_all.
        """
        fitter = self._make_fitter(stop_after="renormalize")

        with self._patch_fit_methods(fitter):
            fitter.fit()

        expected = steps_through(EXPECTED_STEPS_PL_RENORM, "refit_all")
        actual = [(step.name, step.stage) for step in fitter.completed_steps]
        self.assertEqual(actual, expected)

    def test_resume_after_stop_before_renormalize(self):
        """
        When completed_steps ends at fit_point_lens_parallax,
        planned_steps starts at renormalize_datasets.
        """
        fitter = self._make_fitter(dry_run=True)
        fitter.completed_steps = _make_noop_steps(
            steps_through(EXPECTED_STEPS_PL_RENORM, "fit_parallax_u0-")
        )
        fitter.fit()

        expected = _STEPS_RENORM + _STEPS_PARALLAX_GRIDS
        actual = [(step.name, step.stage) for step in fitter.planned_steps]
        self.assertEqual(actual, expected)


class TestBinaryLensWorkflow(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = self.tmp_dir.name

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _make_fitter(self, **kwargs):
        defaults = dict(
            files=GROUND_DATA_FILES,
            coords=COORDS,
            fit_type="binary_lens",
            renormalize_errors=True,
            parallax_point_lens='grid',
        )
        defaults.update(kwargs)
        return MMEXOFASTFitter(**defaults)

    def _patch_fit_methods(self, fitter):
        return patch_fitter_methods(self, fitter, EXPECTED_STEPS_BINARY)

    def test_dry_run_planned_steps(self):
        """
        Dry run for ground-only binary lens fit with renormalize_errors=True
        and point-lens parallax branches enabled produces the expected step queue.
        """
        fitter = self._make_fitter(dry_run=True)
        fitter.fit()

        actual = [(step.name, step.stage) for step in fitter.planned_steps]
        self.assertEqual(actual, EXPECTED_STEPS_BINARY)

    def test_stop_before_search_for_anomaly(self):
        """
        stop_before='search_for_anomaly' halts before
        select_best_point_lens_model.
        """
        fitter = self._make_fitter(stop_before="search_for_anomaly")

        with self._patch_fit_methods(fitter):
            fitter.fit()

        # Renormalization now runs after the anomaly search, so stopping
        # before the search means the parallax fits are the last completed.
        expected = steps_through(EXPECTED_STEPS_BINARY, "fit_parallax_u0-")
        actual = [(step.name, step.stage) for step in fitter.completed_steps]
        self.assertEqual(actual, expected)

    def test_stop_after_fit_binary(self):
        """
        stop_after='fit_binary_lens' halts after fit_binary_lens_models.
        """
        fitter = self._make_fitter(stop_after="fit_binary_lens")

        with self._patch_fit_methods(fitter):
            fitter.fit()

        expected = steps_through(
            EXPECTED_STEPS_BINARY, "fit_binary_lens_models"
        )
        actual = [(step.name, step.stage) for step in fitter.completed_steps]
        self.assertEqual(actual, expected)

    def test_resume_after_stop_before_fit_binary(self):
        """
        When completed_steps ends at estimate_binary_lens_parameters,
        planned_steps starts at fit_binary_lens_models.
        """
        fitter = self._make_fitter(dry_run=True)
        fitter.completed_steps = _make_noop_steps(
            steps_through(
                EXPECTED_STEPS_BINARY, "estimate_binary_lens_parameters"
            )
        )
        fitter.fit()

        expected = (
            _STEPS_FIT_BINARY[1:]
            + _STEPS_CHECK_BINARY_RENORM
        )
        actual = [(step.name, step.stage) for step in fitter.planned_steps]
        self.assertEqual(actual, expected)

    def test_resume_after_stop_after_check_binary_renorm(self):
        """
        When completed_steps ends at check_binary_renorm,
        planned_steps contains only the parallax grid steps.
        """
        fitter = self._make_fitter(dry_run=True)
        fitter.completed_steps = _make_noop_steps(
            steps_through(EXPECTED_STEPS_BINARY, "check_needs_renorm")
        )
        fitter.fit()

        expected = []
        actual = [(step.name, step.stage) for step in fitter.planned_steps]
        self.assertEqual(actual, expected)

    def test_check_needs_renorm_inserts_steps_when_true(self):
        """
        When check_needs_renorm returns True, renormalize_datasets and
        refit_all are inserted and executed as dynamic post-binary steps.
        """
        fitter = self._make_fitter()
        fitter.completed_steps = _make_noop_steps(
            steps_through(EXPECTED_STEPS_BINARY, "fit_binary_lens_models")
        )

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    fitter, "_needs_renormalization", return_value=True
                )
            )
            stack.enter_context(
                patch.object(fitter, "renormalize_datasets", return_value=None)
            )
            stack.enter_context(
                patch.object(fitter, "refit_all", return_value=None)
            )
            fitter.fit()

        actual_names = [step.name for step in fitter.completed_steps]
        self.assertIn("renormalize_datasets", actual_names)
        self.assertIn("refit_all", actual_names)
        self.assertLess(
            actual_names.index("renormalize_datasets"),
            actual_names.index("refit_all"),
        )


class TestPointLensWorkflowWithInitialResults(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = self.tmp_dir.name

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _make_fitter(self, **kwargs):
        defaults = dict(
            files=GROUND_DATA_FILES,
            coords=COORDS,
            fit_type="point_lens",
            renormalize_errors=False,
            initial_results=INITIAL_RESULTS,
        )
        defaults.update(kwargs)
        return MMEXOFASTFitter(**defaults)

    def _patch_fit_methods(self, fitter):
        # Patch the full point-lens workflow; the initial_results causes
        # some steps to be skipped at runtime, but patching unused methods
        # is harmless and keeps the guard meaningful.
        return patch_fitter_methods(self, fitter, EXPECTED_STEPS)

    def test_dry_run_skips_estimate_point_lens_parameters(self):
        """
        When a static PSPL is supplied via initial_results with
        fit_type='point_lens', planned_steps starts at fit_static_point_source_point_lens,
        skipping estimate_point_lens_parameters.
        """
        fitter = self._make_fitter(dry_run=True)
        fitter.fit()

        expected = _STEPS_STATIC_PL[1:] + _STEPS_PL_PARALLAX
        actual = [(step.name, step.stage) for step in fitter.planned_steps]
        self.assertEqual(actual, expected)

    def test_fit_static_point_source_point_lens_uses_supplied_params_as_seed(
        self,
    ):
        """
        fit_static_point_source_point_lens is called with the user-supplied PSPL params as seed.
        """
        fitter = self._make_fitter()

        with self._patch_fit_methods(fitter) as stack:
            fitter.fit()

        call_args = stack.mocks["fit_static_point_source_point_lens"].call_args
        self.assertEqual(
            call_args.kwargs.get("initial_params"), STATIC_PSPL_PARAMS
        )

    def test_initial_results_and_restart_from_raises(self):
        """
        Providing both initial_results and restart_from raises ValueError.
        """
        with self.assertRaises(ValueError):
            self._make_fitter(restart_from="event_search")


class TestBinaryLensWorkflowWithInitialResults(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = self.tmp_dir.name

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _make_fitter(self, **kwargs):
        defaults = dict(
            files=GROUND_DATA_FILES,
            coords=COORDS,
            fit_type="binary_lens",
            renormalize_errors=False,
            initial_results=INITIAL_RESULTS,
        )
        defaults.update(kwargs)
        return MMEXOFASTFitter(**defaults)

    def _patch_fit_methods(self, fitter):
        return patch_fitter_methods(
            self, fitter, _STEPS_SEARCH_ANOMALY + _STEPS_FIT_BINARY
        )

    def test_dry_run_starts_at_search_for_anomaly(self):
        """
        When a static PSPL is supplied via initial_results with
        fit_type='binary_lens', planned_steps starts at search_for_anomaly,
        skipping all point-lens stages.
        """
        fitter = self._make_fitter(dry_run=True)
        fitter.fit()

        expected = _STEPS_SEARCH_ANOMALY + _STEPS_FIT_BINARY
        actual = [(step.name, step.stage) for step in fitter.planned_steps]
        self.assertEqual(actual, expected)

    def test_select_best_point_lens_model_returns_supplied_pspl(self):
        """
        select_best_point_lens_model returns the user-supplied PSPL record
        when initial_results contains a PSPL fit.
        """
        fitter = self._make_fitter()
        result = fitter.select_best_point_lens_model()
        self.assertEqual(result.params, STATIC_PSPL_PARAMS)

    def test_initial_results_and_restart_from_raises(self):
        """
        Providing both initial_results and restart_from raises ValueError.
        """
        with self.assertRaises(ValueError):
            self._make_fitter(restart_from="event_search")


def _make_fake_pickle(path, completed_steps):
    """
    Write a minimal fake restart pickle containing only completed_steps.
    """
    state = {
        "completed_steps": [(s.name, s.stage) for s in completed_steps],
    }
    data = {"config": {}, "state": state}
    with open(path, "wb") as f:
        pickle.dump(data, f)


class TestBinaryLensRestartFromPointLens(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = self.tmp_dir.name

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _make_fitter(self, restart_file, **kwargs):
        defaults = dict(
            files=GROUND_DATA_FILES,
            coords=COORDS,
            fit_type="binary_lens",
            renormalize_errors=True,
        )
        defaults.update(kwargs)
        return MMEXOFASTFitter(restart_file=restart_file, **defaults)

    def test_binary_steps_added_after_point_lens_restart(self):
        """
        Restarting from a completed point-lens run (renormalize_errors=True,
        parallax_grid=False) with fit_type='binary_lens' produces a step
        queue that starts at search_for_anomaly.
        """
        # A completed POINT-LENS run: its workflow keeps renormalize right
        # after the parallax fits (only the binary workflow defers it past
        # the anomaly search).
        pl_completed = _make_noop_steps(
            steps_through(EXPECTED_STEPS_PL_RENORM, "refit_all")
        )

        pkl_path = os.path.join(self.tmp_path, "fake.pkl")
        _make_fake_pickle(pkl_path, pl_completed)

        fitter = self._make_fitter(restart_file=pkl_path, dry_run=True)
        fitter.fit()

        expected = (
            _STEPS_SEARCH_ANOMALY
            + _STEPS_FIT_BINARY
            + _STEPS_CHECK_BINARY_RENORM
        )
        actual = [(step.name, step.stage) for step in fitter.planned_steps]
        self.assertEqual(actual, expected)


class TestExecutionLoopDynamicSteps(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = self.tmp_dir.name

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_action_returning_none_does_not_insert_steps(self):
        """
        estimate_point_lens_parameters runs for real, returns None from the step action,
        and the execution loop continues normally without inserting steps.
        The result is stored in intermediate_results.estimate_point_lens_parameters.
        """
        fitter = MMEXOFASTFitter(
            files=GROUND_DATA_FILES,
            coords=COORDS,
            fit_type="point_lens",
            renormalize_errors=False,
            stop_after="fit_static_point_lens:estimate_point_lens_parameters",
        )

        fitter.completed_steps = _make_noop_steps(_STEPS_EVENT_SEARCH)
        fitter.intermediate_results.best_ef_grid_point = BEST_EF_GRID_POINT

        fitter.fit()

        self.assertIsNotNone(
            fitter.intermediate_results.estimate_point_lens_parameters
        )
        self.assertEqual(
            fitter.completed_steps[-1].name, "estimate_point_lens_parameters"
        )

    def test_action_returning_steps_inserts_at_front_of_queue(self):
        """
        check_needs_renorm runs for real with a binary FitRecord present.
        The dynamic renorm steps are inserted at the front of the queue and
        visible in planned_steps before they execute.

        _needs_renormalization is patched because re-renormalization after
        the binary fit is currently disabled in the fitter (it returns False
        and logs a warning; see the TODO in _needs_renormalization). The
        subject here is the execution loop's handling of an action that
        returns steps, not the renormalization criterion itself.
        """
        fitter = MMEXOFASTFitter(
            files=OB05390_FILES,
            coords=OB05390_COORDS,
            fit_type="binary_lens",
            renormalize_errors=True,
            parallax_point_lens='grid',
            stop_after="check_binary_renorm:check_needs_renorm",
        )

        fitter.completed_steps = _make_noop_steps(
            steps_through(EXPECTED_STEPS_BINARY, "fit_binary_lens_models")
        )

        binary_fitter = MulensFitter(
            datasets=fitter.datasets,
            initial_model_params=BINARY_PARAMS,
            mag_methods=[2453591.0, "VBBL", 2453594.0],
            model_config=ModelConfig(coords=OB05390_COORDS),
            event_config=EventConfig(coords=OB05390_COORDS),
        )
        binary_fitter.best = binary_fitter.initial_model_params
        binary_fitter.best["chi2"] = 562.0

        binary_record = mmexo.FitRecord.from_full_result(
            model_key=BINARY_FIT_KEY,
            full_result=mmexo.MMEXOFASTFitResults(binary_fitter),
        )

        fitter.all_fit_results.set(binary_record)
        with patch.object(fitter, "_needs_renormalization", return_value=True):
            fitter.fit()

        completed_names = [
            (step.name, step.stage) for step in fitter.completed_steps
        ]
        self.assertIn(
            ("check_needs_renorm", "check_binary_renorm"), completed_names
        )

        planned_names = [
            (step.name, step.stage) for step in fitter.planned_steps
        ]
        self.assertIn(
            ("renormalize_datasets", "check_binary_renorm"), planned_names
        )
        self.assertIn(("refit_all", "check_binary_renorm"), planned_names)

        names_only = [name for name, _ in planned_names]
        self.assertLess(
            names_only.index("renormalize_datasets"),
            names_only.index("refit_all"),
        )


class TestSelectBestPointLensModel(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = self.tmp_dir.name

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _make_fitter(self, **kwargs):
        defaults = dict(
            files=GROUND_DATA_FILES,
            coords=COORDS,
            fit_type="point lens",
            renormalize_errors=False,
        )
        defaults.update(kwargs)
        return MMEXOFASTFitter(**defaults)

    def _make_static_key(self, locations_used=None):
        return fit_types.FitKey(
            lens_type=fit_types.LensType.POINT,
            source_type=fit_types.SourceType.POINT,
            parallax_branch=fit_types.ParallaxBranch.NONE,
            lens_orb_motion=fit_types.LensOrbMotion.NONE,
            locations_used=locations_used,
        )

    def _make_parallax_key(self, branch=fit_types.ParallaxBranch.U0_PLUS):
        return fit_types.FitKey(
            lens_type=fit_types.LensType.POINT,
            source_type=fit_types.SourceType.POINT,
            parallax_branch=branch,
            lens_orb_motion=fit_types.LensOrbMotion.NONE,
        )

    def _make_record(self, key, chi2_value=None):
        record = mmexo.FitRecord(
            model_key=key,
            params=STATIC_PSPL_PARAMS,
            is_complete=(chi2_value is not None),
        )
        if chi2_value is not None:
            record.chi2 = lambda: chi2_value
        return record

    def test_raises_when_no_point_lens_fits(self):
        fitter = self._make_fitter()
        with self.assertRaises(RuntimeError):
            fitter.select_best_point_lens_model()

    def test_multiple_incomplete_records_raises(self):
        fitter = self._make_fitter()
        fitter.all_fit_results.set(self._make_record(self._make_static_key()))
        fitter.all_fit_results.set(
            self._make_record(self._make_parallax_key())
        )
        with self.assertRaises(RuntimeError):
            fitter.select_best_point_lens_model()

    def test_static_fits_only_returns_best_chi2(self):
        fitter = self._make_fitter()
        better = self._make_record(
            self._make_static_key(locations_used="a"), chi2_value=100.0
        )
        worse = self._make_record(
            self._make_static_key(locations_used="b"), chi2_value=200.0
        )
        fitter.all_fit_results.set(better)
        fitter.all_fit_results.set(worse)
        self.assertIs(fitter.select_best_point_lens_model(), better)

    def test_parallax_fits_only_returns_best_chi2(self):
        fitter = self._make_fitter()
        better = self._make_record(
            self._make_parallax_key(fit_types.ParallaxBranch.U0_PLUS),
            chi2_value=80.0,
        )
        worse = self._make_record(
            self._make_parallax_key(fit_types.ParallaxBranch.U0_MINUS),
            chi2_value=120.0,
        )
        fitter.all_fit_results.set(better)
        fitter.all_fit_results.set(worse)
        self.assertIs(fitter.select_best_point_lens_model(), better)

    def test_returns_static_when_parallax_improvement_below_threshold(self):
        fitter = self._make_fitter()
        static = self._make_record(self._make_static_key(), chi2_value=1000.0)
        parallax = self._make_record(
            self._make_parallax_key(), chi2_value=960.0
        )  # improvement = 40
        fitter.all_fit_results.set(static)
        fitter.all_fit_results.set(parallax)
        self.assertIs(fitter.select_best_point_lens_model(), static)

    def test_returns_parallax_when_improvement_above_threshold(self):
        fitter = self._make_fitter()
        static = self._make_record(self._make_static_key(), chi2_value=1000.0)
        parallax = self._make_record(
            self._make_parallax_key(), chi2_value=900.0
        )  # improvement = 100
        fitter.all_fit_results.set(static)
        fitter.all_fit_results.set(parallax)
        self.assertIs(fitter.select_best_point_lens_model(), parallax)

    def test_incomplete_records_ignored_when_complete_records_exist(self):
        fitter = self._make_fitter()
        complete_static = self._make_record(
            self._make_static_key(), chi2_value=1000.0
        )
        incomplete_parallax = self._make_record(
            self._make_parallax_key(), chi2_value=None
        )
        fitter.all_fit_results.set(complete_static)
        fitter.all_fit_results.set(incomplete_parallax)
        self.assertIs(fitter.select_best_point_lens_model(), complete_static)


class TestRestartFromPickleWithStopConditions(unittest.TestCase):
    """
    Covers two restart-from-pickle scenarios:

    1. stop_before / stop_after references a step that already appears in
       the pickle's completed_steps → planned_steps must be empty (nothing
       to re-run).

    2. The previous run halted after 'fit_binary_lens:estimate_binary_lens_parameters';
       the resumed run must plan fit_binary_lens_models as its first step.
    """

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = self.tmp_dir.name

    def tearDown(self):
        self.tmp_dir.cleanup()

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    def _pkl_path(self, name="state.pkl"):
        return os.path.join(self.tmp_path, name)

    def _make_point_lens_fitter(self, restart_file, **kwargs):
        defaults = dict(
            files=GROUND_DATA_FILES,
            coords=COORDS,
            fit_type="point_lens",
            renormalize_errors=False,
        )
        defaults.update(kwargs)
        return MMEXOFASTFitter(restart_file=restart_file, **defaults)

    def _make_binary_lens_fitter(self, restart_file, **kwargs):
        defaults = dict(
            files=GROUND_DATA_FILES,
            coords=COORDS,
            fit_type="binary_lens",
            renormalize_errors=True,
            parallax_point_lens='grid',
        )
        defaults.update(kwargs)
        return MMEXOFASTFitter(restart_file=restart_file, **defaults)

    # ------------------------------------------------------------------
    # Scenario 1a – stop_after step is already in completed_steps
    # ------------------------------------------------------------------

    def test_stop_after_already_completed_yields_empty_plan(self):
        """
        When the restart pickle's completed_steps already contain the step
        named in stop_after, planned_steps is empty: the target has already
        been reached and there is nothing left to run.
        """
        # Pickle records estimate_point_lens_parameters as done; stop_after points to the
        # same step → no remaining work within the stop window.
        completed = _make_noop_steps(
            steps_through(EXPECTED_STEPS, "estimate_point_lens_parameters")
        )
        pkl = self._pkl_path()
        _make_fake_pickle(pkl, completed)

        fitter = self._make_point_lens_fitter(
            restart_file=pkl,
            stop_after="fit_static_point_lens:estimate_point_lens_parameters",
            dry_run=True,
        )
        fitter.fit()

        actual = [(step.name, step.stage) for step in fitter.planned_steps]
        self.assertEqual(actual, [])

    # ------------------------------------------------------------------
    # Scenario 1b – stop_before step is already in completed_steps
    # ------------------------------------------------------------------

    def test_stop_before_already_completed_yields_empty_plan(self):
        """
        When the restart pickle's completed_steps already contain the step
        named in stop_before, the workflow has gone past the intended
        stopping point; planned_steps is empty.
        """
        # Pickle records fit_static_point_source_point_lens as done;
        # stop_before points to fit_static_point_source_point_lens
        # → all steps admissible under the stop_before constraint are
        # already completed.
        completed = _make_noop_steps(
            steps_through(EXPECTED_STEPS, "fit_static_point_source_point_lens")
        )
        pkl = self._pkl_path()
        _make_fake_pickle(pkl, completed)

        fitter = self._make_point_lens_fitter(
            restart_file=pkl,
            stop_before="fit_static_point_lens:fit_static_point_source_point_lens",
            dry_run=True,
        )
        fitter.fit()

        actual = [(step.name, step.stage) for step in fitter.planned_steps]
        self.assertEqual(actual, [])

    # ------------------------------------------------------------------
    # Scenario 2 – resume after estimate_binary_lens_parameters
    # ------------------------------------------------------------------

    def test_restart_after_estimate_binary_lens_parameters_plans_fit_binary_lens_models(
        self,
    ):
        """
        Restarting from a pickle where the previous binary-lens run halted
        after 'fit_binary_lens:estimate_binary_lens_parameters' produces a plan whose
        first step is fit_binary_lens_models, followed by check_binary_renorm
        and parallax_grids.
        """
        completed = _make_noop_steps(
            steps_through(
                EXPECTED_STEPS_BINARY, "estimate_binary_lens_parameters"
            )
        )
        pkl = self._pkl_path()
        _make_fake_pickle(pkl, completed)

        fitter = self._make_binary_lens_fitter(restart_file=pkl, dry_run=True)
        fitter.fit()

        expected = (
            _STEPS_FIT_BINARY[1:]  # fit_binary_lens_models only
            + _STEPS_CHECK_BINARY_RENORM
        )
        actual = [(step.name, step.stage) for step in fitter.planned_steps]
        self.assertEqual(actual, expected)


def _make_fake_pickle_with_stop_conditions(
    path, completed_steps, stop_before=None, stop_after=None, dry_run=False
):
    """
    Like _make_fake_pickle, but also embeds stop_before, stop_after, and
    dry_run in the saved config section. Used to verify that the loader
    ignores invocation directives stored in a previous run's pickle.
    """
    state = {
        "completed_steps": [(s.name, s.stage) for s in completed_steps],
    }
    config = {
        "stop_before": stop_before,
        "stop_after": stop_after,
        "dry_run": dry_run,
    }
    data = {"config": config, "state": state}
    with open(path, "wb") as f:
        pickle.dump(data, f)


class TestRestartIgnoresPickledStopConditions(unittest.TestCase):
    """
    If a previous run saved stop_before, stop_after, or dry_run into its
    pickle's config section, a new run that loads that pickle without
    specifying those arguments must not inherit them. The resumed run
    should continue past the old stopping point, and dry_run must not
    silently suppress execution.
    """

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = self.tmp_dir.name

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _pkl_path(self, name="state.pkl"):
        return os.path.join(self.tmp_path, name)

    def _make_fitter(self, restart_file, **kwargs):
        defaults = dict(
            files=GROUND_DATA_FILES,
            coords=COORDS,
            fit_type="point_lens",
            renormalize_errors=False,
            dry_run=True,
        )  # dry_run=True on the *new* fitter so we
        defaults.update(kwargs)  # can inspect planned_steps safely
        return MMEXOFASTFitter(restart_file=restart_file, **defaults)

    def test_pickled_stop_before_is_not_restored(self):
        """
        A pickle whose config contains stop_before='fit_static_point_lens:fit_static_point_source_point_lens'
        must not re-impose that stop when the new fitter is constructed
        without an explicit stop_before. fit_static_point_source_point_lens and later steps must appear
        in planned_steps.

        completed_steps ends at run_event_search — well before the stop point.
        If stop_before were incorrectly restored it would truncate the plan
        to [estimate_point_lens_parameters] only (everything before fit_static_point_source_point_lens that remains).
        Asserting fit_static_point_source_point_lens is present is therefore a genuine discriminator
        between bug-present and bug-absent.

        The previous version ended completed_steps at estimate_point_lens_parameters
        (immediately before fit_static_point_source_point_lens). Both the buggy and correct
        implementations then plan fit_static_point_source_point_lens as the next step — by coincidence
        in the buggy case — so the test could not catch the bug.
        """
        # End at run_event_search so there is a real gap between completed work
        # and the stop_before cutoff.
        completed = _make_noop_steps(
            steps_through(EXPECTED_STEPS, "run_event_search")
        )
        pkl = self._pkl_path()
        _make_fake_pickle_with_stop_conditions(
            pkl,
            completed,
            stop_before="fit_static_point_lens:estimate_point_lens_parameters",
        )  # ← bug bait

        fitter = self._make_fitter(restart_file=pkl)  # no stop_before
        fitter.fit()

        actual = [(step.name, step.stage) for step in fitter.planned_steps]
        # fit_static_point_source_point_lens must appear; if stop_before were restored it would be absent
        self.assertIn(
            ("fit_static_point_source_point_lens", "fit_static_point_lens"),
            actual,
        )

    def test_pickled_stop_after_is_not_restored(self):
        """
        A pickle whose config contains stop_after='event_search:run_event_search'
        must not re-impose that stop when the new fitter is constructed
        without an explicit stop_after. estimate_point_lens_parameters and later steps must
        appear in planned_steps.
        """
        completed = _make_noop_steps(
            steps_through(EXPECTED_STEPS, "run_event_search")
        )
        pkl = self._pkl_path()
        _make_fake_pickle_with_stop_conditions(
            pkl, completed, stop_after="event_search:run_event_search"
        )  # ← bug bait

        fitter = self._make_fitter(restart_file=pkl)  # no stop_after
        fitter.fit()

        actual = [(step.name, step.stage) for step in fitter.planned_steps]
        self.assertIn(
            ("estimate_point_lens_parameters", "fit_static_point_lens"), actual
        )

    def test_pickled_dry_run_is_not_restored(self):
        """
        A pickle whose config contains dry_run=True must not suppress
        execution when the new fitter is constructed with dry_run=False.
        completed_steps must be non-empty after fit() returns, proving
        that the execution loop actually ran rather than being skipped.
        """
        completed = _make_noop_steps(
            steps_through(EXPECTED_STEPS, "run_event_search")
        )
        pkl = self._pkl_path()
        _make_fake_pickle_with_stop_conditions(
            pkl, completed, dry_run=True
        )  # ← bug bait

        with patch_fitter_methods(
            self,
            # Construct a temporary fitter just to get a patchable
            # instance; the real fitter is created inside the with-block.
            MMEXOFASTFitter(
                files=GROUND_DATA_FILES,
                coords=COORDS,
                fit_type="point_lens",
                renormalize_errors=False,
                dry_run=True,
            ),
            EXPECTED_STEPS,
        ) as stack:
            fitter = MMEXOFASTFitter(
                files=GROUND_DATA_FILES,
                coords=COORDS,
                fit_type="point_lens",
                renormalize_errors=False,
                restart_file=pkl,
                dry_run=False,  # explicit False
                stop_after="fit_static_point_lens:estimate_point_lens_parameters",
            )

            # Re-attach mocks to the real fitter's methods.
            for step_name, method_name in _STEP_TO_METHOD.items():
                if hasattr(fitter, method_name):
                    setattr(
                        fitter,
                        method_name,
                        stack.mocks.get(
                            step_name, MagicMock(return_value=None)
                        ),
                    )
            fitter.fit()

        # At least estimate_point_lens_parameters must have been executed, not just planned.
        completed_names = [
            step.name
            for step in fitter.completed_steps
            if step.name != "run_event_search"
        ]  # pre-loaded step
        self.assertIn("estimate_point_lens_parameters", completed_names)


class TestWorkflowStepValueError(unittest.TestCase):
    """
    WorkflowStep.run() when func() always raises ValueError.

    Covers the Cartesian product of:
        max_retries : 0, 1, 2
        required    : True, False
    """

    def _make_failing_step(self, *, max_retries=0, required=True):
        """Return a WorkflowStep whose func always raises ValueError('boom')."""
        return WorkflowStep(
            name="failing_step",
            stage="test_stage",
            func=MagicMock(side_effect=ValueError("boom")),
            description="A step that always fails",
            max_retries=max_retries,
            required=required,
        )

    # ------------------------------------------------------------------ #
    # required=True — ValueError must propagate                           #
    # ------------------------------------------------------------------ #

    def test_zero_retries_required_raises(self):
        """max_retries=0, required=True: ValueError is re-raised."""
        step = self._make_failing_step(max_retries=0, required=True)
        with self.assertRaises(ValueError):
            step.run()

    def test_one_retry_required_raises(self):
        """max_retries=1, required=True: ValueError is re-raised."""
        step = self._make_failing_step(max_retries=1, required=True)
        with self.assertRaises(ValueError):
            step.run()

    def test_two_retries_required_raises(self):
        """max_retries=2, required=True: ValueError is re-raised."""
        step = self._make_failing_step(max_retries=2, required=True)
        with self.assertRaises(ValueError):
            step.run()

    # ------------------------------------------------------------------ #
    # required=False — run() returns without raising                      #
    # ------------------------------------------------------------------ #

    def test_zero_retries_not_required_does_not_raise(self):
        """max_retries=0, required=False: run() returns without raising."""
        step = self._make_failing_step(max_retries=0, required=False)
        step.run()

    def test_one_retry_not_required_does_not_raise(self):
        """max_retries=1, required=False: run() returns without raising."""
        step = self._make_failing_step(max_retries=1, required=False)
        step.run()

    def test_two_retries_not_required_does_not_raise(self):
        """max_retries=2, required=False: run() returns without raising."""
        step = self._make_failing_step(max_retries=2, required=False)
        step.run()

    # ------------------------------------------------------------------ #
    # Status is always FAILED after all attempts are exhausted            #
    # ------------------------------------------------------------------ #

    def test_zero_retries_required_status_failed(self):
        """max_retries=0, required=True: status is FAILED."""
        step = self._make_failing_step(max_retries=0, required=True)
        with self.assertRaises(ValueError):
            step.run()
        self.assertEqual(step.status, StepStatus.FAILED)

    def test_zero_retries_not_required_status_failed(self):
        """max_retries=0, required=False: status is FAILED."""
        step = self._make_failing_step(max_retries=0, required=False)
        step.run()
        self.assertEqual(step.status, StepStatus.FAILED)

    def test_two_retries_required_status_failed(self):
        """max_retries=2, required=True: status is FAILED."""
        step = self._make_failing_step(max_retries=2, required=True)
        with self.assertRaises(ValueError):
            step.run()
        self.assertEqual(step.status, StepStatus.FAILED)

    def test_two_retries_not_required_status_failed(self):
        """max_retries=2, required=False: status is FAILED."""
        step = self._make_failing_step(max_retries=2, required=False)
        step.run()
        self.assertEqual(step.status, StepStatus.FAILED)

    # ------------------------------------------------------------------ #
    # Attempt count equals 1 + max_retries                                #
    # ------------------------------------------------------------------ #

    def test_zero_retries_one_attempt(self):
        """max_retries=0: exactly one attempt is made."""
        step = self._make_failing_step(max_retries=0, required=False)
        step.run()
        self.assertEqual(step._attempts, 1)

    def test_one_retry_two_attempts(self):
        """max_retries=1: exactly two attempts are made."""
        step = self._make_failing_step(max_retries=1, required=False)
        step.run()
        self.assertEqual(step._attempts, 2)

    def test_two_retries_three_attempts(self):
        """max_retries=2: exactly three attempts are made."""
        step = self._make_failing_step(max_retries=2, required=False)
        step.run()
        self.assertEqual(step._attempts, 3)

    # ------------------------------------------------------------------ #
    # func call count mirrors attempt count                               #
    # ------------------------------------------------------------------ #

    def test_zero_retries_func_called_once(self):
        """max_retries=0: func is called exactly once."""
        step = self._make_failing_step(max_retries=0, required=False)
        step.run()
        step.func.assert_called_once()

    def test_two_retries_func_called_three_times(self):
        """max_retries=2: func is called exactly three times."""
        step = self._make_failing_step(max_retries=2, required=False)
        step.run()
        self.assertEqual(step.func.call_count, 3)

    # ------------------------------------------------------------------ #
    # error attribute stores the raised exception                         #
    # ------------------------------------------------------------------ #

    def test_zero_retries_error_stored(self):
        """max_retries=0: the ValueError is stored in step.error."""
        step = self._make_failing_step(max_retries=0, required=False)
        step.run()
        self.assertIsInstance(step.error, ValueError)

    def test_two_retries_error_stored(self):
        """max_retries=2: the ValueError is stored in step.error."""
        step = self._make_failing_step(max_retries=2, required=False)
        step.run()
        self.assertIsInstance(step.error, ValueError)

    # ------------------------------------------------------------------ #
    # Log output: one WARNING per retry attempt, one ERROR on final fail  #
    # ------------------------------------------------------------------ #

    def test_zero_retries_logs_error_no_warning(self):
        """
        max_retries=0: one ERROR is logged; no WARNING is logged
        (there are no retry attempts to warn about).
        """
        step = self._make_failing_step(max_retries=0, required=False)
        with self.assertLogs(level="WARNING") as cm:
            step.run()
        warnings = [m for m in cm.output if m.startswith("WARNING")]
        errors = [m for m in cm.output if m.startswith("ERROR")]
        self.assertEqual(len(warnings), 0)
        self.assertEqual(len(errors), 1)

    def test_one_retry_logs_one_warning_then_error(self):
        """
        max_retries=1: one WARNING (first attempt, retrying)
        then one ERROR (second attempt, final failure).
        """
        step = self._make_failing_step(max_retries=1, required=False)
        with self.assertLogs(level="WARNING") as cm:
            step.run()
        warnings = [m for m in cm.output if m.startswith("WARNING")]
        errors = [m for m in cm.output if m.startswith("ERROR")]
        self.assertEqual(len(warnings), 1)
        self.assertEqual(len(errors), 1)

    def test_two_retries_logs_two_warnings_then_error(self):
        """
        max_retries=2: two WARNINGs (attempts 1 and 2, retrying)
        then one ERROR (attempt 3, final failure).
        """
        step = self._make_failing_step(max_retries=2, required=False)
        with self.assertLogs(level="WARNING") as cm:
            step.run()
        warnings = [m for m in cm.output if m.startswith("WARNING")]
        errors = [m for m in cm.output if m.startswith("ERROR")]
        self.assertEqual(len(warnings), 2)
        self.assertEqual(len(errors), 1)

    def test_two_retries_required_logs_before_raising(self):
        """
        max_retries=2, required=True: log entries are emitted before the
        exception propagates — the raise does not suppress them.
        """
        step = self._make_failing_step(max_retries=2, required=True)
        with self.assertLogs(level="WARNING") as cm:
            with self.assertRaises(ValueError):
                step.run()
        warnings = [m for m in cm.output if m.startswith("WARNING")]
        errors = [m for m in cm.output if m.startswith("ERROR")]
        self.assertEqual(len(warnings), 2)
        self.assertEqual(len(errors), 1)
