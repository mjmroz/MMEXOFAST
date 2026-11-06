import os.path
import unittest
from unittest.mock import Mock, patch

import matplotlib.pyplot as plt
import MulensModel as mm
import numpy as np

import mmexofast as mmexo
from mmexofast import gridsearches
from mmexofast.config import DATA_PATH


class TestEventFinderGridSearch_1(unittest.TestCase):
    def setUp(self):
        datafile = os.path.join(DATA_PATH, "unit_test_data", "pspl4EF_1.dat")
        self.data = mm.MulensData(file_name=datafile, phot_fmt="mag")
        self.expected = self._parse_header(datafile)
        self.ef_grid = gridsearches.EventFinderGridSearch(datasets=self.data)

    def _parse_header(self, datafile):
        with open(datafile, "r") as file_:
            lines = file_.readlines()

        elements = lines[2].split()
        expected = {}
        for i, element in enumerate(elements):
            if element == "=":
                expected[elements[i - 1]] = float(elements[i + 1].strip(","))

        return expected

    def test_grid_t_0(self):
        np.testing.assert_almost_equal(
            self.ef_grid.grid_params["t_0_min"],
            self.data.time[0] - (1.0 / 3.0),
        )
        np.testing.assert_almost_equal(
            self.ef_grid.grid_params["t_0_max"],
            self.data.time[-1] + (1.0 / 3.0),
        )

    def test_grid_t_eff(self):
        np.testing.assert_almost_equal(
            self.ef_grid.grid_params["t_eff_3"], 1.0
        )
        np.testing.assert_almost_equal(
            self.ef_grid.grid_params["t_eff_max"], 99.0
        )
        np.testing.assert_almost_equal(
            self.ef_grid.grid_params["d_t_eff"], 1.0 / 3.0
        )

    def test_get_trimmed_datasets(self):
        params = {"t_0": 3.0, "t_eff": 5.0}
        # print(self.ef_grid.z_t_eff)
        n_expected = (
            params["t_eff"] * 2 * 5 / (self.data.time[1] - self.data.time[0])
        )
        trimmed_data = self.ef_grid.get_trimmed_datasets(parameters=params)
        for dataset in trimmed_data:
            # print(n_expected, dataset.n_epochs, self.data.n_epochs)
            np.testing.assert_almost_equal(dataset.n_epochs, n_expected)

    def test_get_flat_chi2(self):
        self.skipTest(
            "Technically this part of the code is covered. But it might be a good idea to have a special test for "
            + "debugging."
        )

    def test_do_fits(self):
        self.skipTest(
            "Technically this part of the code is covered. But it might be a good idea to have a special test for "
            + "debugging."
        )

    def test_best(self):
        self.ef_grid.run()
        # for i in range(self.ef_grid.results.shape[1]):
        #    plt.figure()
        #    plt.title(i + 1)
        #    plt.scatter(self.ef_grid.grid_t_0, self.ef_grid.grid_t_eff, c=self.ef_grid.results[:, i])
        #    plt.colorbar()
        #
        # plt.show()

        index = self.ef_grid.grid_t_eff == self.ef_grid.best["t_eff"]
        closest_t_0 = np.argmin(
            np.abs(self.ef_grid.grid_t_0[index] - self.expected["t_0"])
        )
        best_expected = self.ef_grid.grid_t_0[index][closest_t_0]
        np.testing.assert_almost_equal(self.ef_grid.best["t_0"], best_expected)


class TestEventFinderGridSearch_2(TestEventFinderGridSearch_1):
    def setUp(self):
        datafile = os.path.join(DATA_PATH, "unit_test_data", "pspl4EF_2.dat")
        self.data = mm.MulensData(file_name=datafile, phot_fmt="mag")
        self.expected = self._parse_header(datafile)
        self.ef_grid = gridsearches.EventFinderGridSearch(datasets=self.data)

    def test_bad_input_data(self):
        with self.assertRaises(ValueError):
            foo = gridsearches.EventFinderGridSearch(datasets=None)

        with self.assertRaises(TypeError):
            foo = gridsearches.EventFinderGridSearch(datasets=16.0)

    def test_set_t_0_multiple_datasets(self):
        alt_times = [time for time in self.data.time]
        alt_times[0] -= 10.0
        alt_times[-1] += 10.0
        data_2 = mm.MulensData(
            [alt_times, self.data.mag, self.data.err_mag], phot_fmt="mag"
        )

        datasets = [self.data, data_2]
        test_grid = gridsearches.EventFinderGridSearch(datasets=datasets)

        np.testing.assert_almost_equal(
            test_grid.grid_params["t_0_min"], alt_times[0] - (1.0 / 3.0)
        )
        np.testing.assert_almost_equal(
            test_grid.grid_params["t_0_max"], alt_times[-1] + (1.0 / 3.0)
        )

    def test_set_t_0_manual(self):
        t_0_min = -5.0
        t_0_max = 16.0
        test_grid = gridsearches.EventFinderGridSearch(
            datasets=self.data, t_0_min=t_0_min, t_0_max=t_0_max
        )
        np.testing.assert_almost_equal(
            test_grid.grid_params["t_0_min"], t_0_min
        )
        np.testing.assert_almost_equal(
            test_grid.grid_params["t_0_max"], t_0_max
        )


class TestFlatSFitFunction(unittest.TestCase):
    def test_flatten_data(self):
        self.skipTest(" NotImplemented")

    def test_calc_model(self):
        self.skipTest(" NotImplemented")

    def test_calc_df(self):
        self.skipTest(" NotImplemented")


class TestBinaryGridSearch(unittest.TestCase):
    def test_explicit_s_and_q_values_are_used(self):
        search = gridsearches.BinaryGridSearch(
            datasets=[],
            static_params={},
            s_values=[0.41, 0.73, 2.46],
            q_values=[0.001, 0.017, 0.3],
            alpha_min=0.0,
            alpha_max=0.0,
            alpha_step=1.0,
        )

        metadata = search._build_grid_metadata()
        np.testing.assert_array_equal(
            metadata["param_arrays"]["s"], [0.41, 0.73, 2.46]
        )
        np.testing.assert_array_equal(
            metadata["param_arrays"]["q"], [0.001, 0.017, 0.3]
        )

    def test_kh_parameterization_converts_grid_points_to_s_and_q(self):
        search = gridsearches.BinaryGridSearch(
            datasets=[],
            static_params={},
            parameterization="kh",
            k_values=[-2.0, -1.0],
            h_values=[-1.0, 1.0],
            alpha_min=0.0,
            alpha_max=0.0,
            alpha_step=1.0,
        )

        model_params = search._set_model_params_from_grid(
            {"k": -1.0, "h": 1.0, "alpha": 0.0}
        )
        expected_s, expected_q = (
            gridsearches.KHParameterization().from_kh_to_sq(-1.0, 1.0)
        )
        self.assertAlmostEqual(model_params["s"], expected_s)
        self.assertAlmostEqual(model_params["q"], expected_q)
        self.assertEqual(search.grid_params.keys(), {"k", "h", "alpha"})

    def test_kh_parameterization_applies_s_and_q_bounds(self):
        search = gridsearches.BinaryGridSearch(
            datasets=[],
            static_params={},
            parameterization="kh",
            k_values=[-4.0, 2.5],
            h_values=[0.0, 5.0],
            alpha_min=0.0,
            alpha_max=0.0,
            alpha_step=1.0,
            bounds={"s": (0.0, 10.0), "q": (0.0, 1.0)},
        )

        result = search._fit_grid_point({"k": 2.5, "h": 5.0, "alpha": 0.0})

        self.assertFalse(result["success"])
        self.assertIn("outside bounds", result["error"])

    def test_kh_chi2_result_keeps_grid_parameters_for_refinement(self):
        model = object()
        event = Mock()
        event.get_chi2.return_value = 12.5
        model_config = Mock()
        model_config.build.return_value = model
        event_config = Mock()
        event_config.build.return_value = event
        search = gridsearches.BinaryGridSearch(
            datasets=[],
            static_params={},
            parameterization="kh",
            k_values=[-2.0],
            h_values=[1.0],
            alpha_min=0.0,
            alpha_max=0.0,
            alpha_step=1.0,
            fitter_kwargs={
                "model_config": model_config,
                "event_config": event_config,
            },
        )

        result = search._fit_grid_point({"k": -2.0, "h": 1.0, "alpha": 0.0})

        self.assertEqual(result["params"]["k"], -2.0)
        self.assertEqual(result["params"]["h"], 1.0)
        self.assertEqual(result["params"]["alpha"], 0.0)

    def test_refinement_can_be_limited_to_best_minima(self):
        search = gridsearches.BinaryGridSearch(
            datasets=[],
            static_params={},
            max_refinement_minima=1,
        )
        worse = {
            "chi2": 20.0,
            "indices": (1, 1),
            "params": {},
            "refinement_history": [],
        }
        best = {
            "chi2": 5.0,
            "indices": (1, 1),
            "params": {},
            "refinement_history": [],
        }
        minima = [
            worse,
            best,
        ]
        search.results_history = [
            {
                "chi2_grid": np.zeros((3, 3)),
                "result_grid": np.empty((3, 3), dtype=object),
                "metadata": {},
            }
        ]
        search._find_minima_in_grid = Mock(return_value=minima)
        search._is_edge_minimum = Mock(return_value=False)
        search._refine_minimum = Mock(return_value=[])

        search._run_refinement(5, 1, 1, "standard", False, 1)

        search._refine_minimum.assert_called_once_with(best, 5, 1, False, 1)
        self.assertEqual(search.minima, [worse])

    def test_refinement_bounds_clip_refined_grid(self):
        search = gridsearches.BinaryGridSearch(
            datasets=[],
            static_params={},
            s_min=0.5,
            s_max=2.0,
            s_n=3,
            q_min=0.01,
            q_max=0.1,
            q_n=3,
            alpha_min=0.0,
            alpha_max=30.0,
            alpha_step=30.0,
            refinement_bounds={"alpha": (0.0, 20.0), "q": (0.02, 0.08)},
        )

        spec = search._compute_refined_spec(
            {
                "params": {"s": 1.0, "q": 0.03, "alpha": 5.0},
            },
            {
                "param_names": ["s", "q", "alpha"],
                "steps": {"s": 0.5, "q": 0.03, "alpha": 30.0},
            },
            n=5,
        )

        assert spec["alpha"][0] >= 0.0
        assert spec["alpha"][1] <= 20.0
        assert spec["q"][0] >= 0.02
        assert spec["q"][1] <= 0.08

    def test_run_uses_binary_grid_configuration(self):
        search = gridsearches.BinaryGridSearch(
            datasets=[],
            static_params={},
            s_min=0.5,
            s_max=2.0,
            s_n=2,
            q_min=0.01,
            q_max=0.1,
            q_n=2,
            alpha_min=0.0,
            alpha_max=30.0,
            alpha_step=30.0,
        )
        search._fit_grid_point = lambda params: {
            "chi2": sum(params.values()),
            "params": {},
            "success": True,
        }

        search.run()

        self.assertEqual(len(search.results), 8)
        np.testing.assert_allclose(
            sorted(search.results["s"].unique()), [0.5, 2.0]
        )
        np.testing.assert_allclose(
            sorted(search.results["q"].unique()), [0.01, 0.1]
        )

    def test_progress_bar_is_enabled(self):
        search = gridsearches.BinaryGridSearch(
            datasets=[],
            static_params={},
            s_min=0.5,
            s_max=2.0,
            s_n=2,
            q_min=0.01,
            q_max=0.1,
            q_n=2,
            alpha_min=0.0,
            alpha_max=30.0,
            alpha_step=30.0,
            progress=True,
        )
        search._fit_grid_point = lambda params: {
            "chi2": sum(params.values()),
            "params": {},
            "success": True,
        }

        with patch(
            "mmexofast.gridsearches.tqdm", side_effect=lambda x, **_: x
        ) as progress:
            search.run()

        progress.assert_called_once()
        self.assertEqual(progress.call_args.kwargs["total"], 8)

    def test_lsfitter_bounds_use_numeric_neighboring_grid_values(self):
        search = gridsearches.BinaryGridSearch(
            datasets=[],
            static_params={"t_0": 1.0},
            s_min=0.5,
            s_max=2.0,
            s_n=3,
            q_min=0.001,
            q_max=1.0,
            q_n=3,
            alpha_min=0.0,
            alpha_max=360.0,
            alpha_step=30.0,
            fitter_kwargs={"parameters_to_fit": ["s", "q", "alpha"]},
            fit_method="LSFitter",
        )

        bounds = search._get_bounds({"s": 0.5, "q": 0.001, "alpha": 360.0})

        np.testing.assert_allclose(bounds["s"], (0.5, 1.0))
        np.testing.assert_allclose(bounds["q"], (0.001, 0.03162277660168379))
        np.testing.assert_allclose(bounds["alpha"], (330.0, 360.0))

    def test_grid_cell_only_evaluates_chi2(self):
        model = object()
        event = Mock()
        event.get_chi2.return_value = 12.5
        model_config = Mock()
        model_config.build.return_value = model
        event_config = Mock()
        event_config.build.return_value = event
        search = gridsearches.BinaryGridSearch(
            datasets=[],
            static_params={"t_0": 1.0},
            s_min=1.0,
            s_max=1.0,
            s_n=1,
            q_min=0.1,
            q_max=0.1,
            q_n=1,
            alpha_min=0.0,
            alpha_max=0.0,
            alpha_step=1.0,
            fitter_kwargs={
                "model_config": model_config,
                "event_config": event_config,
            },
        )

        result = search._fit_grid_point({"s": 1.0, "q": 0.1, "alpha": 0.0})

        self.assertEqual(result["chi2"], 12.5)
        model_config.build.assert_called_once()
        event_config.build.assert_called_once_with(model=model, datasets=[])
        event.get_chi2.assert_called_once_with()

    def test_grid_cell_lsfitter_sets_initial_guess_and_uses_result(self):
        fitter = Mock()
        fitter.best = {
            "t_0": 1.1,
            "s": 1.0,
            "q": 0.1,
            "alpha": 0.0,
            "chi2": 3.5,
        }
        fitter.run.return_value.success = True

        with patch("mmexofast.fitters.LSFitter", return_value=fitter) as lsf:
            search = gridsearches.BinaryGridSearch(
                datasets=[],
                static_params={"t_0": 1.0, "u_0": 0.1, "t_E": 10.0},
                s_min=1.0,
                s_max=1.0,
                s_n=1,
                q_min=0.1,
                q_max=0.1,
                q_n=1,
                alpha_min=0.0,
                alpha_max=0.0,
                alpha_step=1.0,
                fitter_kwargs={"parameters_to_fit": ["s", "q", "alpha"]},
                fit_method="LSFitter",
            )
            result = search._fit_grid_point({"s": 1.0, "q": 0.1, "alpha": 0.0})

        self.assertEqual(result["chi2"], 3.5)
        lsf.assert_called_once()
        self.assertEqual(
            lsf.call_args.kwargs["initial_guess"],
            {
                "t_0": 1.0,
                "u_0": 0.1,
                "t_E": 10.0,
                "s": 1.0,
                "q": 0.1,
                "alpha": 0.0,
            },
        )
        fitter.run.assert_called_once_with()

    def test_grid_cell_can_use_sfit(self):
        fitter = Mock()
        fitter.best = {"t_0": 1.1, "chi2": 3.5}
        fitter.results.success = True
        with patch(
            "mmexofast.fitters.SFitFitter", return_value=fitter
        ) as sfit:
            search = gridsearches.BinaryGridSearch(
                datasets=[],
                static_params={"t_0": 1.0},
                fitter_kwargs={"mag_methods": "point_source_point_lens"},
                fit_method="sfit",
            )
            result = search._fit_grid_point({"s": 1.0, "q": 0.1, "alpha": 0.0})

        self.assertEqual(result["chi2"], 3.5)
        sfit.assert_called_once()
        self.assertEqual(
            sfit.call_args.kwargs["parameters_to_fit"], ["s", "q", "alpha"]
        )
        fitter.run.assert_called_once_with()

    def test_grid_cell_preserves_custom_parameters_to_fit(self):
        fitter = Mock()
        fitter.best = {"t_0": 1.1, "chi2": 3.5}
        fitter.results.success = True
        with patch(
            "mmexofast.fitters.SFitFitter", return_value=fitter
        ) as sfit:
            search = gridsearches.BinaryGridSearch(
                datasets=[],
                static_params={"t_0": 1.0},
                fitter_kwargs={
                    "mag_methods": "point_source_point_lens",
                    "parameters_to_fit": ["t_0", "u_0", "t_E", "rho"],
                },
                fit_method="sfit",
            )
            search._fit_grid_point({"s": 1.0, "q": 0.1, "alpha": 0.0})

        self.assertEqual(
            sfit.call_args.kwargs["parameters_to_fit"],
            ["t_0", "u_0", "t_E", "rho"],
        )

    def test_invalid_fit_method(self):
        with self.assertRaises(ValueError):
            gridsearches.BinaryGridSearch(
                datasets=[], static_params={}, fit_method="invalid"
            )

    def test_results_include_failed_cells_without_fitted_params(self):
        search = gridsearches.BinaryGridSearch(
            datasets=[],
            static_params={},
            s_min=1.0,
            s_max=1.0,
            s_n=1,
            q_min=0.1,
            q_max=0.1,
            q_n=1,
            alpha_min=0.0,
            alpha_max=0.0,
            alpha_step=1.0,
        )
        search._fit_grid_point = lambda params: {
            "chi2": np.nan,
            "params": None,
            "success": False,
        }

        search.run()

        self.assertFalse(search.results.loc[0, "success"])
        self.assertEqual(search.results.loc[0, "s"], 1.0)


class TestEFSFitFunction(unittest.TestCase):
    def test_calc_model(self):
        self.skipTest(" NotImplemented")

    def test_calc_residuals(self):
        self.skipTest(" NotImplemented")

    def test_calc_df(self):
        self.skipTest(" NotImplemented")

    def test_q(self):
        self.skipTest(" NotImplemented")

    def test_magnification(self):
        self.skipTest(" NotImplemented")

    def test_get_magnification(self):
        self.skipTest(" NotImplemented")


class TestAnomalyFinderGridSearch(unittest.TestCase):
    def setUp(self):
        datafile = os.path.join(DATA_PATH, "unit_test_data", "planet4AF.dat")
        self.data = mm.MulensData(file_name=datafile, phot_fmt="mag")
        self.expected = self._parse_header(datafile)
        print(self.expected)

        self.residuals = self._get_residuals()
        self.af_grid = gridsearches.AnomalyFinderGridSearch(
            residuals=self.residuals
        )

    def _parse_header(self, datafile):
        with open(datafile, "r") as file_:
            lines = file_.readlines()

        elements = lines[0].split()
        expected = {}
        for i, element in enumerate(elements):
            if element[-2:] == "':":
                key = element.strip("{")[1:-2]
                expected[key] = float(elements[i + 1].strip(",").strip("}"))

        elements = lines[2].split()
        for i, element in enumerate(elements):
            if element == "=":
                expected[elements[i - 1]] = float(elements[i + 1].strip(","))

        return expected

    def _get_residuals(self):
        pspl_model = mm.Model(
            parameters={
                "t_0": self.expected["t_0"],
                "u_0": self.expected["u_0"],
                "t_E": self.expected["t_E"],
            }
        )
        fit = mm.FitData(model=pspl_model, dataset=self.data)
        fit.fit_fluxes()
        res, err = fit.get_residuals(phot_fmt="flux")
        residuals = mm.MulensData([self.data.time, res, err], phot_fmt="flux")
        return residuals

    def test_run(self):
        self.af_grid.run()

        # t_eff is reasonable
        t_star = self.expected["t_E"] * self.expected["rho"]
        assert self.af_grid.best["t_eff"] < (5.0 * t_star)

        # t_0 is as close to the expected value as possible given t_eff and the data spacing.
        index = self.af_grid.grid_t_eff == self.af_grid.best["t_eff"]
        closest_t_0 = np.argmin(
            np.abs(self.af_grid.grid_t_0[index] - self.expected["t_pl"])
        )
        best_expected = self.af_grid.grid_t_0[index][closest_t_0]
        np.testing.assert_allclose(
            self.af_grid.best["t_0"],
            best_expected,
            atol=(self.data.time[1] - self.data.time[0]),
        )

    def test_grid_t_0(self):
        np.testing.assert_almost_equal(
            self.af_grid.grid_params["t_0_min"],
            self.data.time[0] - (1.0 / 6.0),
        )
        np.testing.assert_almost_equal(
            self.af_grid.grid_params["t_0_max"],
            self.data.time[-1] + (1.0 / 6.0),
        )

    def test_grid_t_eff(self):
        np.testing.assert_almost_equal(
            self.af_grid.grid_params["t_eff_3"], 0.75
        )
        np.testing.assert_almost_equal(
            self.af_grid.grid_params["t_eff_max"], 10.0
        )
        np.testing.assert_almost_equal(
            self.af_grid.grid_params["d_t_eff"], 1.0 / 3.0
        )

    def test_get_zero_chi2(self):
        self.skipTest(" NotImplemented")

    def test_check_successive(self):
        self.skipTest(" NotImplemented")

    def test_get_anomalies(self):
        self.skipTest(" NotImplemented")

    def test_filter_anomalies(self):
        self.skipTest(" NotImplemented")

    def test_anomalies(self):
        self.skipTest(" NotImplemented")
