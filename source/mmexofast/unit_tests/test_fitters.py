import os.path
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import MulensModel as mm
import numpy as np
import pytest

import mmexofast as mmexo
from mmexofast import fitters
from mmexofast.config import DATA_PATH


class TestSFitFitter_1(unittest.TestCase):
    def setUp(self):
        datafile = os.path.join(DATA_PATH, "unit_test_data", "pspl4EF_1.dat")
        self.data = mm.MulensData(file_name=datafile, phot_fmt="mag")
        self.true_params, self.input_fluxes = self._parse_header(datafile)
        self.initial_guess = {"t_0": 4.0, "u_0": 0.01, "t_E": 20.0}

    def _parse_header(self, datafile):
        with open(datafile, "r") as file_:
            lines = file_.readlines()

        elements = lines[0].split()
        ulens_params = {}
        for i, element in enumerate(elements):
            if element[-2:] == "':":
                key = element.strip("{")[1:-2]
                ulens_params[key] = float(
                    elements[i + 1].strip(",").strip("}")
                )

        elements = lines[1].split()
        fluxes = {}
        for i, element in enumerate(elements):
            if element == "=":
                fluxes[elements[i - 1]] = float(elements[i + 1].strip(","))

        return ulens_params, fluxes

    def test_initial_model(self):
        sfit = fitters.SFitFitter(
            datasets=self.data, initial_model_params=self.initial_guess
        )
        for key, value in sfit.initial_model_params.items():
            assert self.initial_guess[key] == value

    def test_run(self):
        sfit = fitters.SFitFitter(
            datasets=[self.data], initial_model_params=self.initial_guess
        )
        sfit.run()
        for key, value in sfit.best.items():
            if key in self.true_params.keys():
                np.testing.assert_allclose(
                    value, self.true_params[key], rtol=0.001
                )


class TestSFitFitter_2(TestSFitFitter_1):
    def setUp(self):
        datafile = os.path.join(DATA_PATH, "unit_test_data", "pspl4EF_2.dat")
        self.data = mm.MulensData(file_name=datafile, phot_fmt="mag")
        self.true_params, self.input_fluxes = self._parse_header(datafile)
        self.initial_guess = {"t_0": 4.0, "u_0": 0.7, "t_E": 20.0}


class TestLSFitter(unittest.TestCase):
    @pytest.mark.slow
    def test_run_on_real_anomaly_event(self):
        datafile = os.path.join(DATA_PATH, "unit_test_data", "planet4AF.dat")
        data = mm.MulensData(file_name=datafile, phot_fmt="mag")
        initial_guess = {
            "t_0": 5.2,
            "u_0": 0.28,
            "t_E": 15.5,
            "s": 1.5,
            "q": 1e-5,
            "alpha": 159.0,
            "rho": 0.01,
        }
        fitter = fitters.LSFitter(
            datasets=[data],
            initial_guess=initial_guess,
            parameters_to_fit=["t_0", "u_0", "t_E"],
            bounds={
                "t_0": (4.0, 6.0),
                "u_0": (0.05, 0.8),
                "t_E": (5.0, 30.0),
            },
            mag_methods=[13.0, "VBBL", 22.0],
            mag_methods_parameters={"VBBL": {"accuracy": 0.01}},
        )

        result = fitter.run()

        self.assertEqual(result.x.shape, (3,))
        self.assertTrue(np.all(np.isfinite(result.x)))
        self.assertTrue(np.isfinite(fitter.best["chi2"]))
        self.assertGreaterEqual(fitter.best["t_0"], 4.0)
        self.assertLessEqual(fitter.best["t_0"], 6.0)
        self.assertGreaterEqual(fitter.best["u_0"], 0.05)
        self.assertLessEqual(fitter.best["u_0"], 0.8)
        self.assertGreaterEqual(fitter.best["t_E"], 5.0)
        self.assertLessEqual(fitter.best["t_E"], 30.0)

    def test_parse_bounds_dict_fills_unspecified_parameters(self):
        fitter = fitters.LSFitter(
            initial_model_params={"t_0": 1.0, "u_0": 0.1, "t_E": 10.0},
            parameters_to_fit=["t_0", "u_0", "t_E"],
            bounds={"u_0": (0.0, 1.0)},
        )

        self.assertEqual(
            fitter.bounds,
            ([-np.inf, 0.0, -np.inf], [np.inf, 1.0, np.inf]),
        )

    def test_run_minimizes_negative_log_probability_and_stores_result(self):
        parameters = SimpleNamespace(
            t_0=1.0,
            u_0=0.1,
            parameters={"t_0": 1.0, "u_0": 0.1},
        )
        event = MagicMock()
        event.model.parameters = parameters
        event.get_chi2.return_value = 12.5
        result = SimpleNamespace(x=np.array([1.2, 0.2]), success=True)

        fitter = fitters.LSFitter(
            initial_model_params={"t_0": 1.0, "u_0": 0.1},
            parameters_to_fit=["t_0", "u_0"],
            bounds=([-np.inf, -np.inf], [np.inf, np.inf]),
        )
        fitter._event = event

        with (
            patch.object(fitter, "ln_prob", return_value=-7.0) as ln_prob,
            patch(
                "mmexofast.fitters.minimize", return_value=result
            ) as minimize,
        ):
            returned = fitter.run()
            objective = minimize.call_args.args[0]
            self.assertEqual(objective(result.x), 7.0)
            ln_prob.assert_called_once_with(result.x)

        minimize.assert_called_once()
        np.testing.assert_array_equal(fitter.best_theta, result.x)
        self.assertIs(returned, result)
        self.assertEqual(fitter.results["chi2"], 12.5)
        self.assertEqual(fitter.best["chi2"], 12.5)

    def test_invalid_bounds_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            fitters.LSFitter(
                initial_model_params={"t_0": 1.0},
                parameters_to_fit=["t_0"],
                bounds=([-np.inf],),
            )


class TestAnomalyFitter(unittest.TestCase):
    def setUp(self):
        datafile = os.path.join(DATA_PATH, "unit_test_data", "planet4AF.dat")
        self.data = mm.MulensData(file_name=datafile, phot_fmt="mag")
        self.true_params, self.input_fluxes = self._parse_header(datafile)
        # self.pspl = {key: self.true_params[key] for key in ['t_0', 'u_0', 't_E']}
        self.anomaly_lc_params = {
            "t_0": 5.012676614091774,
            "u_0": 0.3001165942835024,
            "t_E": 16.049057513235873,
            "t_pl": 17.43489583333333,
            "dt": 0.421875,
            "dmag": -0.12254925495518165,
        }
        # self.fitter = fitters.WidePlanetFitter(datasets=[self.data], anomaly_lc_params=self.anomaly_lc_params)

    def _parse_header(self, datafile):
        return TestSFitFitter_1._parse_header(self, datafile=datafile)

    #    def test_run(self):
    # This test takes too long and then segfaults
    #        fitter = fitters.WidePlanetFitter(
    #            datasets=[self.data], anomaly_lc_params=self.anomaly_lc_params)
    #        fitter.run()
    #        for key, value in fitter.best.items():
    #            if key in self.true_params.keys():
    #                np.testing.assert_allclose(value, self.true_params[key], rtol=0.01)

    def test_datasets(self):
        fitter = fitters.AnomalyFitter(
            datasets=[self.data],
            initial_guess=self.true_params,
            anomaly_lc_params=self.anomaly_lc_params,
            mag_methods=[],
        )
        assert fitter.datasets[0] == self.data
        assert fitter.datasets[0].n_epochs == 1600

    def test_mag_methods_error(self):
        with self.assertRaises(ValueError):
            test_fitter = fitters.AnomalyFitter(
                datasets=[self.data], initial_guess=self.true_params
            )
            test_fitter.initialize_event()

    def test_event_creation(self):
        test_fitter = fitters.AnomalyFitter(
            datasets=[self.data],
            initial_guess=self.true_params,
            mag_methods=[
                self.anomaly_lc_params["t_pl"]
                - 10.0 * self.anomaly_lc_params["dt"],
                "VBBL",
                self.anomaly_lc_params["t_pl"]
                + 10.0 * self.anomaly_lc_params["dt"],
            ],
        )

        assert test_fitter._event is None
        assert isinstance(test_fitter.initial_guess, dict)

        assert len(test_fitter.parameters_to_fit) == 7
        test_fitter.initialize_event()
        for parameter in self.true_params.keys():
            assert (
                test_fitter.event.model.parameters.as_dict()[parameter]
                == self.true_params[parameter]
            )

    @pytest.mark.slow
    def test_event_update(self):
        fitter = fitters.AnomalyFitter(
            datasets=[self.data],
            initial_guess=self.true_params,
            anomaly_lc_params=self.anomaly_lc_params,
            mag_methods=[
                self.anomaly_lc_params["t_pl"]
                - 10.0 * self.anomaly_lc_params["dt"],
                "VBBL",
                self.anomaly_lc_params["t_pl"]
                + 10.0 * self.anomaly_lc_params["dt"],
            ],
        )
        theta = 9 + np.arange(len(fitter.parameters_to_fit), dtype=int)
        # print(dict(zip(fitter.parameters_to_fit, theta)))
        # Assigning theta before initialize_event() lazily builds the event
        # (a multiprocessing pool worker arrives with _event=None, since
        # __getstate__ drops it; it must not raise).
        assert fitter._event is None
        fitter.event = theta
        assert fitter._event is not None

        fitter.initialize_event()
        fitter.event = theta
        for parameter in fitter.event.model.parameters.parameters.keys():
            if parameter in fitter.parameters_to_fit:
                index = fitter.parameters_to_fit.index(parameter)
                assert (
                    fitter.event.model.parameters.parameters[parameter]
                    == 9 + index
                )
            elif "log_{0}".format(parameter) in fitter.parameters_to_fit:
                index = fitter.parameters_to_fit.index(
                    "log_{0}".format(parameter)
                )
                np.testing.assert_almost_equal(
                    fitter.event.model.parameters.parameters[parameter],
                    10.0 ** (9 + index),
                )
            else:
                print("{0} not in parameters_to_fit".format(parameter))
