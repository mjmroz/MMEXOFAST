"""
Analyze a planet light curve from the 2018 Data Challenge. Minimal user effort.
"""

import glob
import os.path
import traceback
from pathlib import Path

import numpy as np
from DC18_classes import TestDataSet, dir_

import mmexofast as mmexo
from mmexofast.config import MODULE_PATH

base_dir = os.path.join(MODULE_PATH, "DC18Test", "temp_output", "no_par")


def fit_lc(lc_num, verbose=False):
    data = TestDataSet(lc_num)
    output_dir = os.path.join(base_dir, "W{0:03}".format(lc_num))
    os.makedirs(output_dir, exist_ok=True)

    file_prefix = "WFIRST.{0:03}".format(lc_num)
    fitter = mmexo.MMEXOFASTFitter(
        files=[data.file_w149, data.file_z087],
        coords=data.coords,
        fit_type="binary_lens",
        verbose=verbose,
        renormalize_errors=False,
        no_parallax=True,
        stop_after="fit_binary_lens:estimate_binary_lens_parameters",

        log_file=os.path.join(output_dir, file_prefix + ".log"),
        restart_file=os.path.join(output_dir, file_prefix + ".pkl"),
        limb_darkening_coeffs_gamma={"W149": 0.0, "Z087": 0.0},
        output_config=mmexo.OutputConfig(
            output_dir=Path(output_dir),
            file_prefix=file_prefix,
            save_plots=True,
            save_table=True,
            save_exozippy_init=False,
        ),

    )
    try:
        fitter.fit()
        print("exozippy input:\n", fitter.initialize_exozippy())
    finally:
        fitter.close()
    # results = fitter.all_fit_results


def evaluate_results(lc_num):
    """
    Calculate metrics between input and output values
    Assume pymc output.
    """
    pass


files = glob.glob(os.path.join(dir_, "n2018*.W149.*.txt"))
lc_nums = []
for file_ in files:
    elements = file_.split(".")
    lc_nums.append(int(elements[-2]))

# lc_nums for special cases
wide_planets = [8, 53, 107, 131, 152, 194, 208, 214, 217, 226]
big_wide_planets = [4, 62]
close_planets = [32, 40, 50, 74, 92, 95, 87, 186, 227]
big_close_planets = [27, 120, 124, 128, 172]
slow_parallax = [124, 128, 217]  # 66 is broke
dip_anom = [47, 74, 95, 103]
bad_t0 = [12, 40, 81, 107, 208, 217, 227]

# lc_nums = [74]
for lc_num in np.sort(lc_nums):
    print("\n...Fitting light curve {0}...".format(lc_num))
    try:
        results = fit_lc(lc_num, verbose=True)
        evaluate_results(lc_num)
    except NotImplementedError:
        pass
    except Exception as e:
        print("Run {0} ABORTED. {1}: {2}".format(lc_num, type(e).__name__, e))
        traceback.print_exc()
