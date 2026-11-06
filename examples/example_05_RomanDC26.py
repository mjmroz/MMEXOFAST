"""
Analyze a planet light curve from the 2018 Data Challenge. Minimal user effort.
"""

import glob
import os.path
import traceback
from pathlib import Path

import numpy as np
import concurrent.futures

from DC18_classes import TestDataSet, TestDataSet26, dir_, dir26Beginner

import mmexofast as mmexo
from mmexofast.config import MODULE_PATH
import MulensModel

prefix = (
    "high_grid_chi2_refine_kh"  # or "no_par_small_grid" or "no_par_original"
)

base_dir = os.path.join(MODULE_PATH, "DC26Test", "temp_output", prefix)


def fit_lc(lc_num, verbose=False):
    data = TestDataSet26(lc_num)
    output_dir = os.path.join(base_dir, "W{0:03}".format(lc_num))
    os.makedirs(output_dir, exist_ok=True)

    file_prefix = "WFIRST.{0:03}".format(lc_num)
    
    datasets = [
        MulensModel.MulensData(file_name=data.file_F146, phot_fmt="mag", zeropoint=26.0),
        MulensModel.MulensData(file_name=data.file_F087, phot_fmt="mag", zeropoint=26.0),
        MulensModel.MulensData(file_name=data.file_F213, phot_fmt="mag", zeropoint=26.0), 
    ]
    
    fitter = mmexo.MMEXOFASTFitter(
        datasets=datasets,
        coords=data.coords,
        fit_type="binary_lens",
        verbose=verbose,
        renormalize_errors=False,
        no_parallax=True,
        grid_n_jobs=1,
        parallel_refinement=False,
        stop_after="fit_binary_lens:estimate_binary_lens_parameters",
        log_file=os.path.join(output_dir, file_prefix + ".log"),
        restart_file=os.path.join(output_dir, file_prefix + ".pkl"),
        #location_files=[data.file_F146_loc, data.file_F087_loc, data.file_F213_loc],
        limb_darkening_coeffs_gamma={"F146": 0.0, "F087": 0.0, "F213": 0.0},
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




files = glob.glob(os.path.join(dir26Beginner, "RMDC26_*"))
lc_nums = []
for file_ in files:
    elements = file_.split("_")
    lc_nums.append(int(elements[-1]))

# lc_nums for special cases
wide_planets = [8, 53, 107, 131, 152, 194, 208, 214, 217, 226]
big_wide_planets = [4, 62]
close_planets = [32, 40, 50, 74, 92, 95, 87, 186, 227]
big_close_planets = [27, 120, 124, 128, 172]
slow_parallax = [124, 128, 217]  # 66 is broke
dip_anom = [47, 74, 95, 103]
bad_t0 = [12, 40, 81, 107, 208, 217, 227]

# lc_nums = [69]
#lc_nums= [62, 69, 163,193, 258] #high_mag
workers = 20
def _run_parallel(lc_list):
    # run fits in parallel processes
    # limit CPU usage: default to min(4, available_cpus) but allow override via
    # environment variable EXAMPLES_MAX_WORKERS
    try:
        avail = os.cpu_count() or 1
    except Exception:
        avail = 1
    max_workers_env = os.environ.get("EXAMPLES_MAX_WORKERS")
    if max_workers_env is not None:
        try:
            max_workers = max(workers, int(max_workers_env))
        except Exception:
            max_workers = min(workers, avail)
    else:
        max_workers = min(workers, avail)

    # also limit threaded libraries to one thread per process to avoid oversubscription
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    if max_workers == 1:
        for lc in lc_list:
            try:
                fit_lc(lc, True)
                try:
                    evaluate_results(lc)
                except NotImplementedError:
                    pass
            except Exception as e:
                print(f"Run {lc} ABORTED. {type(e).__name__}: {e}")
                traceback.print_exc()
        return

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers
    ) as exe:
        futures = {
            exe.submit(fit_lc, lc_num, True): lc_num for lc_num in lc_list
        }
        for fut in concurrent.futures.as_completed(futures):
            lc = futures[fut]
            try:
                res = fut.result()
                try:
                    evaluate_results(lc)
                except NotImplementedError:
                    pass
            except Exception as e:
                print(f"Run {lc} ABORTED. {type(e).__name__}: {e}")
                traceback.print_exc()


if __name__ == "__main__":
    for lc_num in np.sort(lc_nums):
        print("\n...Scheduling light curve {0}...".format(lc_num))
    _run_parallel(np.sort(lc_nums))
