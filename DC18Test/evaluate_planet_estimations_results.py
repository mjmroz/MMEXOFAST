import ast
import glob
from os import name
import os.path
import pickle
import re
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from DC18_classes import DC18Answers

prefix = "no_par_small_grid"  # or 'no_par_original'
# prefix = 'no_par_original'  # or 'no_par_small_grid'
# prefix = 'no_par_smaller_grid_0.01'  # or 'no_par_small_grid'
# prefix = 'no_par_smaller_grid_0.0'  # or 'no_par_small_grid'
# prefix = 'no_par_no_grid'  # or 'no_par_small_grid'
# prefix = 'no_par_no_grid_close_qmodf'  # or 'no_par_small_grid'
prefixs = [
    "no_par_no_grid_close_qmodf",
    "no_par_no_grid",
    "no_par_smaller_grid_0.01",
    "no_par_smaller_grid_0.0",
    "no_par_small_grid",
    "no_par_original",
]


class SelectionStrategy(Enum):
    """
    Determines which estimated parameter sets are used for evaluation.

    PRIMARY_FIRST : One row per LC — the first-listed primary (non-alternate)
                    solution, which corresponds to the classified anomaly type
                    (e.g. WidePlanet when anomaly_type = wide).
                    Backwards-compatible; gives a strict one-to-one LC↔row mapping.

    ALL_PRIMARY   : All primary (non-alternate) solutions. Typically yields
                    WidePlanet, CloseUpperPlanet, and CloseLowerPlanet rows per
                    LC.  A single LC can appear in both "good" and "bad" lists.

    ALL           : Every solution including alternate (s_dagger) solutions.

    Note: A future ANY_CORRECT strategy — flagging a LC as good if *any*
    solution is close to truth — requires grouping by idx rather than counting
    rows and is not yet implemented here.
    """

    PRIMARY_FIRST = "primary_first"
    ALL_PRIMARY = "all_primary"
    ALL = "all"
    BEST_S = "best_s"
    BEST_S_AND_Q = "best_s_and_q"


def _get_last_run_lines(lines: list[str]) -> list[str] | None:
    """
    Return lines from the last run (last 'Planned workflow:' onwards),
    or None if no such marker is found.
    """
    last_run_start = None
    for i, line in enumerate(lines):
        if line.startswith("Planned workflow:"):
            last_run_start = i

    if last_run_start is None:
        return None

    return lines[last_run_start:]


def _parse_log_dict(line_tail: str) -> dict:
    """Parse a dict literal from a log line, handling np.float64() formatting."""
    dict_str = re.sub(r"np\.float64\(([^)]+)\)", r"\1", line_tail)
    return ast.literal_eval(dict_str)


def get_results() -> pd.DataFrame:
    """
    Parse all log files and return every estimated binary parameter set found
    in the *last run* of each file.

    Returns
    -------
    pd.DataFrame with standard fit-parameter columns plus:
        solution_type : str   e.g. 'WidePlanet', 'CloseUpperPlanet',
                              'CloseLowerPlanet', or 'Unknown' (legacy format)
        is_alternate  : bool  True for s_dagger / alternate solutions
        idx           : int   zero-based light-curve index
    """
    results = []
    results_files = glob.glob(f"temp_output/{prefix}/W*/WFIRST.*.pkl")
    failed = []
    hm = []
    print("\nTotal Fits:", len(results_files), "\n")

    for results_file in results_files:
        lc_num = int(os.path.basename(results_file).split(".")[1])

        with open(results_file, "rb") as f:
            results_one = pickle.load(f)

        if (
            results_one["state"][
                "intermediate_results"
            ].estimate_binary_lens_parameters
            is None
        ):
            failed.append(lc_num)
            continue
        if (
            results_one["state"]["intermediate_results"].anomaly_type
            == "high_mag"
        ):
            hm.append(lc_num)
            continue

        lc_params = []
        current_solution_type = None
        for key, value in results_one["state"][
            "intermediate_results"
        ].estimate_binary_lens_parameters.items():
            params = value.ulens.copy()
            params["idx"] = lc_num - 1
            params["solution_type"] = key
            if key.endswith("_alt"):
                params["is_alternate"] = True
            else:
                params["is_alternate"] = False
            params["is_inverse"] = False
            params["t_0"] -= 2458234.0
            params["alpha"] = params["alpha"] % 360.0
            lc_params.append(params)

        if lc_params:
            results.extend(lc_params)
        else:
            failed.append(lc_num)

    print("\nclassified as hm:", sorted(hm))
    print("Total: ", len(hm), "\n")
    print(
        "\nestimate_binary_lens_parameters failed: ",
        sorted(failed),
        "\nTotal: ",
        len(failed),
        "\n",
    )

    return pd.DataFrame(results)


def get_ef_grid_results() -> pd.DataFrame:
    """
    Parse all log files and return the best EF grid point from the last run
    of each file.

    Returns
    -------
    pd.DataFrame with columns:
        lc_num : int    one-based light-curve number
        t_0    : float  adjusted by -2458234. for consistency with get_results()
        t_eff  : float
        j      : int
        chi2   : float
    """
    results = []
    missing = []
    log_files = glob.glob(f"temp_output/{prefix}/W*/WFIRST.*.log")

    for log_file in log_files:
        lc_num = int(os.path.basename(log_file).split(".")[1])

        with open(log_file) as f:
            lines = f.readlines()

        run_lines = _get_last_run_lines(lines)
        if run_lines is None:
            missing.append(lc_num)
            continue

        for line in run_lines:
            if line.startswith("Best EF grid point:"):
                params = _parse_log_dict(line.split(": ", 1)[1])
                params["lc_num"] = lc_num
                params["t_0"] -= 2458234.0
                results.append(params)
                break
        else:
            missing.append(lc_num)

    print(f"\nEF grid results found: {len(results)}")
    print(f"EF grid results missing: {sorted(missing)}\n")

    return pd.DataFrame(results)


class EvaluateResults:
    eval_columns = [
        "idx",
        "u0_true",
        "u_0",
        "tE_true",
        "t_E",
        "s_true",
        "s",
        "q_true",
        "q",
    ]
    print_columns = [
        "lc_num",
        "solution_type",
        "is_alternate",
        "t_0",
        "t0_true",
        "u_0",
        "u0_true",
        "t_E",
        "tE_true",
        "rho",
        "rhos_true",
        "s",
        "s_true",
        "q",
        "q_true",
        "alpha",
        "alpha_true",
    ]

    def __init__(
        self, strategy: SelectionStrategy = SelectionStrategy.ALL_PRIMARY
    ):
        self.prefix = prefix
        raw = get_results()
        truth = DC18Answers()

        # Merge truth once onto the full set; pandas fans it out to every row
        # sharing the same idx (many-to-one merge).
        self.all_results = raw.merge(
            truth.data.add_suffix("_true"),
            left_on="idx",
            right_index=True,
            how="left",
        )
        self.all_results["lc_num"] = self.all_results["idx"] + 1

        self.strategy = strategy
        self.results = self._apply_strategy(strategy)
        self._print_summary()

    # ------------------------------------------------------------------
    # Strategy management
    # ------------------------------------------------------------------

    def _apply_strategy(self, strategy: SelectionStrategy) -> pd.DataFrame:
        """Return a subset of all_results according to the given strategy."""
        if strategy == SelectionStrategy.ALL:
            return self.all_results.copy()

        elif strategy == SelectionStrategy.ALL_PRIMARY:
            return self.all_results[
                ~self.all_results["is_alternate"]
            ].reset_index(drop=True)

        elif strategy == SelectionStrategy.PRIMARY_FIRST:
            # drop_duplicates preserves insertion order, so the first primary
            # solution per LC (= the classified anomaly type) is kept.
            primary = self.all_results[~self.all_results["is_alternate"]]
            return primary.drop_duplicates(
                subset="idx", keep="first"
            ).reset_index(drop=True)
        elif strategy == SelectionStrategy.BEST_S:
            self.all_results["delta_s"] = np.abs(
                self.all_results["s_true"] - self.all_results["s"]
            )
            self.all_results["delta_q"] = np.abs(
                self.all_results["q_true"] - self.all_results["q"]
            )
            # self.all_results['delta_s'].min(groupby='lc_num')
            idx = []
            for lc_num in self.all_results["lc_num"].unique():
                idx.append(
                    self.all_results[self.all_results["lc_num"] == lc_num]
                    .sort_values(by="delta_s")
                    .index[0]
                )

            self.all_results = self.all_results.iloc[idx]
            return self.all_results.drop(columns="delta_s").reset_index()
        elif strategy == SelectionStrategy.BEST_S_AND_Q:
            self.all_results["delta_s"] = np.abs(
                self.all_results["s_true"] - self.all_results["s"]
            )
            self.all_results["delta_q"] = np.abs(
                self.all_results["q_true"] - self.all_results["q"]
            )
            self.all_results["delta_sq"] = np.sqrt(
                self.all_results["delta_s"] ** 2
                + self.all_results["delta_q"] ** 2
            )
            # self.all_results['delta_s'].min(groupby='lc_num')
            idx = []
            for lc_num in self.all_results["lc_num"].unique():
                idx.append(
                    self.all_results[self.all_results["lc_num"] == lc_num]
                    .sort_values(by="delta_sq")
                    .index[0]
                )

            self.all_results = self.all_results.iloc[idx]
            return self.all_results.drop(columns="delta_s").reset_index()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def set_strategy(self, strategy: SelectionStrategy):
        """Switch to a different selection strategy and refresh self.results."""
        self.strategy = strategy
        self.results = self._apply_strategy(strategy)
        self._print_summary()

    def _print_summary(self):
        with pd.option_context(
            "display.width", None, "display.max_rows", None
        ):
            print(
                self.results[self.print_columns].sort_values(
                    ["lc_num", "is_alternate", "solution_type"]
                )
            )
        print(f"\nStrategy  : {self.strategy.value}")
        print(f"Rows      : {len(self.results)}")
        print(f"Unique LCs: {self.results['idx'].nunique()}\n")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_truth_key(self, key):
        root = "rhos" if key == "rho" else key.replace("_", "")
        return root + "_true"

    def _plot(self, x, y, ylim=None, log=False):
        plt.figure()
        colors = np.where(self.results["s_true"] < 1, "darkorange", "darkcyan")
        facecolors = np.where(self.results["q_true"] < 0.03, colors, "none")
        low_mag = self.results["u0_true"].abs() > 0.05

        for mask, marker in zip([low_mag, ~low_mag], ["o", "d"]):
            if marker == "o":
                label = "u0_true > 0.05"
            else:
                label = "u0_true <= 0.05"
            if colors[mask][0] == "darkorange":
                label += " & s_true < 1"
            else:
                label += " & s_true > 1"
            plt.scatter(
                x[mask],
                y[mask],
                c=colors[mask],
                facecolors=facecolors[mask],
                marker=marker,
                clip_on=False,
                label=label,
                zorder=3,
            )

        if log and ylim is not None:
            for tri_y, tri_mask, tri_marker in [
                (ylim[1], y > ylim[1], "^"),
                (ylim[0], y < ylim[0], "v"),
            ]:
                if tri_mask.any():
                    plt.scatter(
                        x[tri_mask],
                        np.full(tri_mask.sum(), tri_y),
                        marker=tri_marker,
                        c=colors[tri_mask],
                        facecolors=facecolors[tri_mask],
                        clip_on=False,
                        zorder=3,
                    )

    # ------------------------------------------------------------------
    # Evaluation methods
    # ------------------------------------------------------------------

    def print_indices(self, good: pd.Series):
        """
        Print LC numbers categorised by the boolean mask *good*.

        With multi-row strategies (ALL_PRIMARY, ALL) a single LC can appear in
        both lists if some of its solutions are good and others are not.
        """
        good_lcs = sorted(self.results[good]["lc_num"].unique())
        bad_lcs = sorted(self.results[~good]["lc_num"].unique())
        print(f"  good ({len(good_lcs)} LCs): {good_lcs}")
        print(f"  bad  ({len(bad_lcs)} LCs): {bad_lcs}")

        if self.strategy != SelectionStrategy.PRIMARY_FIRST:
            overlap = sorted(set(good_lcs) & set(bad_lcs))
            if overlap:
                print(
                    f"  Note: {len(overlap)} LC(s) appear in both lists "
                    f"(mixed solutions): {overlap}"
                )

    def get_good_t0(self):
        delta_t0 = np.abs(self.results["t_0"] - self.results["t0_true"])
        good_t0 = delta_t0 < self.results["tE_true"] * 0.1
        msg = f"\nt0 is good (within 0.1tE): {np.sum(good_t0)}"
        return good_t0, msg

    def get_good_PSPL(self):
        good_t0, _ = self.get_good_t0()

        delta_u0 = np.abs(
            (self.results["u_0"] - self.results["u0_true"])
            / self.results["u0_true"]
        )
        good_u0 = delta_u0 < 0.5

        delta_tE = np.abs(
            (self.results["t_E"] - self.results["tE_true"])
            / self.results["tE_true"]
        )
        good_tE = delta_tE < 0.5
        good = good_t0 & good_u0 & good_tE
        msg = f"\nPSPL is good (t0 w/in 0.1tE, u0 50%, tE 50%): {np.sum(good)}"

        return good, msg

    def is_the_pspl_fit_good(self):
        good_t0, msg = self.get_good_t0()
        print(msg)
        self.print_indices(good_t0)

        good, msg = self.get_good_PSPL()
        print(msg)
        self.print_indices(good)

    def is_log_q_good(self, threshold=1.0):
        delta = np.log10(self.results["q"]) - np.log10(self.results["q_true"])
        good = np.abs(delta) < threshold
        print(
            f"\n|dlog q| < {threshold}: "
            f"{good.sum()} rows, {self.results[good]['idx'].nunique()} unique LCs"
        )
        self.print_indices(good)

    def is_sign_s_good(self):
        self.results["sign_s_good"] = (
            np.log10(self.results["s"]) * np.log10(self.results["s_true"])
            > 0.0
        )
        self.results["s_neg"] = 1.0 / self.results["s"]
        good = (self.results["s_true"] > 1.0) & (
            self.results["solution_type"].str.match("ClosePlanet")
        )
        print(np.sum(good))
        with pd.option_context(
            "display.max_rows", None, "display.width", None
        ):
            print(
                self.results[
                    [
                        "lc_num",
                        "solution_type",
                        "is_alternate",
                        "is_inverse",
                        "s_true",
                        "s",
                        "sign_s_good",
                        "u0_true",
                        "tE_true",
                    ]
                ].sort_values(
                    by=["s_true", "lc_num", "solution_type", "is_alternate"]
                )
            )
            print(
                self.results[good][
                    [
                        "lc_num",
                        "solution_type",
                        "is_alternate",
                        "is_inverse",
                        "s_true",
                        "s",
                        "s_neg",
                        "sign_s_good",
                        "u0_true",
                        "tE_true",
                    ]
                ].sort_values(by=["lc_num"])
            )

    def is_the_planet_good(self, log_q_threshold=0.5):
        delta = np.log10(self.results["q"]) - np.log10(self.results["q_true"])
        good = (np.abs(delta) < log_q_threshold) & (
            (self.results["s"] - 1) * (self.results["s_true"] - 1) > 0
        )
        print(
            f"\n|dlog q| < {log_q_threshold} AND close/wide correct: "
            f"{good.sum()} rows, {self.results[good]['idx'].nunique()} unique LCs"
        )
        self.print_indices(good)

    def is_there_a_good_guess(self):
        self.results["sol_dist"] = np.sqrt(
            (np.log10(self.results["s_true"]) - np.log10(self.results["s"]))
            ** 2
            + (np.log10(self.results["q_true"]) - np.log10(self.results["q"]))
            ** 2
        )
        # self.results["sol_dist"] = np.abs(
        #     self.results["s_true"] - self.results["s"]
        # )
        # good = self.results.groupby('lc_num').min('sol_dist')
        good = self.results["sol_dist"] < 0.1
        self.print_indices(good)
        # with pd.option_context('display.max_rows', None, 'display.width', None):
        #    print(self.results[['sol_dist'] + self.print_columns].sort_values(by=['lc_num', 'sol_dist']))

        # ['lc_num', 'solution_type', 'is_alternate', 'is_inverse', 'dist', 's_true', 's', 'q_true', 'q',
        #   'u0_true', 'tE_true']])

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _make_scatter_plot(self, key, log=False):
        truth_key = self._get_truth_key(key)
        fit_value = self.results[key]
        true_value = self.results[truth_key]

        if key == "u_0" and log:
            fit_value = np.abs(fit_value)
            true_value = np.abs(true_value)

        if log and key == "q":
            delta = np.log10(fit_value) - np.log10(true_value)
            value = np.log10(true_value)
            xlabel = f"log ({key}_true)"
            ylabel = f"log({key}) - log(True)"
            ylim = (-6, -1)
        if log and key == "rho":
            delta = np.log10(fit_value) - np.log10(true_value)
            value = np.log10(true_value)
            xlabel = f"log ({key}_true)"
            ylabel = f"log({key}) - log(True)"
            ylim = (-4, 1)
        else:
            value = true_value
            delta = fit_value - true_value
            xlabel = f"{key}_true"
            ylabel = f"({key} - True)"
            ylim = None

        self._plot(value, delta, ylim=ylim, log=log)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.ylim(ylim)
        plt.xlim(ylim)
        plt.gca().set_aspect(1)
        plt.minorticks_on()
        plt.legend(loc="upper left", frameon=False)
        plt.tight_layout()

    def make_all_delta_plots(self):
        plot_list = {
            "u_0": True,
            "t_E": False,
            "rho": True,
            "alpha": False,
            "s": True,
            "q": True,
        }
        for key, log in plot_list.items():
            self._make_scatter_plot(key, log=log)  # _plot() creates the figure
            plt.savefig(
                f"temp_output/{self.prefix}/figs/{key}_deltas.png", dpi=300
            )

    def make_scatter_plot(self, key, log=False):
        fit_value = self.results[key]
        true_value = self.results[self._get_truth_key(key)]

        self._plot(true_value, fit_value)  # _plot() creates the figure

        ax = plt.gca()
        ax.set_aspect("equal")
        lim = (
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
        )
        ax.plot(lim, lim, zorder=0, color="black", clip_on=True)

        if log:
            plt.xlabel(f"log {key} (True)")
            plt.xscale("log")
            plt.ylabel(f"log {key} (Fitted)")
            plt.yscale("log")
        else:
            plt.xlabel(f"{key} (True)")
            plt.ylabel(f"{key} (Fitted)")
        plt.gca().set_aspect(1)
        plt.minorticks_on()
        plt.legend(loc="upper left", frameon=False)
        plt.tight_layout()

    def make_all_scatter_plots(self):
        for key in ["u_0", "t_E", "rho", "alpha", "s", "q"]:
            self.make_scatter_plot(key, log=(key in ("q", "rho")))
            if not os.path.exists(f"temp_output/{self.prefix}/figs/"):
                os.makedirs(f"temp_output/{self.prefix}/figs/")
            plt.savefig(
                f"temp_output/{self.prefix}/figs/{key}_vs.png", dpi=300
            )

    def make_planet_plots(self):
        keys = ["s", "q"]
        deltas = {}
        for key in keys:
            truth_key = self._get_truth_key(key)
            fit_value = self.results[key]
            true_value = self.results[truth_key]
            delta = np.log10(fit_value) - np.log10(true_value)
            deltas[key] = delta

        self._plot(deltas["s"], deltas["q"])
        plt.xlabel("log (s_fitted) - log (s_true)")
        plt.ylabel("log (q_fitted) - log (q_true)")
        plt.xlim(-4, 4)
        plt.ylim(-4, 3)
        plt.minorticks_on()
        plt.gca().set_aspect(1)
        plt.legend(loc="upper left", frameon=False)
        plt.tight_layout()
        plt.title(
            f"strategy: {self.strategy.value} ({len(self.results)} rows) prefix: {self.prefix}"
        )
        plt.savefig(
            f"temp_output/{self.prefix}/figs/planet_deltas.png", dpi=300
        )


def check_bad_t0(evaluator):
    ef_results = (
        get_ef_grid_results()
        .sort_values(by="lc_num")
        .rename(columns={"t_0": "t0_ef"})
    )
    with pd.option_context("display.width", None, "display.max_rows", None):
        print(ef_results.sort_values(by="t_eff"))

    evaluator.results = pd.merge(evaluator.results, ef_results, on="lc_num")
    good_t0, _ = evaluator.get_good_t0()
    with pd.option_context("display.width", None, "display.max_rows", None):
        print(
            evaluator.results[~good_t0][
                evaluator.print_columns + ["t_eff"]
            ].sort_values(by="lc_num")
        )


if __name__ == "__main__":
    for p in prefixs:
        prefix = p

        print("\nPrefix:", prefix)
        truth = [
            "lc_num",
            "u0_true",
            "tE_true",
            "s_true",
            "q_true",
            "rhos_true",
        ]
        evaluator = EvaluateResults(strategy=SelectionStrategy.BEST_S_AND_Q)
        with pd.option_context(
            "display.width", None, "display.max_rows", None
        ):
            print(evaluator.results[truth].sort_values(by="lc_num"))
        check_bad_t0(evaluator)

        evaluator.is_there_a_good_guess()
        evaluator.is_the_pspl_fit_good()
        evaluator.is_log_q_good()
        evaluator.is_sign_s_good()
        evaluator.is_the_planet_good()
        evaluator.make_all_scatter_plots()
        evaluator.make_all_delta_plots()

        evaluator.make_planet_plots()
        # plt.show()

        # Switch strategies at any time without re-parsing:
        # evaluator.set_strategy(SelectionStrategy.PRIMARY_FIRST)
        # evaluator.is_the_planet_good()
