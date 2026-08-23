#!/usr/bin/env python
"""Run reproducible statistical tests for HiArch summary tables.

The input must be a delimited table with a header.  Use ``correlation`` for
Pearson or Spearman tests and ``groups`` for tests on a numeric value across
groups.  Results are written as a tab-separated table for direct inclusion in
analysis records.

Examples
--------
python scripts/statistical_analysis.py correlation scores.tsv \
    --x GFS --y CBS --method spearman --output gfs_cbs_spearman.tsv

python scripts/statistical_analysis.py groups scores.tsv \
    --value GFS --group clade --test anova --control Arthropoda \
    --posthoc control-bonferroni --output gfs_by_clade.tsv

python scripts/statistical_analysis.py groups scores.tsv \
    --value CBS --group kingdom --test kruskal \
    --posthoc pairwise-mannwhitney-bonferroni --output cbs_by_kingdom.tsv

Exact Dunnett testing requires ``scipy.stats.dunnett`` (SciPy >= 1.11).
The project pins SciPy 1.11.4; the runtime check prevents an incorrect result
if the script is used in a different environment.
"""

import argparse
import itertools
import sys

import numpy as np
import pandas as pd
import scipy
from scipy import stats


RESULT_COLUMNS = [
    "analysis",
    "test",
    "group_1",
    "group_2",
    "n_1",
    "n_2",
    "statistic",
    "p_value",
    "p_value_adjusted",
    "reject_null_0_05",
]


def read_table(path):
    """Read CSV/TSV/whitespace-delimited input and reject an empty table."""
    table = pd.read_csv(path, sep=None, engine="python")
    if table.empty:
        raise ValueError("Input table has no rows: {}".format(path))
    return table


def numeric_pair(table, first, second):
    missing = [column for column in (first, second) if column not in table]
    if missing:
        raise ValueError("Missing column(s): {}".format(", ".join(missing)))
    data = table[[first, second]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(data) < 3:
        raise ValueError("At least three complete numeric pairs are required.")
    return data


def correlation_results(table, first, second, method):
    data = numeric_pair(table, first, second)
    if method == "pearson":
        statistic, p_value = stats.pearsonr(data[first], data[second])
    else:
        statistic, p_value = stats.spearmanr(data[first], data[second])
    return [{
        "analysis": "correlation",
        "test": method,
        "group_1": first,
        "group_2": second,
        "n_1": len(data),
        "n_2": np.nan,
        "statistic": statistic,
        "p_value": p_value,
        "p_value_adjusted": p_value,
        "reject_null_0_05": bool(p_value < 0.05),
    }]


def grouped_values(table, value_column, group_column):
    missing = [column for column in (value_column, group_column) if column not in table]
    if missing:
        raise ValueError("Missing column(s): {}".format(", ".join(missing)))
    data = table[[value_column, group_column]].copy()
    data[value_column] = pd.to_numeric(data[value_column], errors="coerce")
    data = data.dropna()
    groups = {}
    for group_name, values in data.groupby(group_column)[value_column]:
        values = values.to_numpy(dtype=float)
        if len(values) >= 2:
            groups[str(group_name)] = values
    if len(groups) < 2:
        raise ValueError("At least two groups with two numeric observations are required.")
    return groups


def bonferroni(p_values):
    count = len(p_values)
    return [min(1.0, float(p_value) * count) for p_value in p_values]


def omnibus_result(groups, test):
    ordered_values = [groups[name] for name in sorted(groups)]
    if test == "anova":
        statistic, p_value = stats.f_oneway(*ordered_values)
    else:
        statistic, p_value = stats.kruskal(*ordered_values)
    return {
        "analysis": "groups",
        "test": test,
        "group_1": "all",
        "group_2": "all",
        "n_1": int(sum(len(values) for values in ordered_values)),
        "n_2": len(ordered_values),
        "statistic": statistic,
        "p_value": p_value,
        "p_value_adjusted": p_value,
        "reject_null_0_05": bool(p_value < 0.05),
    }


def pairwise_results(groups, mode, control=None):
    names = sorted(groups)
    if mode == "control-bonferroni":
        if control is None:
            raise ValueError("--control is required for control-bonferroni.")
        if control not in groups:
            raise ValueError("Control group is absent after filtering: {}".format(control))
        pairs = [(control, name) for name in names if name != control]
    else:
        pairs = list(itertools.combinations(names, 2))

    unadjusted = []
    result_rows = []
    for first, second in pairs:
        if mode == "control-bonferroni":
            statistic, p_value = stats.ttest_ind(
                groups[first], groups[second], equal_var=False, nan_policy="omit"
            )
            test_name = "welch_t_control_bonferroni"
        else:
            statistic, p_value = stats.mannwhitneyu(
                groups[first], groups[second], alternative="two-sided"
            )
            test_name = "mannwhitney_bonferroni"
        unadjusted.append(p_value)
        result_rows.append({
            "analysis": "groups",
            "test": test_name,
            "group_1": first,
            "group_2": second,
            "n_1": len(groups[first]),
            "n_2": len(groups[second]),
            "statistic": statistic,
            "p_value": p_value,
        })

    for row, adjusted in zip(result_rows, bonferroni(unadjusted)):
        row["p_value_adjusted"] = adjusted
        row["reject_null_0_05"] = bool(adjusted < 0.05)
    return result_rows


def dunnett_results(groups, control):
    """Run the exact SciPy Dunnett test when the installed SciPy supports it."""
    if not hasattr(stats, "dunnett"):
        raise RuntimeError(
            "Exact Dunnett testing requires scipy.stats.dunnett (SciPy >= 1.11). "
            "The current scipy=={} does not provide it.".format(scipy.__version__)
        )
    if control is None:
        raise ValueError("--control is required for dunnett.")
    if control not in groups:
        raise ValueError("Control group is absent after filtering: {}".format(control))

    treatments = [name for name in sorted(groups) if name != control]
    result = stats.dunnett(*[groups[name] for name in treatments], control=groups[control])
    rows = []
    for treatment, statistic, p_value in zip(treatments, result.statistic, result.pvalue):
        rows.append({
            "analysis": "groups",
            "test": "dunnett",
            "group_1": control,
            "group_2": treatment,
            "n_1": len(groups[control]),
            "n_2": len(groups[treatment]),
            "statistic": statistic,
            "p_value": p_value,
            "p_value_adjusted": p_value,
            "reject_null_0_05": bool(p_value < 0.05),
        })
    return rows


def group_results(table, value_column, group_column, test, posthoc, control):
    groups = grouped_values(table, value_column, group_column)
    rows = [omnibus_result(groups, test)]
    if posthoc != "none":
        if posthoc == "dunnett":
            if test != "anova":
                raise ValueError("dunnett is only valid after --test anova.")
            rows.extend(dunnett_results(groups, control))
            return rows
        if posthoc == "control-bonferroni" and test != "anova":
            raise ValueError("control-bonferroni is only valid after --test anova.")
        if posthoc == "pairwise-mannwhitney-bonferroni" and test != "kruskal":
            raise ValueError(
                "pairwise-mannwhitney-bonferroni is only valid after --test kruskal."
            )
        rows.extend(pairwise_results(groups, posthoc, control))
    return rows


def write_results(rows, output):
    result = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    result.to_csv(output, sep="\t", index=False, float_format="%.8g")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="analysis", required=True)

    correlation = subparsers.add_parser("correlation")
    correlation.add_argument("input")
    correlation.add_argument("--x", required=True, help="First numeric column.")
    correlation.add_argument("--y", required=True, help="Second numeric column.")
    correlation.add_argument("--method", choices=("pearson", "spearman"), required=True)
    correlation.add_argument("--output", required=True)

    groups = subparsers.add_parser("groups")
    groups.add_argument("input")
    groups.add_argument("--value", required=True, help="Numeric response column.")
    groups.add_argument("--group", required=True, help="Categorical group column.")
    groups.add_argument("--test", choices=("anova", "kruskal"), required=True)
    groups.add_argument(
        "--posthoc",
        choices=("none", "dunnett", "control-bonferroni", "pairwise-mannwhitney-bonferroni"),
        default="none",
    )
    groups.add_argument("--control", help="Required only by control-bonferroni.")
    groups.add_argument("--output", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    table = read_table(args.input)
    if args.analysis == "correlation":
        rows = correlation_results(table, args.x, args.y, args.method)
    else:
        rows = group_results(
            table, args.value, args.group, args.test, args.posthoc, args.control
        )
    write_results(rows, args.output)


if __name__ == '__main__':
    try:
        main()
    except (OSError, RuntimeError, ValueError, pd.errors.ParserError) as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        sys.exit(2)
