"""Regenerate expected-output fixture files under tests/backtesting/.

Each out file is rebuilt by running the backtest exactly as the corresponding
test does (same sample data, same mocks, same optimizer arguments), so it must
only be used when the current behavior is known to be correct.

Usage:
    python scripts/regenerate_fixtures.py <suite> [fixture_name ...]
    python scripts/regenerate_fixtures.py --all

Suites: vectorized, iterative (in/out), vectorized-margin, iterative-margin
(in_margin/out_margin, leveraged), combining (in/out).
With no fixture names, every fixture of the suite is regenerated.
"""
import argparse
import os
import sys
from unittest.mock import patch

# Tests run with the numba JIT disabled (pytest.ini); fixtures must be
# generated under the same setting to be bit-for-bit comparable.
os.environ["NUMBA_DISABLE_JIT"] = "1"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np

from stratestic.backtesting import VectorizedBacktester, IterativeBacktester
from stratestic.backtesting.combining import StrategyCombiner
from tests.setup.test_data.sample_data import data as sample_data

SUITES = {
    "vectorized": VectorizedBacktester,
    "iterative": IterativeBacktester,
    "vectorized-margin": VectorizedBacktester,
    "iterative-margin": IterativeBacktester,
    "combining": None,
    "strategies": None,
}

# suite -> (path under tests/, in dir, out dir)
SUITE_DIRS = {
    "vectorized": ("backtesting/vectorized", "in", "out"),
    "iterative": ("backtesting/iterative", "in", "out"),
    "vectorized-margin": ("backtesting/vectorized", "in_margin", "out_margin"),
    "iterative-margin": ("backtesting/iterative", "in_margin", "out_margin"),
    "combining": ("backtesting/combining", "in", "out"),
    "strategies": ("strategies", "in", "out"),
}

# Mirror the test parametrizations
GEN_ALG_ARGS = {"pop_size": 4, "max_gen": 1, "random_state": 42}
MARGIN_LEVERAGES = {"vectorized-margin": [1, 10, 100], "iterative-margin": [1, 10]}
MARGIN_AMOUNT = 1000
MARGIN_SYMBOL = "BTCUSDT"
COMBINING_TRADING_COSTS = 0.1


def load_in_fixture(path):
    variables = {}
    with open(path) as f:
        exec(f.read(), variables)
    return variables


def fmt_value(value):
    if isinstance(value, float):
        if np.isnan(value):
            return "np.nan"
        if np.isinf(value):
            return "np.inf" if value > 0 else "-np.inf"
    return repr(value)


def fmt_obj(obj):
    if isinstance(obj, dict):
        return "{" + ", ".join(f"{k!r}: {fmt_obj(v)}" for k, v in obj.items()) + "}"
    if isinstance(obj, (tuple, list)):
        inner = ", ".join(fmt_obj(v) for v in obj)
        if isinstance(obj, tuple):
            return f"({inner},)" if len(obj) == 1 else f"({inner})"
        return f"[{inner}]"
    return fmt_value(obj)


def fmt_records(records):
    lines = ["expected_results = ["]
    for record in records:
        lines.append("    {")
        for key, value in record.items():
            lines.append(f'        "{key}": {fmt_value(value)},')
        lines.append("    },")
    lines.append("]")
    return "\n".join(lines)


def test_mocks():
    """The same side-effect mocks the tests use (no plots, no model dumps)."""
    import plotly.graph_objs
    import matplotlib.pyplot as plt
    import dill
    import builtins
    from contextlib import ExitStack

    stack = ExitStack()
    stack.enter_context(patch.object(plotly.graph_objs.Figure, "show", lambda self: None))
    stack.enter_context(patch.object(plt, "show", lambda: None))
    stack.enter_context(patch.object(dill, "dump", lambda obj, file: None))
    stack.enter_context(patch.object(builtins, "open", lambda filename, file_type: None))
    return stack


def quiet_run(backtester, method, *args, **kwargs):
    with test_mocks():
        return getattr(backtester, method)(*args, **kwargs)


def regenerate(suite, name):
    backtester_class = SUITES[suite]
    suite_dir, in_dir, out_dir = SUITE_DIRS[suite]
    in_path = os.path.join(REPO_ROOT, "tests", suite_dir, in_dir, f"{name}.py")
    out_path = os.path.join(REPO_ROOT, "tests", suite_dir, out_dir, f"{name}.py")

    fixture = load_in_fixture(in_path)

    if suite == "combining":
        regenerate_combining(fixture, out_path)
        return

    if suite == "strategies":
        regenerate_strategies(fixture, out_path)
        return

    if suite.endswith("-margin"):
        regenerate_margin(suite, fixture, backtester_class, out_path)
        return
    strategy = fixture["strategy"]
    params = fixture["params"]
    trading_costs = fixture["trading_costs"]
    optimization_params = fixture["optimization_params"]

    test_data = sample_data.set_index("open_time")

    def new_backtester():
        return backtester_class(strategy(**params, data=test_data), trading_costs=trading_costs)

    backtester = new_backtester()
    quiet_run(backtester, "run")
    records = backtester.processed_data.to_dict(orient="records")
    performance, outperformance = backtester.perf, backtester.outperf

    if suite == "iterative":
        optimization_results = {
            "brute_force": quiet_run(new_backtester(), "optimize", *optimization_params),
            "gen_alg": quiet_run(
                new_backtester(), "optimize", *optimization_params,
                optimizer="gen_alg", **GEN_ALG_ARGS
            ),
        }
        optimization_block = (
            "expected_optimization_results = {\n"
            + "".join(
                f"    \"{optimizer}\": {fmt_obj(result)},\n"
                for optimizer, result in optimization_results.items()
            )
            + "}"
        )
    else:
        optimization_results = quiet_run(new_backtester(), "optimize", *optimization_params)
        optimization_block = f"expected_optimization_results = {fmt_obj(optimization_results)}"

    content = "\n".join([
        "import numpy as np",
        "from pandas import Timestamp",
        "",
        f"expected_performance = {fmt_value(round(performance, 6))}",
        f"expected_outperformance = {fmt_value(round(outperformance, 6))}",
        "",
        optimization_block,
        "",
        fmt_records(records),
        "",
    ])

    with open(out_path, "w") as f:
        f.write(content)

    print(f"regenerated {os.path.relpath(out_path, REPO_ROOT)}")


def regenerate_margin(suite, fixture, backtester_class, out_path):
    """Mirrors tests/backtesting/*/test_*_with_margin.py::test_run."""
    strategy = fixture["strategy"]
    params = fixture["params"]
    trading_costs = fixture["trading_costs"]

    test_data = sample_data.set_index("open_time")

    records_per_leverage = {}
    performance = outperformance = None

    for leverage in MARGIN_LEVERAGES[suite]:
        backtester = backtester_class(
            strategy(**params, data=test_data),
            symbol=MARGIN_SYMBOL,
            amount=MARGIN_AMOUNT,
            trading_costs=trading_costs,
            leverage=leverage,
        )
        quiet_run(backtester, "run")
        records_per_leverage[leverage] = backtester.processed_data.to_dict(orient="records")

        if leverage == 1:
            performance, outperformance = backtester.perf, backtester.outperf

    lines = [
        "import numpy as np",
        "from pandas import Timestamp",
        "",
        f"expected_performance = {fmt_value(round(performance, 6))}",
        f"expected_outperformance = {fmt_value(round(outperformance, 6))}",
        "",
        "expected_results = {",
    ]
    for leverage, records in records_per_leverage.items():
        lines.append(f"    {leverage}: [")
        for record in records:
            lines.append("        {")
            for key, value in record.items():
                lines.append(f'            "{key}": {fmt_value(value)},')
            lines.append("        },")
        lines.append("    ],")
    lines.append("}")
    lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))

    print(f"regenerated {os.path.relpath(out_path, REPO_ROOT)}")


def fmt_dataframe(name, df):
    lines = [f"{name} = pd.DataFrame(", "    ["]
    for record in df.to_dict(orient="records"):
        lines.append("        {")
        for key, value in record.items():
            lines.append(f'            "{key}": {fmt_value(value)},')
        lines.append("        },")
    lines.append("    ],")

    try:
        index_repr = repr([int(i) for i in df.index])
    except (TypeError, ValueError):
        index_repr = "[" + ", ".join(fmt_value(i) for i in df.index) + "]"

    lines.append(f"    index={index_repr}")
    lines.append(")")
    return "\n".join(lines)


def regenerate_strategies(fixture, out_path):
    """Mirrors tests/strategies/test_strategies.py data/set_parameters/get_signal tests."""
    strategy = fixture["strategy"]
    params = fixture["params"]
    new_parameters = fixture["new_parameters"]

    with test_mocks():
        instance = strategy(**params, data=sample_data)
        expected_data = instance.data.copy()
        expected_signal = instance.get_signal()

        instance = strategy(**params, data=sample_data)
        instance.set_parameters(new_parameters)
        expected_data_set_parameters = instance.data.copy()

    content = "\n".join([
        "import numpy as np",
        "import pandas as pd",
        "from pandas import Timestamp",
        "",
        fmt_dataframe("expected_data", expected_data),
        "",
        fmt_dataframe("expected_data_set_parameters", expected_data_set_parameters),
        "",
        f"expected_signal = {fmt_value(expected_signal)}",
        "",
    ])

    with open(out_path, "w") as f:
        f.write(content)

    print(f"regenerated {os.path.relpath(out_path, REPO_ROOT)}")


def regenerate_combining(fixture, out_path):
    """Mirrors test_strategy_combiner.py::test_strategy_combiner_with_backtester."""
    strategies = fixture["strategies"]
    method = fixture["method"]
    backtesting_class = fixture["backtester"]

    test_data = sample_data.set_index("open_time")

    strategy_combiner = StrategyCombiner(strategies, method, data=test_data)
    backtester = backtesting_class(strategy_combiner, trading_costs=COMBINING_TRADING_COSTS)
    quiet_run(backtester, "run")

    records = backtester.processed_data.to_dict(orient="records")

    content = "\n".join([
        "import numpy as np",
        "from pandas import Timestamp",
        "",
        fmt_records(records),
        "",
    ])

    with open(out_path, "w") as f:
        f.write(content)

    print(f"regenerated {os.path.relpath(out_path, REPO_ROOT)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", nargs="?", choices=list(SUITES))
    parser.add_argument("fixtures", nargs="*")
    parser.add_argument("--all", action="store_true", help="regenerate every suite and fixture")
    args = parser.parse_args()

    if not args.all and not args.suite:
        parser.error("provide a suite (vectorized/iterative) or --all")

    suites = list(SUITES) if args.all else [args.suite]

    for suite in suites:
        names = args.fixtures
        if args.all or not names:
            suite_dir, in_dir, _ = SUITE_DIRS[suite]
            in_dir_path = os.path.join(REPO_ROOT, "tests", suite_dir, in_dir)
            names = sorted(
                f[:-3] for f in os.listdir(in_dir_path)
                if f.endswith(".py") and f != "__init__.py"
            )
        for name in names:
            regenerate(suite, name)


if __name__ == "__main__":
    main()
