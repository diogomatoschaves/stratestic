import builtins
import logging

import dill
import matplotlib.pyplot as plt
import plotly
import pytest


@pytest.fixture
def mocked_plotly_figure_show(mocker):
    mocker.patch.object(plotly.graph_objs.Figure, "show", lambda self: None)


@pytest.fixture
def mocked_matplotlib_show(mocker):
    return mocker.patch.object(plt, "show", lambda: None)


@pytest.fixture
def spy_logging_info(mocker):
    return mocker.spy(logging, "info")


@pytest.fixture
def mocked_dill_dump(mocker):
    return mocker.patch.object(dill, 'dump', lambda obj, file: None)


@pytest.fixture
def mocked_builtin_open(mocker):
    return mocker.patch.object(builtins, 'open', lambda filename, file_type: None)
