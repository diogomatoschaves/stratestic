import matplotlib
import plotly
import pytest


@pytest.fixture
def mocked_plotly_figure_show(mocker):
    mocker.patch.object(plotly.graph_objs.Figure, "show", lambda self: None)
