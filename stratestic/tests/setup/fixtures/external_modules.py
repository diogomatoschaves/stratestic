import matplotlib
import pytest


@pytest.fixture
def mocked_matplotlib_show(mocker):
    mocker.patch.object(matplotlib.pyplot, "show", lambda: None)
