import pytest

from modelkit.utils.serialization import safe_np_dump

np = pytest.importorskip("numpy")


@pytest.mark.parametrize(
    "value, result",
    [
        (np.arange(4), [0, 1, 2, 3]),
        (np.zeros((1,))[0], 0),
        (np.zeros((1,), dtype=int)[0], 0),
        (1, 1),
    ],
)