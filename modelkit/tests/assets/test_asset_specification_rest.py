import pytest

from modelkit.assets import errors
from modelkit.assets.settings import AssetSpec




@pytest.mark.parametrize(
    "test, valid",
    [
        ("_", False),
        ("a_", False),
        ("", False),
        ("o", True),
        ("1", True),
        ("some_go0d_name", True),
        ("some_go/0d_name", True),
        ("SOME_GOOD_NAME_AS_WELL", True),
        ("50M3_G00D_N4ME_4S_W3LL", True),
        ("C:\\A\\L0cAL\\Windows\\file.ext", True),
    ],
)


@pytest.mark.parametrize(
    "test, valid",
    [
        ("_", True),
        ("a_", True),
        ("a.a", True),
        ("1.a", True),
        ("a.1", True),
        (".1", True),
        ("12.", True),
        ("", True),
        ("1", True),
        ("12", True),
        ("12.1", True),
        ("12/1", False),
        ("12\1", False),
    ],
)