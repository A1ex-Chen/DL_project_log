import os

import pytest

import tests
from modelkit.assets import errors
from modelkit.assets.settings import AssetSpec
from modelkit.assets.versioning.major_minor import (
    InvalidMajorVersionError,
    MajorMinorAssetsVersioningSystem,
    MajorVersionDoesNotExistError,
)

TEST_CASES_PARSE = [
    ("ok", False, None),
    ("1.", False, None),
    ("_", False, None),
    ("a_", False, None),
    ("a.a", False, None),
    ("1.a", False, None),
    ("a.1", False, None),
    (".1", False, None),
    ("12.", False, None),
    ("1/2", False, None),
    ("1\2", False, None),
    ("", True, None),
    (None, True, (None, None)),
    ("1", True, (1, None)),
    ("1.1", True, (1, 1)),
    ("10.1", True, (10, 1)),
    ("10.10", True, (10, 10)),
    ("123.4", True, (123, 4)),
]


@pytest.mark.parametrize("version, valid, values", TEST_CASES_PARSE)


@pytest.mark.parametrize("version, valid, values", TEST_CASES_PARSE)


TEST_CASES_INCREMENT = [
    (["0.0"], False, None, True, "0.1"),
    (["0.0"], False, "0", True, "0.1"),
    (["0.0"], True, None, True, "1.0"),
    (["0.9"], False, None, True, "0.10"),
    (["0.9", "1.0"], False, None, True, "1.1"),
    (["0.9", "1.0"], False, "0", True, "0.10"),
    (["9.0"], True, None, True, "10.0"),
    (["123.456"], False, None, True, "123.457"),
    (["123.456"], True, None, True, "124.0"),
    (["123.456"], True, "0", True, "124.0"),
    (["123.456"], False, "0", False, None),
]


@pytest.mark.parametrize(
    "string, bump_major, major, valid, result", TEST_CASES_INCREMENT
)


TEST_CASES_SORT = [
    (["0.0", "1.0", "2.0"], ["2.0", "1.0", "0.0"]),
    (["0.0", "2.0"], ["2.0", "0.0"]),
    (["0.0", "0.1", "0.2"], ["0.2", "0.1", "0.0"]),
    (["0.0", "0.1", "1.0", "1.1"], ["1.1", "1.0", "0.1", "0.0"]),
    (["0.2", "0.10", "2"], ["2", "0.10", "0.2"]),
]


@pytest.mark.parametrize("version_list, result", TEST_CASES_SORT)


TEST_CASES_FILTER = [
    (["0.0", "1.0", "2.0"], "0", True, ["0.0"]),
    (["0.0", "1.0", "2.0"], "ok", False, None),
    (
        ["123.0", "123.1", "123.2", "0.1", "1.2", "2.3", "3.4"],
        "123",
        True,
        ["123.0", "123.1", "123.2"],
    ),
]


@pytest.mark.parametrize("version_list, major, valid, result", TEST_CASES_FILTER)


TEST_CASES_LATEST = [
    (["0.0", "1.0", "2.0"], None, True, "2.0"),
    (["0.0", "1.0", "2.0"], "1", True, "1.0"),
    (["0.0", "1.0", "2.0"], "123", False, None),
    (["123.0", "123.1", "123.2", "0.1", "1.2", "2.3", "3.4"], "123", True, "123.2"),
]


@pytest.mark.parametrize("version_list, major, valid, result", TEST_CASES_LATEST)




@pytest.mark.parametrize(
    "version, bump_major, major",
    [
        ("1.0", True, "1"),
        ("2.1", False, "2"),
    ],
)


TEST_SPECS = [
    ({"name": "blebla/blabli"}, True),
    ({"name": "blabli"}, True),
    ({"name": "ontologies/skills.csv", "version": "1"}, True),
    ({"name": "ontologies/skills.csv", "version": "1.1"}, True),
    ({"name": "ontologies/skills.csv", "version": "1.1.1"}, False),
    ({"name": "ontologies/skills.csv", "version": "10"}, True),
    ({"name": "ontologies/skills.csv", "version": ".10"}, False),
    ({"name": "ontologies/skills.csv:", "version": "1"}, False),
]


@pytest.mark.parametrize("spec_dict, valid", TEST_SPECS)




TEST_CASES_SORT = [
    (["0.0", "1.0", "2.0"], ["2.0", "1.0", "0.0"]),
    (["0.0", "2.0"], ["2.0", "0.0"]),
    (["0.0", "0.1", "0.2"], ["0.2", "0.1", "0.0"]),
    (["0.0", "0.1", "1.0", "1.1"], ["1.1", "1.0", "0.1", "0.0"]),
    (["0.2", "0.10", "2"], ["2", "0.10", "0.2"]),
]


@pytest.mark.parametrize("version_list, result", TEST_CASES_SORT)




@pytest.mark.parametrize(
    "test, valid",
    [("_", False), ("a_", False), ("", True), (None, True), ("1", True), ("12", True)],
)




@pytest.mark.parametrize("s, spec", get_string_spec(["1", "1.2", "12"]))

