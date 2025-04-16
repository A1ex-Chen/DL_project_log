import json
import os
import time

import pytest
from test_major_minor_versioning import get_string_spec

import modelkit
import tests
from modelkit.assets import errors
from modelkit.assets.settings import AssetSpec
from modelkit.assets.versioning.simple_date import SimpleDateAssetsVersioningSystem


@pytest.mark.parametrize(
    "version,valid",
    [
        ("2021-11-15T17-30-56Z", True),
        ("0000-00-00T00-00-00Z", True),
        ("9999-99-99T99-99-99Z", True),
        ("2021-11-15T17-30-56", False),
        ("21-11-15T17-30-56Z", False),
        ("", False),
    ],
)


















@pytest.mark.parametrize("s, spec", get_string_spec(["2021-11-14T18-00-00Z"]))




    monkeypatch.setenv("MODELKIT_ASSETS_DIR", working_dir)
    monkeypatch.setenv("MODELKIT_ASSETS_VERSIONING_SYSTEM", "simple_date")
    monkeypatch.setenv("MODELKIT_STORAGE_PROVIDER", "local")
    monkeypatch.setenv(
        "MODELKIT_STORAGE_BUCKET",
        os.path.join(tests.TEST_DIR, "testdata", "test-bucket"),
    )
    monkeypatch.setenv("MODELKIT_STORAGE_PREFIX", "assets-prefix")

    model = modelkit.load_model("my_model", models=MyModel)
    assert model.predict({}) == "asset-2021-11-14T18-00-00Z"

    my_last_model = modelkit.load_model("my_last_model", models=MyModel)
    assert my_last_model.predict({}) == "asset-2021-11-15T17-31-06Z"


def test_asset_spec_sort_versions():
    spec = AssetSpec(name="name", versioning="simple_date")
    version_list = [
        "2021-11-15T17-30-56Z",
        "2020-11-15T17-30-56Z",
        "2021-10-15T17-30-56Z",
    ]
    result = [
        "2021-11-15T17-30-56Z",
        "2021-10-15T17-30-56Z",
        "2020-11-15T17-30-56Z",
    ]
    assert spec.sort_versions(version_list) == result


def test_asset_spec_get_local_versions():
    spec = AssetSpec(name="name", versioning="simple_date")
    assert spec.get_local_versions("not_a_dir") == []
    asset_dir = [
        "testdata",
        "test-bucket",
        "assets-prefix",
        "category",
        "simple_date_asset",
    ]
    local_path = os.path.join(tests.TEST_DIR, *asset_dir)
    assert spec.get_local_versions(local_path) == [
        "2021-11-15T17-31-06Z",
        "2021-11-14T18-00-00Z",
    ]


@pytest.mark.parametrize("s, spec", get_string_spec(["2021-11-14T18-00-00Z"]))
def test_string_asset_spec(s, spec):
    assert AssetSpec.from_string(s, versioning="simple_date") == AssetSpec(
        versioning="simple_date", **spec
    )


def test_asset_spec_set_latest_version():
    spec = AssetSpec(name="a", versioning="simple_date")
    spec.set_latest_version(["2021-11-15T17-31-06Z", "2021-11-14T18-00-00Z"])
    assert spec.version == "2021-11-15T17-31-06Z"

    spec = AssetSpec(name="a", version="2021-11-14T18-00-00Z", versioning="simple_date")
    spec.set_latest_version(["2021-11-15T17-31-06Z", "2021-11-14T18-00-00Z"])
    assert spec.version == "2021-11-15T17-31-06Z"