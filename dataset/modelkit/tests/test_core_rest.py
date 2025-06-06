import asyncio
import os

import pytest

from modelkit.assets.errors import AssetDoesNotExistError, InvalidAssetSpecError
from modelkit.core import errors
from modelkit.core.library import (
    ConfigurationNotFoundException,
    ModelLibrary,
    download_assets,
    load_model,
)
from modelkit.core.model import Asset, AsyncModel, Model
from modelkit.core.model_configuration import (
    ModelConfiguration,
    _configurations_from_objects,
    configure,
    list_assets,
)
from modelkit.core.settings import LibrarySettings
from tests import TEST_DIR, testmodels
from tests.assets.test_versioning import test_versioning




@pytest.mark.asyncio
async def test_predict_async():
    class SomeModel(AsyncModel):
        async def _predict(self, item):
            return item

    m = SomeModel()
    r = await m.predict({})
    assert r == {}

    class SomeModelBatch(AsyncModel):
        async def _predict(self, item):
            return item

    m = SomeModelBatch()
    r = await m.predict({})
    assert r == {}














@pytest.mark.parametrize("error", [KeyError, ZeroDivisionError])






@pytest.mark.parametrize(*test_versioning.TWO_VERSIONING_PARAMETRIZE)


@pytest.mark.parametrize(*test_versioning.TWO_VERSIONING_PARAMETRIZE)
















SYNC_ASYNC_TEST_CASES = [
    {"item": "", "result": 0},
    {"item": "a", "result": 1},
    {"item": ["a", "b", "c"], "result": 3},
    {"item": range(100), "result": 100},
]

















    m = SomeModel()
    assert m.predict({}) == {}

    class SomeModelBatch(Model):

    m = SomeModelBatch()
    assert m.predict({}) == {}


@pytest.mark.asyncio
async def test_predict_async():
    class SomeModel(AsyncModel):
        async def _predict(self, item):
            return item

    m = SomeModel()
    r = await m.predict({})
    assert r == {}

    class SomeModelBatch(AsyncModel):
        async def _predict(self, item):
            return item

    m = SomeModelBatch()
    r = await m.predict({})
    assert r == {}


def test_override_asset():
    class TestModel(Model):


    class TestDepModel(Model):

    config = {
        "some_asset": ModelConfiguration(
            model_type=TestModel,
            asset="asset/that/does/not/exist",
            model_dependencies={"dep_model"},
        ),
        "dep_model": ModelConfiguration(model_type=TestDepModel),
    }
    # The asset does not exist
    with pytest.raises(AssetDoesNotExistError):
        model_library = ModelLibrary(
            required_models=["some_asset"], configuration=config
        )

    # It does when overriden
    model_library = ModelLibrary(
        required_models={"some_asset": {"asset_path": "/the/path"}},
        configuration=config,
    )
    model = model_library.get("some_asset")
    assert "/the/path" == model({})

    # Dependent models are loaded properly
    model = model_library.get("dep_model")
    assert "dep" == model({})

    # Finally, it is possible to also specify
    # an asset for the dependent model
    config["dep_model"] = ModelConfiguration(
        model_type=TestDepModel, asset="cat/someasset"
    )
    model_library = ModelLibrary(
        required_models={
            "some_asset": {"asset_path": "/the/path"},
            "dep_model": {"asset_path": "/the/dep/path"},
        },
        configuration=config,
    )
    # Dependent models are loaded properly
    model = model_library.get("dep_model")
    assert "dep/the/dep/path" == model({})


def test_model_library_inexistent_model():
    with pytest.raises(ConfigurationNotFoundException):
        ModelLibrary(required_models=["model_that_does_not_exist"])

    configuration = {
        "existent_model": ModelConfiguration(
            model_type=Model, model_dependencies={"inexistent_model"}
        )
    }
    with pytest.raises(ConfigurationNotFoundException):
        ModelLibrary(required_models=["existent_model"], configuration=configuration)

    p = ModelLibrary(
        required_models=["model_that_does_not_exist"], settings={"lazy_loading": True}
    )
    with pytest.raises(ConfigurationNotFoundException):
        p.get("model_that_does_not_exist")
    with pytest.raises(ConfigurationNotFoundException):
        p.get("other_model_that_does_not_exist")


def test__configurations_from_objects():
    class SomeModel(Model):
        CONFIGURATIONS = {"yolo": {}, "les simpsons": {}}

    class SomeModel2(Asset):
        CONFIGURATIONS = {"yolo2": {}, "les simpsons2": {}}

    class ModelNoConf(Asset):
        pass

    configurations = _configurations_from_objects(SomeModel)
    assert "yolo" in configurations
    assert "les simpsons" in configurations

    configurations = _configurations_from_objects(ModelNoConf)
    assert {} == configurations

    configurations = _configurations_from_objects([SomeModel, SomeModel2, ModelNoConf])
    assert "yolo" in configurations
    assert "yolo2" in configurations
    assert "les simpsons" in configurations
    assert "les simpsons2" in configurations

    class Something:
        pass

    with pytest.raises(ValueError):
        _configurations_from_objects(Something)


def test_configure_override():
    class SomeModel(Model):
        CONFIGURATIONS = {"yolo": {"asset": "ok/boomer"}, "les simpsons": {}}

    class SomeOtherModel(Model):
        pass

    configurations = configure(models=SomeModel)
    assert configurations["yolo"].model_type == SomeModel
    assert configurations["yolo"].asset == "ok/boomer"
    assert configurations["les simpsons"].model_type == SomeModel
    assert configurations["les simpsons"].asset is None

    configurations = configure(
        models=SomeModel,
        configuration={"somethingelse": ModelConfiguration(model_type=SomeOtherModel)},
    )
    assert configurations["yolo"].model_type == SomeModel
    assert configurations["yolo"].asset == "ok/boomer"
    assert configurations["les simpsons"].model_type == SomeModel
    assert configurations["les simpsons"].asset is None
    assert configurations["somethingelse"].model_type == SomeOtherModel
    assert configurations["somethingelse"].asset is None

    configurations = configure(
        models=SomeModel,
        configuration={"yolo": ModelConfiguration(model_type=SomeOtherModel)},
    )
    assert configurations["yolo"].model_type == SomeOtherModel
    assert configurations["yolo"].asset is None

    configurations = configure(
        models=SomeModel,
        configuration={"yolo": {"asset": "something/else"}},
    )
    assert configurations["yolo"].model_type == SomeModel
    assert configurations["yolo"].asset == "something/else"

    configurations = configure(
        models=SomeModel,
        configuration={"yolo2": {"model_type": SomeOtherModel}},
    )
    assert configurations["yolo"].model_type == SomeModel
    assert configurations["yolo"].asset == "ok/boomer"
    assert configurations["yolo2"].model_type == SomeOtherModel
    assert configurations["yolo2"].asset is None


def test_modellibrary_required_models():
    class SomeModel(Model):
        CONFIGURATIONS = {"yolo": {}, "les simpsons": {}}


    p = ModelLibrary(models=SomeModel)
    m = p.get("yolo")
    assert m
    assert m.configuration_key == "yolo"
    assert m.__class__.__name__ == "SomeModel"
    assert m.model_settings == {}
    assert m.asset_path == ""
    assert m.batch_size is None

    class SomeOtherModel(Model):
        pass

    with pytest.raises(ValueError):
        # model does not exist
        p.get("yolo", model_type=SomeOtherModel)


def test_modellibrary_no_models(monkeypatch):
    monkeypatch.setenv("modelkit_MODELS", "")
    p = ModelLibrary(models=None)
    assert p.configuration == {}
    assert p.required_models == {}

    with pytest.raises(errors.ModelsNotFound):
        # model does not exist
        p.get("some_model")


@pytest.mark.parametrize("error", [KeyError, ZeroDivisionError])
def test_modellibrary_error_in_load(error):
    class SomeModel(Model):
        CONFIGURATIONS = {"model": {}}



    library = ModelLibrary(
        models=SomeModel,
        settings={"lazy_loading": True},
    )

    try:
        library.get("model")
        assert False
    except error as err:
        assert "not loaded" not in str(err)


def test_lazy_loading_dependencies():
    class Model0(Asset):
        CONFIGURATIONS = {"model0": {}}


    class Model1(Model):
        CONFIGURATIONS = {"model1": {"model_dependencies": {"model0"}}}



    p = ModelLibrary(models=[Model1, Model0], settings={"lazy_loading": True})
    m = p.get("model1")
    assert m({}) == "ok"
    assert m.model_dependencies["model0"].some_attribute == "ok"
    assert m.some_attribute == "ok"


def test_list_assets():
    class SomeModel(Asset):
        CONFIGURATIONS = {"model0": {"asset": "some/asset"}}

    class SomeOtherModel(Asset):
        CONFIGURATIONS = {"model1": {"asset": "some/asset"}}

    assert {"some/asset"} == list_assets(SomeModel)
    assert {"some/asset"} == list_assets([SomeModel, SomeOtherModel])
    assert {"some/asset", "some/otherasset"} == list_assets(
        [SomeModel, SomeOtherModel],
        configuration={"model1": {"asset": "some/otherasset"}},
    )
    assert {"some/asset"} == list_assets(
        [SomeModel, SomeOtherModel],
        required_models=["model0"],
        configuration={"model1": {"asset": "some/otherasset"}},
    )


@pytest.mark.parametrize(*test_versioning.TWO_VERSIONING_PARAMETRIZE)
def test_download_assets_version(
    version_asset_name,
    version_1,
    version_2,
    versioning,
    assetsmanager_settings,
    monkeypatch,
):
    if versioning:
        monkeypatch.setenv("MODELKIT_ASSETS_VERSIONING_SYSTEM", versioning)

    class SomeModel(Asset):
        CONFIGURATIONS = {
            "model0": {"asset": f"category/{version_asset_name}:{version_1}"}
        }

    model_assets, assets_info = download_assets(
        assetsmanager_settings=assetsmanager_settings,
        models=[SomeModel],
    )
    assert model_assets["model0"] == {f"category/{version_asset_name}:{version_1}"}
    assert (
        assets_info[f"category/{version_asset_name}:{version_1}"].version == version_1
    )

    class SomeModel(Asset):
        CONFIGURATIONS = {"model0": {"asset": f"category/{version_asset_name}"}}

    model_assets, assets_info = download_assets(
        assetsmanager_settings=assetsmanager_settings,
        models=[SomeModel],
    )
    assert model_assets["model0"] == {f"category/{version_asset_name}"}
    assert assets_info[f"category/{version_asset_name}"].version == version_2

    if versioning in (None, "major_minor"):

        class SomeModel(Asset):
            CONFIGURATIONS = {"model0": {"asset": "category/asset:0"}}

        model_assets, assets_info = download_assets(
            assetsmanager_settings=assetsmanager_settings,
            models=[SomeModel],
        )
        assert model_assets["model0"] == {"category/asset:0"}
        assert assets_info["category/asset:0"].version == "0.1"


@pytest.mark.parametrize(*test_versioning.TWO_VERSIONING_PARAMETRIZE)
def test_download_assets_dependencies(
    version_asset_name,
    version_1,
    version_2,
    versioning,
    assetsmanager_settings,
    monkeypatch,
):
    if versioning:
        monkeypatch.setenv("MODELKIT_ASSETS_VERSIONING_SYSTEM", versioning)

    class SomeModel(Asset):
        CONFIGURATIONS = {"model0": {"asset": f"category/{version_asset_name}"}}

    class SomeOtherModel(Asset):
        CONFIGURATIONS = {
            "model1": {
                "asset": f"category/{version_asset_name}:{version_1}",
                "model_dependencies": {"model0"},
            }
        }

    model_assets, assets_info = download_assets(
        assetsmanager_settings=assetsmanager_settings,
        models=[SomeModel, SomeOtherModel],
    )

    assert model_assets["model0"] == {f"category/{version_asset_name}"}
    assert model_assets["model1"] == {
        f"category/{version_asset_name}:{version_1}",
        f"category/{version_asset_name}",
    }
    assert assets_info[f"category/{version_asset_name}"].version == version_2
    assert (
        assets_info[f"category/{version_asset_name}:{version_1}"].version == version_1
    )


def test_load_model():
    class SomeModel(Model):
        CONFIGURATIONS = {"model": {}}


    m = load_model("model", models=SomeModel)
    assert m({"ok": "boomer"}) == {"ok": "boomer"}


def test_required_models():
    class SomeModel(Model):
        CONFIGURATIONS = {"model": {}}


    class SomeOtherModel(Model):
        CONFIGURATIONS = {"other_model": {}}


    lib = ModelLibrary(required_models=[], models=[SomeModel, SomeOtherModel])
    assert len(lib.models) == 0
    assert lib.required_models == {}

    lib = ModelLibrary(models=[SomeModel, SomeOtherModel])
    assert len(lib.models) == 2
    assert lib.required_models == {"model": {}, "other_model": {}}


def test_lazy_loading_setting(monkeypatch):
    monkeypatch.delenv("MODELKIT_LAZY_LOADING", raising=False)
    settings = LibrarySettings()
    assert not settings.lazy_loading
    monkeypatch.setenv("MODELKIT_LAZY_LOADING", "True")
    settings = LibrarySettings()
    assert settings.lazy_loading


def test_environment_asset_load(monkeypatch, assetsmanager_settings):
    class TestModel(Model):


    monkeypatch.setenv("MODELKIT_TESTS_TEST_ASSET_FILE", "path/to/asset")

    model_library = ModelLibrary(
        required_models=["some_asset"],
        configuration={
            "some_asset": ModelConfiguration(
                model_type=TestModel, asset="tests/test_asset"
            )
        },
        assetsmanager_settings=assetsmanager_settings,
    )
    model = model_library.get("some_asset")

    predicted = model({})
    assert predicted == {"some key": "some data"}


def test_environment_asset_load_version(monkeypatch, assetsmanager_settings):
    class TestModel(Model):


    monkeypatch.setenv("MODELKIT_TESTS_TEST_ASSET_VERSION", "undef")

    with pytest.raises(InvalidAssetSpecError):
        ModelLibrary(
            required_models=["some_asset"],
            configuration={
                "some_asset": ModelConfiguration(
                    model_type=TestModel, asset="tests/test_asset"
                )
            },
            assetsmanager_settings=assetsmanager_settings,
        )


def test_rename_dependencies():
    class SomeModel(Model):
        CONFIGURATIONS = {"ok": {}}


    class SomeModel2(Model):
        CONFIGURATIONS = {"boomer": {}}


    class FinalModel(Model):
        CONFIGURATIONS = {
            "model_no_rename": {
                "model_dependencies": {"ok"},
            },
            "model_rename": {
                "model_dependencies": {"ok": "boomer"},
            },
        }


    lib = ModelLibrary(models=[SomeModel, SomeModel2, FinalModel])
    assert lib.get("model_no_rename")({}) == "ok"
    assert lib.get("model_rename")({}) == "boomer"


def test_override_assets_dir(assetsmanager_settings):
    class TestModel(Model):

    model_library = ModelLibrary(
        required_models=["my_model", "my_override_model"],
        configuration={
            "my_model": ModelConfiguration(
                model_type=TestModel, asset="category/asset"
            ),
            "my_override_model": ModelConfiguration(
                model_type=TestModel, asset="category/override-asset"
            ),
        },
        assetsmanager_settings=assetsmanager_settings,
    )

    prediction = model_library.get("my_model").predict({})
    assert prediction.endswith(os.path.join("category", "asset", "1.0"))

    prediction = model_library.get("my_override_model").predict({})
    assert prediction.endswith(os.path.join("category", "override-asset", "0.0"))

    model_library_override = ModelLibrary(
        required_models=["my_model", "my_override_model"],
        configuration={
            "my_model": ModelConfiguration(
                model_type=TestModel, asset="category/asset"
            ),
            "my_override_model": ModelConfiguration(
                model_type=TestModel, asset="category/override-asset"
            ),
        },
        settings={
            "override_assets_dir": os.path.join(
                TEST_DIR, "testdata", "override-assets-dir"
            ),
            "lazy_loading": True,
        },
        assetsmanager_settings=assetsmanager_settings,
    )

    prediction = model_library_override.get("my_model").predict({})
    assert prediction.endswith(os.path.join("category", "asset", "1.0"))

    prediction = model_library_override.get("my_override_model").predict({})
    assert prediction.endswith(os.path.join("category", "override-asset", "0.0"))


SYNC_ASYNC_TEST_CASES = [
    {"item": "", "result": 0},
    {"item": "a", "result": 1},
    {"item": ["a", "b", "c"], "result": 3},
    {"item": range(100), "result": 100},
]


def test_model_sync_test():
    class TestClass(Model):
        TEST_CASES = SYNC_ASYNC_TEST_CASES


    TestClass().test()


def test_model_async_test():
    class TestClass(AsyncModel):
        TEST_CASES = SYNC_ASYNC_TEST_CASES

        async def _predict(self, item, **_):
            await asyncio.sleep(0)
            return len(item)

    TestClass().test()


def test_auto_load():
    class SomeModel(Model):


    m = SomeModel()
    assert m.predict({}) == "OK"

    class SomeModelDep(Model):


    m = SomeModelDep(model_dependencies={"model": SomeModel()})
    assert m.predict({}) == "OK"


def test_model_dependencies_bad_get():
    class SomeModel(Model):
        CONFIGURATIONS = {"some_model": {}}



    class SomeModelDep(Model):
        CONFIGURATIONS = {"some_model_dep": {"model_dependencies": {"some_model"}}}



    lib = ModelLibrary(
        models=[SomeModel, SomeModelDep], required_models=["some_model_dep"]
    )
    lib.get("some_model_dep")


def test_asset_dependencies():
    """
    Test that MyAsset can have model dependencies of any kind
    (SomeAsset, SomeModel, SomeAsyncModel), loaded and available during its _load.
    """

    class SomeAsset(Asset):
        CONFIGURATIONS = {"some_asset": {}}


    class SomeModel(Model):
        CONFIGURATIONS = {"some_model": {}}



    class SomeAsyncModel(AsyncModel):
        CONFIGURATIONS = {"some_async_model": {}}


        async def _predict(self, _):
            await asyncio.sleep(0)
            return self.data

    class MyAsset(Asset):
        CONFIGURATIONS = {
            "my_asset": {
                "model_dependencies": {"some_asset", "some_model", "some_async_model"}
            }
        }


    class MyModel(Model):
        CONFIGURATIONS = {"my_model": {"model_dependencies": {"my_asset"}}}


    model = load_model(
        "my_model", models=[MyModel, MyAsset, SomeAsset, SomeModel, SomeAsyncModel]
    )
    assert model.predict(None) == {
        "my_asset": ["some_asset", "some_model", "some_async_model"]
    }


def test_model_multiple_load():
    loaded = 0

    class SomeModel(Model):
        CONFIGURATIONS = {"a": {}}



    class SomeModel2(Model):
        CONFIGURATIONS = {"b": {"model_dependencies": {"a"}}}



    lib = ModelLibrary(models=[SomeModel, SomeModel2])
    lib.get("b")
    lib.get("a")
    assert loaded == 1


def test_model_multiple_asset_load(working_dir, monkeypatch):
    monkeypatch.setenv("MODELKIT_ASSETS_DIR", working_dir)
    with open(os.path.join(working_dir, "something.txt"), "w") as f:
        f.write("OK")

    class SomeModel(Model):
        CONFIGURATIONS = {"a": {"asset": "something.txt"}}


    class SomeModel2(Model):
        CONFIGURATIONS = {"b": {"asset": "something.txt"}}


    fetched = 0

    def fake_fetch_asset(asset_spec, return_info=True):
        nonlocal fetched
        fetched += 1
        return {"path": os.path.join(working_dir, "something.txt")}

    lib = ModelLibrary(models=[SomeModel, SomeModel2], settings={"lazy_loading": True})
    monkeypatch.setattr(lib.assets_manager, "fetch_asset", fake_fetch_asset)
    lib.preload()

    assert fetched == 1


def test_model_sub_class(working_dir, monkeypatch):
    monkeypatch.setenv("MODELKIT_ASSETS_DIR", working_dir)
    with open(os.path.join(working_dir, "something.txt"), "w") as f:
        f.write("OK")

    class BaseAsset(Asset):

    class DerivedAsset(BaseAsset):
        CONFIGURATIONS = {"derived": {"asset": "something.txt"}}


    # Abstract models are not loaded even when explicitly requesting them
    lib = ModelLibrary(models=[DerivedAsset, BaseAsset])
    lib.preload()
    assert ["derived"] == list(lib.models.keys())

    # Abstract models are ignored when walking through modules
    lib = ModelLibrary(models=testmodels)
    lib.preload()
    assert ["derived_asset", "derived_model"] == sorted(list(lib.models.keys()))