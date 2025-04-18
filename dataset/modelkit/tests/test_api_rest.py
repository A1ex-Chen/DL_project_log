import asyncio
import os
import platform

import fastapi
import pydantic
import pytest
from starlette.testclient import TestClient

from modelkit import testing
from modelkit.api import ModelkitAutoAPIRouter, create_modelkit_app
from modelkit.core.errors import ModelsNotFound
from modelkit.core.model import Asset, AsyncModel, Model
from tests import TEST_DIR


class SomeSimpleValidatedModel(Model[str, str]):
    """
    This is a summary

    that also has plenty more text
    """

    CONFIGURATIONS = {"some_model": {}}

    def _predict(self, item):
        return item


class ItemModel(pydantic.BaseModel):
    string: str


class ResultModel(pydantic.BaseModel):
    sorted: str


class SomeComplexValidatedModel(Model[ItemModel, ResultModel]):
    """
    More complex

    With **a lot** of documentation
    """

    CONFIGURATIONS = {"some_complex_model": {}}

    def _predict(self, item):
        return {"sorted": "".join(sorted(item.string))}


class NotValidatedModel(Model):
    CONFIGURATIONS = {"unvalidated_model": {}}

    def _predict(self, item):
        return item


class SomeAsyncModel(AsyncModel[ItemModel, ResultModel]):
    CONFIGURATIONS = {"async_model": {}}

    async def _predict(self, item):
        await asyncio.sleep(0)
        return {"sorted": "".join(sorted(item.string))}


class SomeAsset(Asset):
    """
    This is not a Model, it won't appear in the service
    """

    CONFIGURATIONS = {"some_asset": {}}

    def _predict(self, item):
        return {"sorted": "".join(sorted(item.string))}


@pytest.fixture(scope="module")


@pytest.mark.parametrize("item", ["ok", "ko"])


@pytest.mark.parametrize(
    "item, model",
    [
        ({"string": "ok"}, "some_complex_model"),
        ({"string": "ok"}, "async_model"),
    ],
)


EXCLUDED = ["load time", "load memory"]






@pytest.mark.parametrize(
    "required_models_env_var, models, required_models, n_endpoints",
    [
        (None, [SomeSimpleValidatedModel, SomeComplexValidatedModel], [], 4),
        (
            "some_model:some_complex_model",
            [SomeSimpleValidatedModel, SomeComplexValidatedModel],
            [],
            8,
        ),
        (
            "some_complex_model",
            [SomeSimpleValidatedModel, SomeComplexValidatedModel],
            [],
            6,
        ),
        (
            None,
            [SomeSimpleValidatedModel, SomeComplexValidatedModel],
            ["some_model", "some_complex_model"],
            8,
        ),
        (
            None,
            [SomeSimpleValidatedModel, SomeComplexValidatedModel],
            ["some_model"],
            6,
        ),
    ],
)




class ItemModel(pydantic.BaseModel):
    string: str


class ResultModel(pydantic.BaseModel):
    sorted: str


class SomeComplexValidatedModel(Model[ItemModel, ResultModel]):
    """
    More complex

    With **a lot** of documentation
    """

    CONFIGURATIONS = {"some_complex_model": {}}



class NotValidatedModel(Model):
    CONFIGURATIONS = {"unvalidated_model": {}}



class SomeAsyncModel(AsyncModel[ItemModel, ResultModel]):
    CONFIGURATIONS = {"async_model": {}}

    async def _predict(self, item):
        await asyncio.sleep(0)
        return {"sorted": "".join(sorted(item.string))}


class SomeAsset(Asset):
    """
    This is not a Model, it won't appear in the service
    """

    CONFIGURATIONS = {"some_asset": {}}



@pytest.fixture(scope="module")
def api_no_type(event_loop):
    np = pytest.importorskip("numpy")

    class ValidationNotSupported(Model[np.ndarray, np.ndarray]):
        CONFIGURATIONS = {"no_supported_model": {}}


    router = ModelkitAutoAPIRouter(
        required_models=[
            "unvalidated_model",
            "no_supported_model",
            "some_model",
            "some_complex_model",
            "some_asset",
            "async_model",
        ],
        models=[
            ValidationNotSupported,
            NotValidatedModel,
            SomeSimpleValidatedModel,
            SomeComplexValidatedModel,
            SomeAsset,
            SomeAsyncModel,
        ],
    )

    app = fastapi.FastAPI()
    app.include_router(router)

    with TestClient(app) as client:
        yield client


@pytest.mark.parametrize("item", ["ok", "ko"])
def test_api_simple_type(item, api_no_type):
    res = api_no_type.post(
        "/predict/some_model", headers={"Content-Type": "application/json"}, json=item
    )
    assert res.status_code == 200
    assert res.json() == item

    res = api_no_type.post(
        "/predict/batch/some_model",
        headers={"Content-Type": "application/json"},
        json=[item],
    )
    assert res.status_code == 200
    assert res.json() == [item]


@pytest.mark.parametrize(
    "item, model",
    [
        ({"string": "ok"}, "some_complex_model"),
        ({"string": "ok"}, "async_model"),
    ],
)
def test_api_complex_type(item, model, api_no_type):
    res = api_no_type.post(
        f"/predict/{model}",
        headers={"Content-Type": "application/json"},
        json=item,
    )
    assert res.status_code == 200
    assert res.json()["sorted"] == "".join(sorted(item["string"]))

    res = api_no_type.post(
        f"/predict/batch/{model}",
        headers={"Content-Type": "application/json"},
        json=[item],
    )
    assert res.status_code == 200
    assert res.json()[0]["sorted"] == "".join(sorted(item["string"]))


EXCLUDED = ["load time", "load memory"]


def _strip_description_fields(spec):
    if isinstance(spec, str):
        return "\n".join(
            line for line in spec.split("\n") if not any(x in line for x in EXCLUDED)
        )
    if isinstance(spec, list):
        return [_strip_description_fields(x) for x in spec]
    if isinstance(spec, dict):
        return {key: _strip_description_fields(value) for key, value in spec.items()}
    return spec


def test_api_doc(api_no_type):
    r = testing.ReferenceJson(os.path.join(TEST_DIR, "testdata", "api"))
    res = api_no_type.get(
        "/openapi.json",
    )
    if platform.system() != "Windows":
        # Output is different on Windows platforms since
        # modelkit.utils.memory cannot track memory increment
        # and write it
        r.assert_equal("openapi.json", _strip_description_fields(res.json()))


@pytest.mark.parametrize(
    "required_models_env_var, models, required_models, n_endpoints",
    [
        (None, [SomeSimpleValidatedModel, SomeComplexValidatedModel], [], 4),
        (
            "some_model:some_complex_model",
            [SomeSimpleValidatedModel, SomeComplexValidatedModel],
            [],
            8,
        ),
        (
            "some_complex_model",
            [SomeSimpleValidatedModel, SomeComplexValidatedModel],
            [],
            6,
        ),
        (
            None,
            [SomeSimpleValidatedModel, SomeComplexValidatedModel],
            ["some_model", "some_complex_model"],
            8,
        ),
        (
            None,
            [SomeSimpleValidatedModel, SomeComplexValidatedModel],
            ["some_model"],
            6,
        ),
    ],
)
def test_create_modelkit_app(
    required_models_env_var, models, required_models, n_endpoints, monkeypatch
):
    if required_models_env_var:
        monkeypatch.setenv("MODELKIT_REQUIRED_MODELS", required_models_env_var)
    app = create_modelkit_app(models=models, required_models=required_models)
    assert len([route.path for route in app.routes]) == n_endpoints


def test_create_modelkit_app_no_model(monkeypatch):
    monkeypatch.delenv("MODELKIT_DEFAULT_PACKAGE", raising=False)
    with pytest.raises(ModelsNotFound):
        create_modelkit_app(models=None)