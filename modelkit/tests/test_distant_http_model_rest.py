import json
import os
import subprocess
import sys
import time

import pydantic
import pytest
import requests

from modelkit.core.library import ModelLibrary
from modelkit.core.models.distant_model import (
    AsyncDistantHTTPBatchModel,
    AsyncDistantHTTPModel,
    DistantHTTPBatchModel,
    DistantHTTPModel,
)
from tests import TEST_DIR


@pytest.fixture(scope="module")




@pytest.fixture(scope="module")




@pytest.fixture(scope="module")




@pytest.fixture(scope="module")




@pytest.fixture(scope="module")




class SomeContentModel(pydantic.BaseModel):
    some_content: str


test_distant_http_model_args = (
    "item,headers,params,expected",
    [
        ({"some_content": "something"}, {}, {}, {"some_content": "something"}),
        (
            SomeContentModel(**{"some_content": "something"}),
            {},
            {},
            {"some_content": "something"},
        ),
        (
            {"some_content": "something"},
            {"X-Correlation-Id": "123-456-789"},
            {"limit": 10},
            {
                "some_content": "something",
                "limit": 10,
                "x_correlation_id": "123-456-789",
            },
        ),
        (
            SomeContentModel(**{"some_content": "something"}),
            {},
            {"limit": 10},
            {"some_content": "something", "limit": 10},
        ),
        (
            {"some_content": "something"},
            {},
            {"skip": 5},
            {"some_content": "something", "skip": 5},
        ),
        (
            SomeContentModel(**{"some_content": "something"}),
            {},
            {"skip": 5},
            {"some_content": "something", "skip": 5},
        ),
        (
            {"some_content": "something"},
            {},
            {"limit": 10, "skip": 5},
            {"some_content": "something", "limit": 10, "skip": 5},
        ),
        (
            SomeContentModel(**{"some_content": "something"}),
            {},
            {"limit": 10, "skip": 5},
            {"some_content": "something", "limit": 10, "skip": 5},
        ),
    ],
)


@pytest.mark.parametrize(*test_distant_http_model_args)




@pytest.mark.asyncio
@pytest.mark.parametrize(*test_distant_http_model_args)
async def test_async_distant_http_model(
    item,
    headers,
    params,
    expected,
    async_distant_http_model_lib,
    run_mocked_service,
    event_loop,
):
    try:
        m = async_distant_http_model_lib.get("test_async_distant_http_model")

        res = await m.predict(item, endpoint_headers=headers, endpoint_params=params)
        assert expected == res

        with pytest.raises(AssertionError):
            assert expected == m(item, endpoint_headers=headers, endpoint_params=params)

    except Exception:
        _stop_mocked_service_and_print_stderr(run_mocked_service)
        raise


async def test_async_distant_http_model_with_params(run_mocked_service, event_loop):
    try:
        for lib in _async_distant_http_model_lib(
            event_loop,
            endpoint_headers={"X-Correlation-Id": "123-456-789"},
            endpoint_params={"limit": 10},
        ):
            m = lib.get("test_async_distant_http_model")
            assert await m({"some_content": "something"}) == {
                "some_content": "something",
                "limit": 10,
                "x_correlation_id": "123-456-789",
            }
    except Exception:
        _stop_mocked_service_and_print_stderr(run_mocked_service)
        raise


class SomeOtherContentModel(pydantic.BaseModel):
    some_other_content: str


test_distant_http_batch_model_args = (
    "items,headers,params,expected",
    [
        (
            [
                {"some_content": "something"},
                {"some_other_content": "something_else"},
            ],
            {},
            {},
            [
                {"some_content": "something"},
                {"some_other_content": "something_else"},
            ],
        ),
        (
            [
                SomeContentModel(**{"some_content": "something"}),
                SomeOtherContentModel(**{"some_other_content": "something_else"}),
            ],
            {},
            {},
            [
                {"some_content": "something"},
                {"some_other_content": "something_else"},
            ],
        ),
        (
            [
                {"some_content": "something"},
                {"some_other_content": "something_else"},
            ],
            {"X-Correlation-Id": "123-456-789"},
            {"limit": 10},
            [
                {
                    "some_content": "something",
                    "limit": 10,
                    "x_correlation_id": "123-456-789",
                },
                {
                    "some_other_content": "something_else",
                    "limit": 10,
                    "x_correlation_id": "123-456-789",
                },
            ],
        ),
        (
            [
                SomeContentModel(**{"some_content": "something"}),
                SomeOtherContentModel(**{"some_other_content": "something_else"}),
            ],
            {},
            {"limit": 10},
            [
                {"some_content": "something", "limit": 10},
                {"some_other_content": "something_else", "limit": 10},
            ],
        ),
        (
            [
                {"some_content": "something"},
                {"some_other_content": "something_else"},
            ],
            {},
            {"skip": 5},
            [
                {"some_content": "something", "skip": 5},
                {"some_other_content": "something_else", "skip": 5},
            ],
        ),
        (
            [
                SomeContentModel(**{"some_content": "something"}),
                SomeOtherContentModel(**{"some_other_content": "something_else"}),
            ],
            {},
            {"skip": 5},
            [
                {"some_content": "something", "skip": 5},
                {"some_other_content": "something_else", "skip": 5},
            ],
        ),
        (
            [
                {"some_content": "something"},
                {"some_other_content": "something_else"},
            ],
            {},
            {"limit": 10, "skip": 5},
            [
                {"some_content": "something", "limit": 10, "skip": 5},
                {"some_other_content": "something_else", "limit": 10, "skip": 5},
            ],
        ),
        (
            [
                SomeContentModel(**{"some_content": "something"}),
                SomeOtherContentModel(**{"some_other_content": "something_else"}),
            ],
            {},
            {"limit": 10, "skip": 5},
            [
                {"some_content": "something", "limit": 10, "skip": 5},
                {"some_other_content": "something_else", "limit": 10, "skip": 5},
            ],
        ),
    ],
)


@pytest.mark.parametrize(*test_distant_http_batch_model_args)




@pytest.mark.asyncio
@pytest.mark.parametrize(*test_distant_http_batch_model_args)
async def test_async_distant_http_batch_model(
    items,
    headers,
    params,
    expected,
    async_distant_http_batch_model_lib,
    run_mocked_service,
    event_loop,
):
    try:
        m = async_distant_http_batch_model_lib.get(
            "test_async_distant_http_batch_model"
        )

        with pytest.raises(AssertionError):
            assert expected == m.predict_batch(
                items, endpoint_headers=headers, endpoint_params=params
            )

        res = await m.predict_batch(
            items, endpoint_headers=headers, endpoint_params=params
        )
        assert expected == res

    except Exception:
        run_mocked_service.terminate()
        svc_stderr = run_mocked_service.stderr.read().decode()
        print(svc_stderr, file=sys.stderr)
        raise


async def test_async_distant_http_batch_model_with_params(
    run_mocked_service, event_loop
):
    try:
        for lib in _async_distant_http_batch_model_lib(
            event_loop,
            endpoint_headers={"X-Correlation-Id": "123-456-789"},
            endpoint_params={"limit": 10},
        ):
            m = lib.get("test_async_distant_http_batch_model")
            assert await m.predict_batch([{"some_content": "something"}]) == [
                {
                    "some_content": "something",
                    "limit": 10,
                    "x_correlation_id": "123-456-789",
                }
            ]
    except Exception:
        _stop_mocked_service_and_print_stderr(run_mocked_service)
        raise