import os
import subprocess
import uuid

import pytest
import requests
import urllib3
from google.api_core.client_options import ClientOptions
from google.auth.credentials import AnonymousCredentials
from google.cloud import storage
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from modelkit.assets.manager import AssetsManager
from modelkit.assets.remote import StorageProvider

test_path = os.path.dirname(os.path.realpath(__file__))




@pytest.fixture(scope="function")




@pytest.fixture(scope="function")


@retry(
    wait=wait_random_exponential(multiplier=1, min=4, max=10),
    stop=stop_after_attempt(5),
    retry=retry_if_exception(lambda x: isinstance(x, Exception)),
    reraise=True,
)


@pytest.fixture(scope="function")


@retry(
    wait=wait_random_exponential(multiplier=1, min=4, max=10),
    stop=stop_after_attempt(5),
    retry=retry_if_exception(lambda x: isinstance(x, Exception)),
    reraise=True,
)


@pytest.fixture(scope="function")

    request.addfinalizer(finalize)

    storage_provider = StorageProvider(
        prefix="test-prefix",
        provider="gcs",
        bucket="test-bucket",
        client=_get_mock_gcs_client(),
    )
    storage_provider.driver.client.create_bucket("test-bucket")
    mng = AssetsManager(assets_dir=working_dir, storage_provider=storage_provider)
    yield mng


@retry(
    wait=wait_random_exponential(multiplier=1, min=4, max=10),
    stop=stop_after_attempt(5),
    retry=retry_if_exception(lambda x: isinstance(x, Exception)),
    reraise=True,
)
def _start_s3_manager(working_dir):
    mng = AssetsManager(
        assets_dir=working_dir,
        storage_provider=StorageProvider(
            prefix=f"test-assets-{uuid.uuid1().hex}",
            provider="s3",
            aws_default_region="us-east-1",
            bucket="test-assets",
            aws_access_key_id="minioadmin",
            aws_secret_access_key="minioadmin",
            aws_session_token=None,
            s3_endpoint="http://127.0.0.1:9000",
        ),
    )
    mng.storage_provider.driver.client.create_bucket(Bucket="test-assets")
    return mng


@pytest.fixture(scope="function")
def s3_assetsmanager(request, working_dir):
    # kill previous minio container (if any)
    subprocess.call(
        ["docker", "rm", "-f", "modelkit-storage-minio-tests"],
        stderr=subprocess.DEVNULL,
    )
    # start minio as docker container
    minio_proc = subprocess.Popen(
        [
            "docker",
            "run",
            "-p",
            "9000:9000",
            "--name",
            "modelkit-storage-minio-tests",
            "minio/minio",
            "server",
            "/data",
        ]
    )
    yield _start_s3_manager(working_dir)

    subprocess.call(["docker", "stop", "modelkit-storage-minio-tests"])
    minio_proc.wait()


@retry(
    wait=wait_random_exponential(multiplier=1, min=4, max=10),
    stop=stop_after_attempt(5),
    retry=retry_if_exception(lambda x: isinstance(x, Exception)),
    reraise=True,
)
def _start_az_manager(working_dir):
    mng = AssetsManager(
        assets_dir=working_dir,
        storage_provider=StorageProvider(
            prefix=f"test-assets-{uuid.uuid1().hex}",
            provider="az",
            bucket="test-assets",
            connection_string=(
                "DefaultEndpointsProtocol=http;"
                "AccountName=devstoreaccount1;"
                "AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSR"
                "Z6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;"
                "BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;"
                "QueueEndpoint=http://127.0.0.1:10001/devstoreaccount1;"
                "TableEndpoint=http://127.0.0.1:10002/devstoreaccount1;"
            ),
        ),
    )
    mng.storage_provider.driver.client.create_container("test-assets")
    return mng


@pytest.fixture(scope="function")
def az_assetsmanager(request, working_dir):
    # kill previous minio container (if any)
    subprocess.call(
        ["docker", "rm", "-f", "modelkit-storage-azurite-tests"],
        stderr=subprocess.DEVNULL,
    )
    # start minio as docker container
    azurite_proc = subprocess.Popen(
        [
            "docker",
            "run",
            "-p",
            "10002:10002",
            "-p",
            "10001:10001",
            "-p",
            "10000:10000",
            "--name",
            "modelkit-storage-azurite-tests",
            "mcr.microsoft.com/azure-storage/azurite",
        ]
    )
    yield _start_az_manager(working_dir)

    subprocess.call(["docker", "stop", "modelkit-storage-azurite-tests"])
    azurite_proc.wait()