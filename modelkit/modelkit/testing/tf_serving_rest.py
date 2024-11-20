import os
import subprocess

from modelkit.core.models.tensorflow_model import TensorflowModel, connect_tf_serving
from modelkit.utils.tensorflow import deploy_tf_models



    else:
        deploy_tf_models(lib, "local-docker", config_name="testing")
        # kill previous tfserving container (if any)
        subprocess.call(
            ["docker", "rm", "-f", "modelkit-tfserving-tests"],
            stderr=subprocess.DEVNULL,
        )
        # start tfserving as docker container
        tfserving_proc = subprocess.Popen(
            [
                "docker",
                "run",
                "--name",
                "modelkit-tfserving-tests",
                "--volume",
                f"{os.environ['MODELKIT_ASSETS_DIR']}:/config",
                "-p",
                "8500:8500",
                "-p",
                "8501:8501",
                f"tensorflow/serving:{tf_version}",
                "--model_config_file=/config/testing.config",
            ]
            + cmd
        )


    request.addfinalizer(finalize)
    connect_tf_serving(
        next(
            x
            for x in lib.required_models
            if issubclass(lib.configuration[x].model_type, TensorflowModel)
        ),
        "localhost",
        8500,
        "grpc",
    )