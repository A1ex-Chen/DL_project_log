# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, NamedTuple, Optional, Union

import torch  # pytype: disable=import-error
import yaml

from ..core import (
    GET_MODEL_FN_NAME,
    BaseConverter,
    BaseLoader,
    BaseRunner,
    BaseRunnerSession,
    BaseSaver,
    Format,
    Model,
    Precision,
    TensorSpec,
    load_from_file,
)
from ..extensions import converters, loaders, runners, savers
from .utils import get_dynamic_axes, get_input_shapes, get_shapes_with_dynamic_axes

LOGGER = logging.getLogger(__name__)


class InputOutputSpec(NamedTuple):
    inputs: Dict[str, TensorSpec]
    outputs: Dict[str, TensorSpec]










### TODO assumption: floating point input
### type has same precision as the model

    input_dtypes = {}
    output_dtypes = {}

    for batch in dataloader:
        _, x, y = batch
        input_dtypes = _get_dtypes(x)
        output_dtypes = _get_dtypes(y)
        break

    return input_dtypes, output_dtypes


### TODO assumption: floating point input
### type has same precision as the model
def _get_io_spec(model, dataloader_fn):
    precision = model.precision

    dataloader = dataloader_fn()
    input_dtypes, output_dtypes = _get_tensor_dtypes(dataloader, precision)
    input_shapes, output_shapes = get_shapes_with_dynamic_axes(dataloader)

    inputs = {
        name: TensorSpec(name=name, dtype=input_dtypes[name], shape=tuple(input_shapes[name])) for name in model.inputs
    }
    outputs = {
        name: TensorSpec(name=name, dtype=output_dtypes[name], shape=tuple(output_shapes[name]))
        for name in model.outputs
    }

    return InputOutputSpec(inputs, outputs)


class PyTorchModelLoader(BaseLoader):
    required_fn_name_for_signature_parsing: Optional[str] = GET_MODEL_FN_NAME




class TorchScriptLoader(BaseLoader):



class TorchScriptTraceConverter(BaseConverter):



class TorchScriptScriptConverter(BaseConverter):



class PYT2ONNXConverter(BaseConverter):



class PYT2TensorRTConverter(BaseConverter):


    @staticmethod


class TorchScriptSaver(BaseSaver):


class PyTorchRunner(BaseRunner):



class PyTorchRunnerSession(BaseRunnerSession):




        # store TensorSpecs from inputs and outputs in a yaml file
        tensor_specs = {
            "inputs": {k: _format_tensor_spec(v) for k, v in model.inputs.items()},
            "outputs": {k: _format_tensor_spec(v) for k, v in model.outputs.items()},
        }

        yaml_path = model_path.parent / f"{model_path.stem}.yaml"
        with Path(yaml_path).open("w") as fh:
            yaml.dump(tensor_specs, fh, indent=4)


class PyTorchRunner(BaseRunner):
    def __init__(self):
        pass

    def init_inference(self, model: Model):
        return PyTorchRunnerSession(model=model)


class PyTorchRunnerSession(BaseRunnerSession):
    def __init__(self, model: Model):
        super().__init__(model)

        assert isinstance(model.handle, torch.jit.ScriptModule) or isinstance(
            model.handle, torch.nn.Module
        ), "The model must be of type 'torch.jit.ScriptModule' or 'torch.nn.Module'. Runner aborted."

        self._model = model
        self._output_names = None

    def __enter__(self):
        self._output_names = list(self._model.outputs)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._output_names = None
        self._model = None

    def __call__(self, x: Dict[str, object]):
        with torch.no_grad():
            feed_list = [torch.from_numpy(v).cuda() for k, v in x.items()]
            y_pred = self._model.handle(*feed_list)
            if isinstance(y_pred, torch.Tensor):
                y_pred = (y_pred,)
            y_pred = [t.cpu().numpy() for t in y_pred]
            y_pred = dict(zip(self._output_names, y_pred))

        return y_pred


loaders.register_extension(Format.PYT.value, PyTorchModelLoader)
loaders.register_extension(Format.TS_TRACE.value, TorchScriptLoader)
loaders.register_extension(Format.TS_SCRIPT.value, TorchScriptLoader)

converters.register_extension(f"{Format.PYT.value}--{Format.TS_SCRIPT.value}", TorchScriptScriptConverter)
converters.register_extension(f"{Format.PYT.value}--{Format.TS_TRACE.value}", TorchScriptTraceConverter)
converters.register_extension(f"{Format.PYT.value}--{Format.ONNX.value}", PYT2ONNXConverter)
converters.register_extension(f"{Format.PYT.value}--{Format.TRT.value}", PYT2TensorRTConverter)

savers.register_extension(Format.TS_SCRIPT.value, TorchScriptSaver)
savers.register_extension(Format.TS_TRACE.value, TorchScriptSaver)

runners.register_extension(Format.PYT.value, PyTorchRunner)
runners.register_extension(Format.TS_SCRIPT.value, PyTorchRunner)
runners.register_extension(Format.TS_TRACE.value, PyTorchRunner)