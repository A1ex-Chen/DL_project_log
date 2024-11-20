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
import sys
from pathlib import Path
from typing import Dict, NamedTuple, Optional, Union

import numpy as np

# pytype: disable=import-error
try:
    import pycuda.autoinit
    import pycuda.driver as cuda
except (ImportError, Exception) as e:
    logging.getLogger(__name__).warning(f"Problems with importing pycuda package; {e}")
# pytype: enable=import-error

import tensorrt as trt  # pytype: disable=import-error

from ..core import BaseLoader, BaseRunner, BaseRunnerSession, BaseSaver, Format, Model, Precision, TensorSpec
from ..extensions import loaders, runners, savers

LOGGER = logging.getLogger(__name__)
TRT_LOGGER = trt.Logger(trt.Logger.INFO)

"""
documentation:
https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/index.html
https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#python_samples_section
"""


class TensorRTLoader(BaseLoader):


class TensorRTSaver(BaseSaver):



class TRTBuffers(NamedTuple):
    x_host: Optional[Dict[str, object]]
    x_dev: Dict[str, object]
    y_pred_host: Dict[str, object]
    y_pred_dev: Dict[str, object]


class TensorRTRunner(BaseRunner):



class TensorRTRunnerSession(BaseRunnerSession):







        for name in self._input_names:
            bindings_idx = self._engine[name]
            data_shape = x_host[name].shape  # pytype: disable=attribute-error
            if self._engine.is_shape_binding(bindings_idx):
                input_shape = self._context.get_shape(bindings_idx)
                if _is_shape_dynamic(input_shape):
                    self._context.set_shape_input(bindings_idx, data_shape)
            else:
                input_shape = self._engine.get_binding_shape(bindings_idx)
                if _is_shape_dynamic(input_shape):
                    self._context.set_binding_shape(bindings_idx, data_shape)

        assert self._context.all_binding_shapes_specified and self._context.all_shape_inputs_specified

    def _prepare_buffers_if_needed(self, x_host: Dict[str, object]):
        # pytype: disable=attribute-error
        new_batch_size = list(x_host.values())[0].shape[0]
        current_batch_size = list(self._buffers.y_pred_host.values())[0].shape[0] if self._buffers else 0
        # pytype: enable=attribute-error

        if self._has_dynamic_shapes or new_batch_size != current_batch_size:
            # TODO: are CUDA buffers dealloc automatically?

            self._set_dynamic_input_shapes(x_host)

            y_pred_host = {}
            for name in self._output_names:
                shape = self._context.get_binding_shape(self._engine[name])
                y_pred_host[name] = np.zeros(shape, dtype=trt.nptype(self._model.outputs[name].dtype))

            y_pred_dev = {name: cuda.mem_alloc(data.nbytes) for name, data in y_pred_host.items()}

            x_dev = {
                name: cuda.mem_alloc(host_input.nbytes)
                for name, host_input in x_host.items()
                if name in self._input_names  # pytype: disable=attribute-error
            }

            self._buffers = TRTBuffers(None, x_dev, y_pred_host, y_pred_dev)

        return self._buffers._replace(x_host=x_host)


if "pycuda.driver" in sys.modules:
    loaders.register_extension(Format.TRT.value, TensorRTLoader)
    runners.register_extension(Format.TRT.value, TensorRTRunner)
    savers.register_extension(Format.TRT.value, TensorRTSaver)
else:
    LOGGER.warning("Do not register TensorRT extension due problems with importing pycuda.driver package.")