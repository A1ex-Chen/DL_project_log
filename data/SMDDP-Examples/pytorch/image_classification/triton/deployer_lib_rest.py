#!/usr/bin/python

# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

import os
import sys
import time
import json
import torch
import argparse
import statistics
from collections import Counter

torch_type_to_triton_type = {
    torch.bool: "TYPE_BOOL",
    torch.int8: "TYPE_INT8",
    torch.int16: "TYPE_INT16",
    torch.int32: "TYPE_INT32",
    torch.int64: "TYPE_INT64",
    torch.uint8: "TYPE_UINT8",
    torch.float16: "TYPE_FP16",
    torch.float32: "TYPE_FP32",
    torch.float64: "TYPE_FP64",
}

CONFIG_TEMPLATE = r"""
name: "{model_name}"
platform: "{platform}"
max_batch_size: {max_batch_size}
input [
    {spec_inputs}
]
output [
    {spec_outputs}
]
{dynamic_batching}
{model_optimizations}
instance_group [
    {{
        count: {engine_count}
        kind: KIND_GPU
        gpus: [ {gpu_list} ]
    }}
]"""

INPUT_TEMPLATE = r"""
{{
    name: "input__{num}"
    data_type: {type}
    dims: {dims}
    {reshape}
}},"""

OUTPUT_TEMPLATE = r""" 
{{
    name: "output__{num}"
    data_type: {type}
    dims: {dims}
    {reshape}
}},"""

MODEL_OPTIMIZATION_TEMPLATE = r"""
optimization {{
  {execution_accelerator}
  cuda {{
    graphs: {capture_cuda_graph}
  }}
}}"""

EXECUTION_ACCELERATOR_TEMPLATE = r"""
  execution_accelerators {{
    gpu_execution_accelerator: [
      {{
        name: "tensorrt"
      }}
    ]
  }},"""






class DeployerLibrary:
















class Deployer:









        model_trt = TRT_model(engine, input_names, output_names, output_types, device)

        # run both models on inputs
        assert not model.training, "internal error - model should be in eval() mode! "
        models = (model, model_trt)
        outputs, time_model, outputs_trt, time_model_trt = self.lib.run_models(
            models, inputs
        )

        # check for errors
        Error_stats = self.lib.compute_errors(outputs, outputs_trt)
        self.lib.print_errors(Error_stats)
        print("time of error check of native model: ", time_model, "seconds")
        print("time of error check of trt model: ", time_model_trt, "seconds")
        print()

        # write TRTIS config
        config_filename = os.path.join(model_folder, "config.pbtxt")
        self.lib.write_config(
            config_filename, input_shapes, input_types, output_shapes, output_types
        )

    def name_onnx_nodes(self, model_path):
        """
        Name all unnamed nodes in ONNX model
            parameter model_path: path  ONNX model
            return: none
        """
        model = onnx.load(model_path)
        node_id = 0
        for node in model.graph.node:
            if len(node.name) == 0:
                node.name = "unnamed_node_%d" % node_id
            node_id += 1
        # This check partially validates model
        onnx.checker.check_model(model)
        onnx.save(model, model_path)
        # Only inference really checks ONNX model for some issues
        # like duplicated node names
        onnxruntime.InferenceSession(model_path, None)

    def to_triton_onnx(self, dataloader, model):
        """ export the model to onnx and test correctness on dataloader """
        import onnx as local_onnx

        global onnx
        onnx = local_onnx
        import onnxruntime as local_onnxruntime

        global onnxruntime
        onnxruntime = local_onnxruntime
        # setup device
        if self.args.triton_no_cuda:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")

        # prepare model
        model.to(device)
        model.eval()
        assert not model.training, "internal error - model should be in eval() mode! "

        # prepare inputs
        inputs = self.lib.prepare_inputs(dataloader, device)

        # generate outputs
        outputs = []
        for input in inputs:
            with torch.no_grad():
                output = model(*input)
            if type(output) is torch.Tensor:
                output = [output]
            outputs.append(output)

        # generate input shapes - dynamic tensor shape support
        input_shapes = self.lib.get_tuple_of_dynamic_shapes(inputs)

        # generate output shapes - dynamic tensor shape support
        output_shapes = self.lib.get_tuple_of_dynamic_shapes(outputs)

        # generate input types
        input_types = [x.dtype for x in inputs[0]]

        # generate output types
        output_types = [x.dtype for x in outputs[0]]

        # get input names
        rng = range(len(input_types))
        input_names = ["input__" + str(num) for num in rng]

        # get output names
        rng = range(len(output_types))
        output_names = ["output__" + str(num) for num in rng]

        # prepare save path
        model_folder = os.path.join(self.args.save_dir, self.args.triton_model_name)
        version_folder = os.path.join(model_folder, str(self.args.triton_model_version))
        if not os.path.exists(version_folder):
            os.makedirs(version_folder)
        final_model_path = os.path.join(version_folder, "model.onnx")

        # get indices of dynamic input and output shapes
        dynamic_axes = {}
        for input_name, input_shape in zip(input_names, input_shapes):
            dynamic_axes[input_name] = [i for i, x in enumerate(input_shape) if x == -1]
        for output_name, output_shape in zip(output_names, output_shapes):
            dynamic_axes[output_name] = [
                i for i, x in enumerate(output_shape) if x == -1
            ]

        # export the model
        assert not model.training, "internal error - model should be in eval() mode! "
        with torch.no_grad():
            torch.onnx.export(
                model,
                inputs[0],
                final_model_path,
                verbose=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=11,
            )

        # syntactic error check
        converted_model = onnx.load(final_model_path)
        # check that the IR is well formed
        onnx.checker.check_model(converted_model)

        # Name unnamed nodes - it helps for some other processing tools
        self.name_onnx_nodes(final_model_path)
        converted_model = onnx.load(final_model_path)

        # load the model
        session = onnxruntime.InferenceSession(final_model_path, None)

        class ONNX_model:



        # switch to eval mode
        model_onnx = ONNX_model(session, input_names, device)

        # run both models on inputs
        assert not model.training, "internal error - model should be in eval() mode! "
        models = (model, model_onnx)
        outputs, time_model, outputs_onnx, time_model_onnx = self.lib.run_models(
            models, inputs
        )

        # check for errors
        Error_stats = self.lib.compute_errors(outputs, outputs_onnx)
        self.lib.print_errors(Error_stats)
        print("time of error check of native model: ", time_model, "seconds")
        print("time of error check of onnx model: ", time_model_onnx, "seconds")
        print()

        # write TRTIS config
        config_filename = os.path.join(model_folder, "config.pbtxt")
        self.lib.write_config(
            config_filename, input_shapes, input_types, output_shapes, output_types
        )

    def to_triton_torchscript(self, dataloader, model):
        """ export the model to torchscript and test correctness on dataloader """
        # setup device
        if self.args.triton_no_cuda:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")

        # prepare model
        model.to(device)
        model.eval()
        assert not model.training, "internal error - model should be in eval() mode! "

        # prepare inputs
        inputs = self.lib.prepare_inputs(dataloader, device)

        # generate input shapes - dynamic tensor shape support
        input_shapes = self.lib.get_tuple_of_dynamic_shapes(inputs)

        # generate input types
        input_types = [x.dtype for x in inputs[0]]

        # prepare save path
        model_folder = os.path.join(self.args.save_dir, self.args.triton_model_name)
        version_folder = os.path.join(model_folder, str(self.args.triton_model_version))
        if not os.path.exists(version_folder):
            os.makedirs(version_folder)
        final_model_path = os.path.join(version_folder, "model.pt")

        # convert the model
        with torch.no_grad():
            if self.args.ts_trace:  # trace it
                model_ts = torch.jit.trace(model, inputs[0])
            if self.args.ts_script:  # script it
                model_ts = torch.jit.script(model)

        # save the model
        torch.jit.save(model_ts, final_model_path)

        # load the model
        model_ts = torch.jit.load(final_model_path)
        model_ts.eval()  # WAR for bug : by default, model_ts gets loaded in training mode

        # run both models on inputs
        assert not model.training, "internal error - model should be in eval() mode! "
        assert (
            not model_ts.training
        ), "internal error - converted model should be in eval() mode! "
        models = (model, model_ts)
        outputs, time_model, outputs_ts, time_model_ts = self.lib.run_models(
            models, inputs
        )

        # check for errors
        Error_stats = self.lib.compute_errors(outputs, outputs_ts)
        self.lib.print_errors(Error_stats)
        print("time of error check of native model: ", time_model, "seconds")
        print("time of error check of ts model: ", time_model_ts, "seconds")
        print()

        # generate output shapes - dynamic tensor shape support
        output_shapes = self.lib.get_tuple_of_dynamic_shapes(outputs)

        # generate output types
        output_types = [x.dtype for x in outputs[0]]

        # now we build the config for TRTIS
        config_filename = os.path.join(model_folder, "config.pbtxt")
        self.lib.write_config(
            config_filename, input_shapes, input_types, output_shapes, output_types
        )