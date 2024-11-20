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

import sys
import os
import torch
import argparse
import triton.deployer_lib as deployer_lib







    return data_loader()


if __name__ == "__main__":
    # don't touch this!
    deployer, model_argv = deployer_lib.create_deployer(
        sys.argv[1:]
    )  # deployer and returns removed deployer arguments

    model_args = get_model_args(model_argv)

    model = initialize_model(model_args)
    dataloader = get_dataloader(model_args)

    if model_args.dump_perf_data:
        input_0 = next(iter(dataloader))
        if model_args.fp16:
            input_0 = input_0.half()

        os.makedirs(model_args.dump_perf_data, exist_ok=True)
        input_0.detach().cpu().numpy()[0].tofile(
            os.path.join(model_args.dump_perf_data, "input__0")
        )

    deployer.deploy(dataloader, model)