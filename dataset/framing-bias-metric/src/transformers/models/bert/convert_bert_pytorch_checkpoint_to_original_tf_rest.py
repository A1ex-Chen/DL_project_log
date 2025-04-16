# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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

"""Convert Huggingface Pytorch checkpoint to Tensorflow checkpoint."""

import argparse
import os

import numpy as np
import tensorflow as tf
import torch

from transformers import BertModel






    tf.reset_default_graph()
    with tf.Session() as session:
        for var_name in state_dict:
            tf_name = to_tf_var_name(var_name)
            torch_tensor = state_dict[var_name].numpy()
            if any([x in var_name for x in tensors_to_transpose]):
                torch_tensor = torch_tensor.T
            tf_var = create_tf_var(tensor=torch_tensor, name=tf_name, session=session)
            tf.keras.backend.set_value(tf_var, torch_tensor)
            tf_weight = session.run(tf_var)
            print("Successfully created {}: {}".format(tf_name, np.allclose(tf_weight, torch_tensor)))

        saver = tf.train.Saver(tf.trainable_variables())
        saver.save(session, os.path.join(ckpt_dir, model_name.replace("-", "_") + ".ckpt"))


def main(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="model name e.g. bert-base-uncased")
    parser.add_argument(
        "--cache_dir", type=str, default=None, required=False, help="Directory containing pytorch model"
    )
    parser.add_argument("--pytorch_model_path", type=str, required=True, help="/path/to/<pytorch-model-name>.bin")
    parser.add_argument("--tf_cache_dir", type=str, required=True, help="Directory in which to save tensorflow model")
    args = parser.parse_args(raw_args)

    model = BertModel.from_pretrained(
        pretrained_model_name_or_path=args.model_name,
        state_dict=torch.load(args.pytorch_model_path),
        cache_dir=args.cache_dir,
    )

    convert_pytorch_checkpoint_to_tf(model=model, ckpt_dir=args.tf_cache_dir, model_name=args.model_name)


if __name__ == "__main__":
    main()