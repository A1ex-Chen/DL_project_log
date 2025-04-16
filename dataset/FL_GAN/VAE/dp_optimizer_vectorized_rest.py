# Copyright 2020, The TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Vectorized differentially private optimizers for TensorFlow."""

from absl import logging
import tensorflow as tf

AdagradOptimizer = tf.compat.v1.train.AdagradOptimizer
AdamOptimizer = tf.compat.v1.train.AdamOptimizer
GradientDescentOptimizer = tf.compat.v1.train.GradientDescentOptimizer
parent_code = tf.compat.v1.train.Optimizer.compute_gradients.__code__
GATE_OP = tf.compat.v1.train.Optimizer.GATE_OP  # pylint: disable=invalid-name





                clipped_grads = tf.vectorized_map(process_microbatch, microbatch_losses)


                final_grads = tf.nest.map_structure(reduce_noise_normalize_batch,
                                                    clipped_grads)

                return list(zip(final_grads, var_list))

        def apply_gradients(self, grads_and_vars, global_step=None, name=None):
            # pylint: disable=g-doc-args, g-doc-return-or-yield
            """DP-SGD version of base class method."""
            assert self._was_compute_gradients_called, (
                'compute_gradients() on the differentially private optimizer was not'
                ' called. Which means that the training is not differentially '
                'private. It happens for example in Keras training in TensorFlow '
                '2.0+.')
            return super(DPOptimizerClass, self).apply_gradients(
                grads_and_vars=grads_and_vars, global_step=global_step, name=name)

    return DPOptimizerClass


VectorizedDPAdagradOptimizer = make_vectorized_optimizer_class(AdagradOptimizer)
VectorizedDPAdamOptimizer = make_vectorized_optimizer_class(AdamOptimizer)
VectorizedDPSGDOptimizer = make_vectorized_optimizer_class(
    GradientDescentOptimizer)

VectorizedDPAdagrad = VectorizedDPAdagradOptimizer
VectorizedDPAdam = VectorizedDPAdamOptimizer
VectorizedDPSGD = VectorizedDPSGDOptimizer