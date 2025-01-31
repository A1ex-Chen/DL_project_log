#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2021, Amazon Web Services. All rights reserved.
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

from sagemakercv.utils.runner.hooks import Hook
from sagemakercv.utils.dist_utils import master_only
import tensorflow as tf
from ..builder import HOOKS

class GraphVisualizer(Hook):
    
    def __init__(self, step=1):
        self.step = step
    
    @master_only
    def before_train_iter(self, runner):
        if runner.iter==self.step:
            tf.summary.trace_on(graph=True, profiler=True)
    
    @master_only
    def after_train_iter(self, runner):
        if runner.iter==self.step:
            writer = tf.summary.create_file_writer(runner.tensorboard_dir)
            with writer.as_default():
                tf.summary.trace_export(
                      name="graph_trace",
                      step=runner.iter,
                      profiler_outdir=runner.work_dir)
            writer.close()

@HOOKS.register("GraphVisualizer")
    
    @master_only
    
    @master_only

@HOOKS.register("GraphVisualizer")
def build_graph_visualizer(cfg):
    return GraphVisualizer()