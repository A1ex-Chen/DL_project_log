import os
import shutil
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from multiprocessing import Event
from typing import List, Optional

import nvidia_smi
import torch
from PIL import Image
from graphviz import Digraph
from pynvml import NVMLError
from torch import optim, Size
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import to_tensor

from .algorithm import Algorithm
from .checkpoint import Checkpoint
from .config import Config
from .dataset import Dataset, ConcatDataset
from .db import DB
from .evaluator import Evaluator
from .model import Model
from .preprocessor import Preprocessor
from ...augmenter import Augmenter
from ...extension.data_parallel import BunchDataParallel, Bunch
from ...extension.lr_scheduler import WarmUpMultiStepLR
from ...logger import Logger
from ...plotter import Plotter
from ...util import Util


class Trainer:

    class Callback:

        @dataclass
        class CheckpointInfo:
            epoch: int
            avg_loss: float
            mean_ap: float

            @staticmethod
            def sorted(checkpoint_infos: List['Trainer.Callback.CheckpointInfo']) -> List['Trainer.Callback.CheckpointInfo']:
                checkpoint_infos = sorted(checkpoint_infos, key=lambda x: x.avg_loss)  # from lower to higher
                checkpoint_infos = sorted(checkpoint_infos, key=lambda x: x.mean_ap, reverse=True)  # from higher to lower
                return checkpoint_infos

        def __init__(self, logger: Logger, config: Config, model: Model,
                     optimizer: Optimizer, scheduler: _LRScheduler, terminator: Event,
                     val_evaluator: Evaluator, test_evaluator: Optional[Evaluator],
                     db: DB, path_to_checkpoints_dir: str):
            super().__init__()
            self.logger = logger
            self.config = config
            self.model = model
            self.optimizer = optimizer
            self.scheduler = scheduler
            self.terminator = terminator
            self.val_evaluator = val_evaluator
            self.test_evaluator = test_evaluator
            self.db = db
            self.path_to_checkpoints_dir = path_to_checkpoints_dir

            self.summary_writer = SummaryWriter(log_dir=os.path.join(path_to_checkpoints_dir, 'tensorboard'))
            self.losses = deque(maxlen=100)
            self.anchor_objectness_losses = deque(maxlen=100)
            self.anchor_transformer_losses = deque(maxlen=100)
            self.proposal_class_losses = deque(maxlen=100)
            self.proposal_transformer_losses = deque(maxlen=100)
            self.mask_losses = deque(maxlen=100)
            self.time_checkpoint = None
            self.batches_counter = None

            self.best_epoch = None
            self.best_mean_ap = -1
            self.checkpoint_infos = []

            checkpoints = db.select_checkpoint_table()

            for available_checkpoint in [checkpoint for checkpoint in checkpoints if checkpoint.is_available]:
                assert available_checkpoint.metrics.specific['categories'][0] == 'mean'
                checkpoint_info = Trainer.Callback.CheckpointInfo(available_checkpoint.epoch,
                                                                  available_checkpoint.avg_loss,
                                                                  available_checkpoint.metrics.specific['aps'][0])
                self.checkpoint_infos.append(checkpoint_info)

            best_checkpoints = [checkpoint for checkpoint in checkpoints if checkpoint.is_best]
            assert len(best_checkpoints) <= 1
            if len(best_checkpoints) > 0:
                best_checkpoint = best_checkpoints[0]
                assert best_checkpoint.metrics.specific['categories'][0] == 'mean'
                self.best_epoch = best_checkpoint.epoch
                self.best_mean_ap = best_checkpoint.metrics.specific['aps'][0]

            self.profile_enabled = True
            try:
                nvidia_smi.nvmlInit()
                self.global_device_count = nvidia_smi.nvmlDeviceGetCount()  # not affected by visible devices
                self.global_device_id_to_handle_dict = {i: nvidia_smi.nvmlDeviceGetHandleByIndex(i)
                                                        for i in range(self.global_device_count)}
            except NVMLError as e:
                self.logger.w(f'Disable profiling due to failure of NVML initialization with reason: {str(e)}')
                self.profile_enabled = False

        def save_model_graph(self, image_shape: Size):
            graph = Digraph()

            graph.node(name='image', label=f'Image\n{str(tuple(image_shape))}', shape='box', style='filled', fillcolor='#ffffff')

            try:
                subgraph, input_node_name, output_node_name = self.model.algorithm.make_graph()
                graph.subgraph(subgraph)
            except NotImplementedError:
                with graph.subgraph(name='cluster_model') as c:
                    c.attr(label=f'Model', style='filled', color='lightgray')
                    c.node(name='net', label=f'{self.config.algorithm_name.value}', shape='box', style='filled', fillcolor='#ffffff')
                    input_node_name = output_node_name = 'net'

            graph.node(name='output', label=f'Output\n{str((self.model.num_classes,))}', shape='box', style='filled', fillcolor='#ffffff')

            graph.edge('image', input_node_name)
            graph.edge(output_node_name, 'output')

            graph.render(filename='model-graph', directory=self.path_to_checkpoints_dir, format='png', cleanup=True)

        def on_train_begin(self, num_batches_in_epoch: int):
            self.db.insert_log_table(DB.Log(
                global_batch=0, status=DB.Log.Status.INITIALIZED, datetime=int(time.time()),
                epoch=0, total_epoch=self.config.num_epochs_to_finish,
                batch=0, total_batch=num_batches_in_epoch,
                avg_loss=-1,
                learning_rate=-1, samples_per_sec=-1,
                eta_hrs=-1
            ))

        def on_epoch_begin(self, epoch: int, num_batches_in_epoch: int):
            self.time_checkpoint = time.time()
            self.batches_counter = 0

        def on_batch_begin(self, epoch: int, num_batches_in_epoch: int, n_batch: int):
            pass

        def on_batch_end(self, epoch: int, num_batches_in_epoch: int, n_batch: int,
                         loss: float,
                         anchor_objectness_loss: float, anchor_transformer_loss: float,
                         proposal_class_loss: float, proposal_transformer_loss: float,
                         mask_loss: float):
            global_batch = (epoch - 1) * num_batches_in_epoch + n_batch
            lr = self.scheduler.get_lr()[0]
            self.losses.append(loss)
            self.anchor_objectness_losses.append(anchor_objectness_loss)
            self.anchor_transformer_losses.append(anchor_transformer_loss)
            self.proposal_class_losses.append(proposal_class_loss)
            self.proposal_transformer_losses.append(proposal_transformer_loss)
            self.mask_losses.append(mask_loss)
            self.batches_counter += 1

            self.summary_writer.add_scalar('loss/loss', loss, global_batch)
            self.summary_writer.add_scalar('learning_rate', lr, global_batch)

            if n_batch % self.config.num_batches_to_display == 0 or n_batch == num_batches_in_epoch:
                elapsed_time = time.time() - self.time_checkpoint
                num_batches_per_sec = self.batches_counter / elapsed_time
                num_samples_per_sec = num_batches_per_sec * self.config.batch_size
                eta = ((self.config.num_epochs_to_finish - epoch) * num_batches_in_epoch + num_batches_in_epoch - n_batch) / num_batches_per_sec / 3600
                avg_loss = sum(self.losses) / len(self.losses)
                avg_anchor_objectness_loss = sum(self.anchor_objectness_losses) / len(self.anchor_objectness_losses)
                avg_anchor_transformer_loss = sum(self.anchor_transformer_losses) / len(self.anchor_transformer_losses)
                avg_proposal_class_loss = sum(self.proposal_class_losses) / len(self.proposal_class_losses)
                avg_proposal_transformer_loss = sum(self.proposal_transformer_losses) / len(self.proposal_transformer_losses)
                avg_mask_loss = sum(self.mask_losses) / len(self.mask_losses)

                self.db.insert_log_table(DB.Log(
                    global_batch, status=DB.Log.Status.RUNNING, datetime=int(time.time()),
                    epoch=epoch, total_epoch=self.config.num_epochs_to_finish,
                    batch=n_batch, total_batch=num_batches_in_epoch,
                    avg_loss=avg_loss,
                    learning_rate=lr, samples_per_sec=num_samples_per_sec, eta_hrs=eta
                ))
                self.db.insert_instance_segmentation_log_table(DB.InstanceSegmentationLog(
                    avg_anchor_objectness_loss=avg_anchor_objectness_loss,
                    avg_anchor_transformer_loss=avg_anchor_transformer_loss,
                    avg_proposal_class_loss=avg_proposal_class_loss,
                    avg_proposal_transformer_loss=avg_proposal_transformer_loss,
                    avg_mask_loss=avg_mask_loss
                ))
                self.logger.i(
                    f'[Epoch ({epoch}/{self.config.num_epochs_to_finish}) Batch ({n_batch}/{num_batches_in_epoch})] '
                    f'Avg. Loss = {avg_loss:.6f}, Learning Rate = {lr:.8f} '
                    f'({num_samples_per_sec:.2f} samples/sec; ETA {eta:.2f} hrs)'
                )

                if self.profile_enabled:
                    self.summary_writer.add_scalar('profile/num_samples_per_sec', num_samples_per_sec, global_batch)
                    for i, handle in self.global_device_id_to_handle_dict.items():
                        device_util_rates = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
                        self.summary_writer.add_scalar(f'profile/device_usage/{i}', device_util_rates.gpu, global_batch)
                        self.summary_writer.add_scalar(f'profile/device_memory/{i}', device_util_rates.memory, global_batch)

                self.time_checkpoint = time.time()
                self.batches_counter = 0

        def on_epoch_end(self, epoch: int, num_batches_in_epoch: int):
            if epoch % self.config.num_epochs_to_validate == 0 or \
                    epoch in [self.config.step_lr_sizes] or \
                    epoch == self.config.num_epochs_to_finish:
                global_batch = epoch * num_batches_in_epoch

                # region ===== save the model =====
                path_to_checkpoint = os.path.join(self.path_to_checkpoints_dir, f'epoch-{epoch:06d}', 'checkpoint.pth')
                os.makedirs(os.path.dirname(path_to_checkpoint), exist_ok=True)
                Checkpoint.save(Checkpoint(epoch, self.model, self.optimizer), path_to_checkpoint)
                self.logger.i(f'Model has been saved to {path_to_checkpoint}')
                # endregion =======================

                # region ===== evaluate the model =====
                self.logger.i('Start evaluating for validation set')
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                checkpoint = Checkpoint.load(path_to_checkpoint, device)
                # NOTE: Evaluate newly created model to keep the original model stay in training mode
                evaluation = self.val_evaluator.evaluate(checkpoint.model)
                mean_ap = evaluation.metric_ap.mean_value
                self.logger.i(f'mean AP = {mean_ap:.4f}')
                self.summary_writer.add_scalar('mean_ap/val', mean_ap, global_batch)
                # endregion ===========================

                # region ===== insert into DB =====
                avg_loss = sum(self.losses) / len(self.losses)
                categories = ['mean']
                aps = [evaluation.metric_ap.mean_value]

                for cls in range(1, self.model.num_classes):
                    categories.append(self.model.class_to_category_dict[cls])
                    aps.append(evaluation.metric_ap.class_to_value_dict[cls])

                metrics = DB.Checkpoint.Metrics(
                    DB.Checkpoint.Metrics.Overall(),
                    DB.Checkpoint.Metrics.Specific(categories, aps)
                )
                self.db.insert_checkpoint_table(DB.Checkpoint(epoch, avg_loss, metrics))
                # endregion =======================

                # region ===== limit number of models =====
                if len(self.checkpoint_infos) > self.config.max_num_checkpoints - 1:
                    reserved_checkpoint_infos = [checkpoint_info for checkpoint_info in self.checkpoint_infos
                                                 if checkpoint_info.epoch in self.config.step_lr_sizes
                                                 or checkpoint_info.epoch == self.config.num_epochs_to_finish]
                    reserved_epochs = [checkpoint_info.epoch for checkpoint_info in reserved_checkpoint_infos]

                    self.checkpoint_infos = Trainer.Callback.CheckpointInfo.sorted(
                        [checkpoint_info for checkpoint_info in self.checkpoint_infos
                         if checkpoint_info.epoch not in reserved_epochs]
                    )

                    removing_checkpoint_info: Trainer.Callback.CheckpointInfo = self.checkpoint_infos.pop()
                    removing_epoch = removing_checkpoint_info.epoch
                    path_to_removing_epoch_dir = os.path.join(self.path_to_checkpoints_dir, f'epoch-{removing_epoch:06d}')
                    shutil.rmtree(path_to_removing_epoch_dir)
                    self.db.update_checkpoint_table_is_available_for_epoch(is_available=False, epoch=removing_checkpoint_info.epoch)

                    self.checkpoint_infos.extend(reserved_checkpoint_infos)

                self.checkpoint_infos.append(self.CheckpointInfo(epoch, avg_loss, mean_ap))
                # endregion ===============================

                # region ===== update best model =====
                if mean_ap >= self.best_mean_ap:
                    last_best_epoch = self.best_epoch
                    if last_best_epoch is not None:
                        self.db.update_checkpoint_table_is_best_for_epoch(is_best=False, epoch=last_best_epoch)
                    self.db.update_checkpoint_table_is_best_for_epoch(is_best=True, epoch=epoch)
                    self.best_mean_ap = mean_ap
                    self.best_epoch = epoch
                self.logger.i(f'best mean AP = {self.best_mean_ap:.4f} at epoch {self.best_epoch}')
                # endregion ==========================

                # region ===== make plots =====
                path_to_epoch_dir = os.path.join(self.path_to_checkpoints_dir, f'epoch-{epoch:06d}')
                path_to_plot_dir = os.path.join(path_to_epoch_dir,
                                                'quality-{:s}'.format(evaluation.quality.value),
                                                'size-{:s}'.format(evaluation.size.value))
                os.makedirs(path_to_plot_dir)

                path_to_plot = os.path.join(path_to_plot_dir, 'metric-ap.png')
                Plotter.plot_pr_curve(self.model.num_classes,
                                      self.model.class_to_category_dict,
                                      mean_ap=evaluation.metric_ap.mean_value,
                                      class_to_ap_dict=evaluation.metric_ap.class_to_value_dict,
                                      class_to_inter_recall_array_dict=evaluation.class_to_inter_recall_array_dict,
                                      class_to_inter_precision_array_dict=evaluation.class_to_inter_precision_array_dict,
                                      path_to_plot=path_to_plot)
                self.summary_writer.add_image('pr_curve/val', to_tensor(Image.open(path_to_plot)), global_batch)

                for target_class in range(1, self.model.num_classes):
                    Plotter.plot_thresh_vs_pr_bar(self.model.num_classes,
                                                  self.model.class_to_category_dict,
                                                  class_to_ap_dict=evaluation.metric_ap.class_to_value_dict,
                                                  class_to_recall_array_dict=evaluation.class_to_recall_array_dict,
                                                  class_to_precision_array_dict=evaluation.class_to_precision_array_dict,
                                                  class_to_f1_score_array_dict=evaluation.class_to_f1_score_array_dict,
                                                  class_to_prob_array_dict=evaluation.class_to_prob_array_dict,
                                                  path_to_placeholder_to_plot=os.path.join(path_to_plot_dir, 'thresh-{}.png'))
                # endregion ===================

        def on_train_end(self, num_batches_in_epoch: int):
            path_to_best_checkpoint = os.path.join(self.path_to_checkpoints_dir, f'epoch-{self.best_epoch:06d}', 'checkpoint.pth')
            self.logger.i(f'The best model is {path_to_best_checkpoint}')

            metric_dict = {
                'hparam/best_epoch': self.best_epoch,
                'hparam/val_mean_ap': self.best_mean_ap
            }

            if self.test_evaluator is not None:
                global_batch = self.best_epoch * num_batches_in_epoch

                self.logger.i('Start evaluating for test set')
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                checkpoint = Checkpoint.load(path_to_best_checkpoint, device)
                evaluation = self.test_evaluator.evaluate(checkpoint.model)
                test_mean_ap = evaluation.metric_ap.mean_value
                self.logger.i(f'mean AP = {test_mean_ap:.4f} at epoch {self.best_epoch}')
                self.summary_writer.add_scalar('mean_ap/test', test_mean_ap, global_batch)

                with tempfile.TemporaryDirectory() as path_to_temp_dir:
                    path_to_plot = os.path.join(path_to_temp_dir, 'metric-ap.png')
                    Plotter.plot_pr_curve(self.model.num_classes,
                                          self.model.class_to_category_dict,
                                          mean_ap=test_mean_ap,
                                          class_to_ap_dict=evaluation.metric_ap.class_to_value_dict,
                                          class_to_inter_recall_array_dict=evaluation.class_to_inter_recall_array_dict,
                                          class_to_inter_precision_array_dict=evaluation.class_to_inter_precision_array_dict,
                                          path_to_plot=path_to_plot)
                    self.summary_writer.add_image('pr_curve/test', to_tensor(Image.open(path_to_plot)), global_batch)

                metric_dict.update({
                    'hparam/test_mean_ap': test_mean_ap
                })

            logs = self.db.select_log_table()
            instance_segmentation_logs = self.db.select_instance_segmentation_log_table()
            global_batches = [log.global_batch for log in logs if log.epoch > 0]
            losses = [log.avg_loss for log in logs if log.epoch > 0]
            anchor_objectness_losses = [log.avg_anchor_objectness_loss for log in instance_segmentation_logs]
            anchor_transformer_losses = [log.avg_anchor_transformer_loss for log in instance_segmentation_logs]
            proposal_class_losses = [log.avg_proposal_class_loss for log in instance_segmentation_logs]
            proposal_transformer_losses = [log.avg_proposal_transformer_loss for log in instance_segmentation_logs]
            mask_losses = [log.avg_mask_loss for log in instance_segmentation_logs]

            legend_to_losses_and_color_dict = {
                'loss': (losses, 'orange'),
                'anchor objectness loss': (anchor_objectness_losses, 'b--'),
                'anchor transformer loss': (anchor_transformer_losses, 'g--'),
                'proposal class loss': (proposal_class_losses, 'c--'),
                'proposal transformer loss': (proposal_transformer_losses, 'm--'),
                'mask loss': (mask_losses, 'r--')
            }
            path_to_loss_plot = os.path.join(self.path_to_checkpoints_dir, 'loss.png')
            Plotter.plot_loss_curve(global_batches,
                                    legend_to_losses_and_color_dict,
                                    path_to_loss_plot)

            status = DB.Log.Status.STOPPED if self.is_terminated() else DB.Log.Status.FINISHED
            self.db.update_log_table_latest_status(status)

            self.summary_writer.add_hparams(
                hparam_dict=self.config.to_hyper_param_dict(),
                metric_dict=metric_dict
            )
            self.summary_writer.close()

        def is_terminated(self) -> bool:
            return self.terminator.is_set()



    @staticmethod

    @staticmethod






                # endregion ===================



        def __init__(self, logger: Logger, config: Config, model: Model,
                     optimizer: Optimizer, scheduler: _LRScheduler, terminator: Event,
                     val_evaluator: Evaluator, test_evaluator: Optional[Evaluator],
                     db: DB, path_to_checkpoints_dir: str):
            super().__init__()
            self.logger = logger
            self.config = config
            self.model = model
            self.optimizer = optimizer
            self.scheduler = scheduler
            self.terminator = terminator
            self.val_evaluator = val_evaluator
            self.test_evaluator = test_evaluator
            self.db = db
            self.path_to_checkpoints_dir = path_to_checkpoints_dir

            self.summary_writer = SummaryWriter(log_dir=os.path.join(path_to_checkpoints_dir, 'tensorboard'))
            self.losses = deque(maxlen=100)
            self.anchor_objectness_losses = deque(maxlen=100)
            self.anchor_transformer_losses = deque(maxlen=100)
            self.proposal_class_losses = deque(maxlen=100)
            self.proposal_transformer_losses = deque(maxlen=100)
            self.mask_losses = deque(maxlen=100)
            self.time_checkpoint = None
            self.batches_counter = None

            self.best_epoch = None
            self.best_mean_ap = -1
            self.checkpoint_infos = []

            checkpoints = db.select_checkpoint_table()

            for available_checkpoint in [checkpoint for checkpoint in checkpoints if checkpoint.is_available]:
                assert available_checkpoint.metrics.specific['categories'][0] == 'mean'
                checkpoint_info = Trainer.Callback.CheckpointInfo(available_checkpoint.epoch,
                                                                  available_checkpoint.avg_loss,
                                                                  available_checkpoint.metrics.specific['aps'][0])
                self.checkpoint_infos.append(checkpoint_info)

            best_checkpoints = [checkpoint for checkpoint in checkpoints if checkpoint.is_best]
            assert len(best_checkpoints) <= 1
            if len(best_checkpoints) > 0:
                best_checkpoint = best_checkpoints[0]
                assert best_checkpoint.metrics.specific['categories'][0] == 'mean'
                self.best_epoch = best_checkpoint.epoch
                self.best_mean_ap = best_checkpoint.metrics.specific['aps'][0]

            self.profile_enabled = True
            try:
                nvidia_smi.nvmlInit()
                self.global_device_count = nvidia_smi.nvmlDeviceGetCount()  # not affected by visible devices
                self.global_device_id_to_handle_dict = {i: nvidia_smi.nvmlDeviceGetHandleByIndex(i)
                                                        for i in range(self.global_device_count)}
            except NVMLError as e:
                self.logger.w(f'Disable profiling due to failure of NVML initialization with reason: {str(e)}')
                self.profile_enabled = False

        def save_model_graph(self, image_shape: Size):
            graph = Digraph()

            graph.node(name='image', label=f'Image\n{str(tuple(image_shape))}', shape='box', style='filled', fillcolor='#ffffff')

            try:
                subgraph, input_node_name, output_node_name = self.model.algorithm.make_graph()
                graph.subgraph(subgraph)
            except NotImplementedError:
                with graph.subgraph(name='cluster_model') as c:
                    c.attr(label=f'Model', style='filled', color='lightgray')
                    c.node(name='net', label=f'{self.config.algorithm_name.value}', shape='box', style='filled', fillcolor='#ffffff')
                    input_node_name = output_node_name = 'net'

            graph.node(name='output', label=f'Output\n{str((self.model.num_classes,))}', shape='box', style='filled', fillcolor='#ffffff')

            graph.edge('image', input_node_name)
            graph.edge(output_node_name, 'output')

            graph.render(filename='model-graph', directory=self.path_to_checkpoints_dir, format='png', cleanup=True)

        def on_train_begin(self, num_batches_in_epoch: int):
            self.db.insert_log_table(DB.Log(
                global_batch=0, status=DB.Log.Status.INITIALIZED, datetime=int(time.time()),
                epoch=0, total_epoch=self.config.num_epochs_to_finish,
                batch=0, total_batch=num_batches_in_epoch,
                avg_loss=-1,
                learning_rate=-1, samples_per_sec=-1,
                eta_hrs=-1
            ))

        def on_epoch_begin(self, epoch: int, num_batches_in_epoch: int):
            self.time_checkpoint = time.time()
            self.batches_counter = 0

        def on_batch_begin(self, epoch: int, num_batches_in_epoch: int, n_batch: int):
            pass

        def on_batch_end(self, epoch: int, num_batches_in_epoch: int, n_batch: int,
                         loss: float,
                         anchor_objectness_loss: float, anchor_transformer_loss: float,
                         proposal_class_loss: float, proposal_transformer_loss: float,
                         mask_loss: float):
            global_batch = (epoch - 1) * num_batches_in_epoch + n_batch
            lr = self.scheduler.get_lr()[0]
            self.losses.append(loss)
            self.anchor_objectness_losses.append(anchor_objectness_loss)
            self.anchor_transformer_losses.append(anchor_transformer_loss)
            self.proposal_class_losses.append(proposal_class_loss)
            self.proposal_transformer_losses.append(proposal_transformer_loss)
            self.mask_losses.append(mask_loss)
            self.batches_counter += 1

            self.summary_writer.add_scalar('loss/loss', loss, global_batch)
            self.summary_writer.add_scalar('learning_rate', lr, global_batch)

            if n_batch % self.config.num_batches_to_display == 0 or n_batch == num_batches_in_epoch:
                elapsed_time = time.time() - self.time_checkpoint
                num_batches_per_sec = self.batches_counter / elapsed_time
                num_samples_per_sec = num_batches_per_sec * self.config.batch_size
                eta = ((self.config.num_epochs_to_finish - epoch) * num_batches_in_epoch + num_batches_in_epoch - n_batch) / num_batches_per_sec / 3600
                avg_loss = sum(self.losses) / len(self.losses)
                avg_anchor_objectness_loss = sum(self.anchor_objectness_losses) / len(self.anchor_objectness_losses)
                avg_anchor_transformer_loss = sum(self.anchor_transformer_losses) / len(self.anchor_transformer_losses)
                avg_proposal_class_loss = sum(self.proposal_class_losses) / len(self.proposal_class_losses)
                avg_proposal_transformer_loss = sum(self.proposal_transformer_losses) / len(self.proposal_transformer_losses)
                avg_mask_loss = sum(self.mask_losses) / len(self.mask_losses)

                self.db.insert_log_table(DB.Log(
                    global_batch, status=DB.Log.Status.RUNNING, datetime=int(time.time()),
                    epoch=epoch, total_epoch=self.config.num_epochs_to_finish,
                    batch=n_batch, total_batch=num_batches_in_epoch,
                    avg_loss=avg_loss,
                    learning_rate=lr, samples_per_sec=num_samples_per_sec, eta_hrs=eta
                ))
                self.db.insert_instance_segmentation_log_table(DB.InstanceSegmentationLog(
                    avg_anchor_objectness_loss=avg_anchor_objectness_loss,
                    avg_anchor_transformer_loss=avg_anchor_transformer_loss,
                    avg_proposal_class_loss=avg_proposal_class_loss,
                    avg_proposal_transformer_loss=avg_proposal_transformer_loss,
                    avg_mask_loss=avg_mask_loss
                ))
                self.logger.i(
                    f'[Epoch ({epoch}/{self.config.num_epochs_to_finish}) Batch ({n_batch}/{num_batches_in_epoch})] '
                    f'Avg. Loss = {avg_loss:.6f}, Learning Rate = {lr:.8f} '
                    f'({num_samples_per_sec:.2f} samples/sec; ETA {eta:.2f} hrs)'
                )

                if self.profile_enabled:
                    self.summary_writer.add_scalar('profile/num_samples_per_sec', num_samples_per_sec, global_batch)
                    for i, handle in self.global_device_id_to_handle_dict.items():
                        device_util_rates = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
                        self.summary_writer.add_scalar(f'profile/device_usage/{i}', device_util_rates.gpu, global_batch)
                        self.summary_writer.add_scalar(f'profile/device_memory/{i}', device_util_rates.memory, global_batch)

                self.time_checkpoint = time.time()
                self.batches_counter = 0

        def on_epoch_end(self, epoch: int, num_batches_in_epoch: int):
            if epoch % self.config.num_epochs_to_validate == 0 or \
                    epoch in [self.config.step_lr_sizes] or \
                    epoch == self.config.num_epochs_to_finish:
                global_batch = epoch * num_batches_in_epoch

                # region ===== save the model =====
                path_to_checkpoint = os.path.join(self.path_to_checkpoints_dir, f'epoch-{epoch:06d}', 'checkpoint.pth')
                os.makedirs(os.path.dirname(path_to_checkpoint), exist_ok=True)
                Checkpoint.save(Checkpoint(epoch, self.model, self.optimizer), path_to_checkpoint)
                self.logger.i(f'Model has been saved to {path_to_checkpoint}')
                # endregion =======================

                # region ===== evaluate the model =====
                self.logger.i('Start evaluating for validation set')
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                checkpoint = Checkpoint.load(path_to_checkpoint, device)
                # NOTE: Evaluate newly created model to keep the original model stay in training mode
                evaluation = self.val_evaluator.evaluate(checkpoint.model)
                mean_ap = evaluation.metric_ap.mean_value
                self.logger.i(f'mean AP = {mean_ap:.4f}')
                self.summary_writer.add_scalar('mean_ap/val', mean_ap, global_batch)
                # endregion ===========================

                # region ===== insert into DB =====
                avg_loss = sum(self.losses) / len(self.losses)
                categories = ['mean']
                aps = [evaluation.metric_ap.mean_value]

                for cls in range(1, self.model.num_classes):
                    categories.append(self.model.class_to_category_dict[cls])
                    aps.append(evaluation.metric_ap.class_to_value_dict[cls])

                metrics = DB.Checkpoint.Metrics(
                    DB.Checkpoint.Metrics.Overall(),
                    DB.Checkpoint.Metrics.Specific(categories, aps)
                )
                self.db.insert_checkpoint_table(DB.Checkpoint(epoch, avg_loss, metrics))
                # endregion =======================

                # region ===== limit number of models =====
                if len(self.checkpoint_infos) > self.config.max_num_checkpoints - 1:
                    reserved_checkpoint_infos = [checkpoint_info for checkpoint_info in self.checkpoint_infos
                                                 if checkpoint_info.epoch in self.config.step_lr_sizes
                                                 or checkpoint_info.epoch == self.config.num_epochs_to_finish]
                    reserved_epochs = [checkpoint_info.epoch for checkpoint_info in reserved_checkpoint_infos]

                    self.checkpoint_infos = Trainer.Callback.CheckpointInfo.sorted(
                        [checkpoint_info for checkpoint_info in self.checkpoint_infos
                         if checkpoint_info.epoch not in reserved_epochs]
                    )

                    removing_checkpoint_info: Trainer.Callback.CheckpointInfo = self.checkpoint_infos.pop()
                    removing_epoch = removing_checkpoint_info.epoch
                    path_to_removing_epoch_dir = os.path.join(self.path_to_checkpoints_dir, f'epoch-{removing_epoch:06d}')
                    shutil.rmtree(path_to_removing_epoch_dir)
                    self.db.update_checkpoint_table_is_available_for_epoch(is_available=False, epoch=removing_checkpoint_info.epoch)

                    self.checkpoint_infos.extend(reserved_checkpoint_infos)

                self.checkpoint_infos.append(self.CheckpointInfo(epoch, avg_loss, mean_ap))
                # endregion ===============================

                # region ===== update best model =====
                if mean_ap >= self.best_mean_ap:
                    last_best_epoch = self.best_epoch
                    if last_best_epoch is not None:
                        self.db.update_checkpoint_table_is_best_for_epoch(is_best=False, epoch=last_best_epoch)
                    self.db.update_checkpoint_table_is_best_for_epoch(is_best=True, epoch=epoch)
                    self.best_mean_ap = mean_ap
                    self.best_epoch = epoch
                self.logger.i(f'best mean AP = {self.best_mean_ap:.4f} at epoch {self.best_epoch}')
                # endregion ==========================

                # region ===== make plots =====
                path_to_epoch_dir = os.path.join(self.path_to_checkpoints_dir, f'epoch-{epoch:06d}')
                path_to_plot_dir = os.path.join(path_to_epoch_dir,
                                                'quality-{:s}'.format(evaluation.quality.value),
                                                'size-{:s}'.format(evaluation.size.value))
                os.makedirs(path_to_plot_dir)

                path_to_plot = os.path.join(path_to_plot_dir, 'metric-ap.png')
                Plotter.plot_pr_curve(self.model.num_classes,
                                      self.model.class_to_category_dict,
                                      mean_ap=evaluation.metric_ap.mean_value,
                                      class_to_ap_dict=evaluation.metric_ap.class_to_value_dict,
                                      class_to_inter_recall_array_dict=evaluation.class_to_inter_recall_array_dict,
                                      class_to_inter_precision_array_dict=evaluation.class_to_inter_precision_array_dict,
                                      path_to_plot=path_to_plot)
                self.summary_writer.add_image('pr_curve/val', to_tensor(Image.open(path_to_plot)), global_batch)

                for target_class in range(1, self.model.num_classes):
                    Plotter.plot_thresh_vs_pr_bar(self.model.num_classes,
                                                  self.model.class_to_category_dict,
                                                  class_to_ap_dict=evaluation.metric_ap.class_to_value_dict,
                                                  class_to_recall_array_dict=evaluation.class_to_recall_array_dict,
                                                  class_to_precision_array_dict=evaluation.class_to_precision_array_dict,
                                                  class_to_f1_score_array_dict=evaluation.class_to_f1_score_array_dict,
                                                  class_to_prob_array_dict=evaluation.class_to_prob_array_dict,
                                                  path_to_placeholder_to_plot=os.path.join(path_to_plot_dir, 'thresh-{}.png'))
                # endregion ===================

        def on_train_end(self, num_batches_in_epoch: int):
            path_to_best_checkpoint = os.path.join(self.path_to_checkpoints_dir, f'epoch-{self.best_epoch:06d}', 'checkpoint.pth')
            self.logger.i(f'The best model is {path_to_best_checkpoint}')

            metric_dict = {
                'hparam/best_epoch': self.best_epoch,
                'hparam/val_mean_ap': self.best_mean_ap
            }

            if self.test_evaluator is not None:
                global_batch = self.best_epoch * num_batches_in_epoch

                self.logger.i('Start evaluating for test set')
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                checkpoint = Checkpoint.load(path_to_best_checkpoint, device)
                evaluation = self.test_evaluator.evaluate(checkpoint.model)
                test_mean_ap = evaluation.metric_ap.mean_value
                self.logger.i(f'mean AP = {test_mean_ap:.4f} at epoch {self.best_epoch}')
                self.summary_writer.add_scalar('mean_ap/test', test_mean_ap, global_batch)

                with tempfile.TemporaryDirectory() as path_to_temp_dir:
                    path_to_plot = os.path.join(path_to_temp_dir, 'metric-ap.png')
                    Plotter.plot_pr_curve(self.model.num_classes,
                                          self.model.class_to_category_dict,
                                          mean_ap=test_mean_ap,
                                          class_to_ap_dict=evaluation.metric_ap.class_to_value_dict,
                                          class_to_inter_recall_array_dict=evaluation.class_to_inter_recall_array_dict,
                                          class_to_inter_precision_array_dict=evaluation.class_to_inter_precision_array_dict,
                                          path_to_plot=path_to_plot)
                    self.summary_writer.add_image('pr_curve/test', to_tensor(Image.open(path_to_plot)), global_batch)

                metric_dict.update({
                    'hparam/test_mean_ap': test_mean_ap
                })

            logs = self.db.select_log_table()
            instance_segmentation_logs = self.db.select_instance_segmentation_log_table()
            global_batches = [log.global_batch for log in logs if log.epoch > 0]
            losses = [log.avg_loss for log in logs if log.epoch > 0]
            anchor_objectness_losses = [log.avg_anchor_objectness_loss for log in instance_segmentation_logs]
            anchor_transformer_losses = [log.avg_anchor_transformer_loss for log in instance_segmentation_logs]
            proposal_class_losses = [log.avg_proposal_class_loss for log in instance_segmentation_logs]
            proposal_transformer_losses = [log.avg_proposal_transformer_loss for log in instance_segmentation_logs]
            mask_losses = [log.avg_mask_loss for log in instance_segmentation_logs]

            legend_to_losses_and_color_dict = {
                'loss': (losses, 'orange'),
                'anchor objectness loss': (anchor_objectness_losses, 'b--'),
                'anchor transformer loss': (anchor_transformer_losses, 'g--'),
                'proposal class loss': (proposal_class_losses, 'c--'),
                'proposal transformer loss': (proposal_transformer_losses, 'm--'),
                'mask loss': (mask_losses, 'r--')
            }
            path_to_loss_plot = os.path.join(self.path_to_checkpoints_dir, 'loss.png')
            Plotter.plot_loss_curve(global_batches,
                                    legend_to_losses_and_color_dict,
                                    path_to_loss_plot)

            status = DB.Log.Status.STOPPED if self.is_terminated() else DB.Log.Status.FINISHED
            self.db.update_log_table_latest_status(status)

            self.summary_writer.add_hparams(
                hparam_dict=self.config.to_hyper_param_dict(),
                metric_dict=metric_dict
            )
            self.summary_writer.close()

        def is_terminated(self) -> bool:
            return self.terminator.is_set()

    def __init__(self, config: Config, logger: Logger, augmenter: Augmenter,
                 device: torch.device, device_count: int,
                 db: DB, terminator: Event,
                 path_to_checkpoints_dir: str):
        algorithm_class = Algorithm.from_name(config.algorithm_name)

        preprocessor = Preprocessor(
            config.image_resized_width, config.image_resized_height,
            config.image_min_side, config.image_max_side,
            config.image_side_divisor
        )

        # region ===== Setup data source =====
        self.train_dataset = ConcatDataset(
            # NOTE: It seems that the get `Item` from dataset sometimes causes an error like
            #       "unable to open shared memory object", hence we get `ItemTuple` instead
            [Dataset(path_to_data_dir, Dataset.Mode.TRAIN, preprocessor, augmenter, exclude_difficulty=True, returns_item_tuple=True)
             for path_to_data_dir in [config.path_to_data_dir] + list(config.path_to_extra_data_dirs)]
        )
        num_train_data = len(self.train_dataset)
        logger.i('Found {:d} training samples'.format(num_train_data))
        assert num_train_data > 0

        self.val_dataset = ConcatDataset(
            [Dataset(path_to_data_dir, Dataset.Mode.VAL, preprocessor, augmenter=None, exclude_difficulty=False)
             for path_to_data_dir in [config.path_to_data_dir] + list(config.path_to_extra_data_dirs)]
        )
        num_val_data = len(self.val_dataset)
        logger.i('Found {:d} validation samples'.format(num_val_data))
        assert num_val_data > 0

        self.test_dataset = ConcatDataset(
            [Dataset(path_to_data_dir, Dataset.Mode.TEST, preprocessor, augmenter=None, exclude_difficulty=False)
             for path_to_data_dir in [config.path_to_data_dir] + list(config.path_to_extra_data_dirs)]
        )
        num_test_data = len(self.test_dataset)
        logger.i('Found {:d} test samples'.format(num_test_data))

        for datasets in [self.train_dataset.datasets, self.val_dataset.datasets, self.test_dataset.datasets]:
            for dataset in datasets:
                dataset: Dataset
                if dataset.setup_lmdb():
                    logger.i(f'Setup LMDB for {dataset.mode.value} dataset: {dataset.path_to_data_dir}')
        # endregion ==========================

        # region ===== Setup model, optimizer and scheduler =====
        algorithm = algorithm_class(num_classes=self.train_dataset.master.num_classes(),
                                    image_min_side=config.image_min_side,
                                    image_max_side=config.image_max_side)
        model = Model(
            algorithm=algorithm, num_classes=self.train_dataset.master.num_classes(), preprocessor=self.train_dataset.master.preprocessor,
            class_to_category_dict=self.train_dataset.master.class_to_category_dict, category_to_class_dict=self.train_dataset.master.category_to_class_dict
        ).to(device)
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)

        if config.needs_freeze_bn:
            Util.freeze_bn_modules(model)
        else:
            assert config.batch_size > 1, \
                'Expected more than 1 batch size when training without frozen batch normalizations'

        if (config.path_to_resuming_checkpoint is None) and (config.path_to_finetuning_checkpoint is None):
            last_epoch = 0
        elif (config.path_to_resuming_checkpoint is not None) and (config.path_to_finetuning_checkpoint is None) and (config.path_to_loading_checkpoint is None):
            # NOTE: To resume from checkpoint
            #           * model: restore weights
            #           * optimizer: restore states
            #           * scheduler: restore last epoch
            #           * epoch: start from resuming epoch
            checkpoint = Checkpoint.load(config.path_to_resuming_checkpoint, device)
            Trainer._validate_config_for_resuming(config, checkpoint)
            model.load_state_dict(checkpoint.model.state_dict())
            optimizer.load_state_dict(checkpoint.optimizer.state_dict())
            last_epoch = checkpoint.epoch
            logger.i(f'Model has been restored from file: {config.path_to_resuming_checkpoint}')
        elif (config.path_to_resuming_checkpoint is None) and (config.path_to_finetuning_checkpoint is not None) and (config.path_to_loading_checkpoint is None):
            # NOTE: To fine-tune from checkpoint
            #           * model: remove output module and restore weights
            #           * epoch: start from 0
            checkpoint = Checkpoint.load(config.path_to_finetuning_checkpoint, device)
            Trainer._validate_config_for_finetuning(config, checkpoint)
            checkpoint.model.algorithm.remove_output_modules()
            model.load_state_dict(checkpoint.model.state_dict(), strict=False)
            last_epoch = 0
            logger.i(f'Model has been restored from file: {config.path_to_finetuning_checkpoint}')
        elif (config.path_to_resuming_checkpoint is None) and (config.path_to_finetuning_checkpoint is None) and (config.path_to_loading_checkpoint is not None):
            # NOTE: To load from checkpoint
            #           * model: restore weights
            #           * epoch: start from 0
            checkpoint = Checkpoint.load(config.path_to_loading_checkpoint, device)
            model.load_state_dict(checkpoint.model.state_dict())
            last_epoch = 0
            logger.i(f'Model has been restored from file: {config.path_to_loading_checkpoint}')
        else:
            raise ValueError('Only one of `path_to_resuming_checkpoint`, `path_to_finetuning_checkpoint` or `path_to_loading_checkpoint` is expected to have value')

        scheduler = WarmUpMultiStepLR(optimizer, milestones=config.step_lr_sizes, gamma=config.step_lr_gamma,
                                      factor=config.warm_up_factor, num_iters=config.warm_up_num_iters, last_epoch=last_epoch)

        # endregion =============================================

        val_evaluator = Evaluator(self.val_dataset, batch_size=device_count, num_workers=config.num_workers)
        test_evaluator = Evaluator(self.test_dataset, batch_size=device_count, num_workers=config.num_workers) if num_test_data > 0 else None
        self._callback = Trainer.Callback(logger, config, model,
                                          optimizer, scheduler, terminator,
                                          val_evaluator, test_evaluator,
                                          db, path_to_checkpoints_dir)

        self._dataloader = DataLoader(self.train_dataset, config.batch_size,
                                      shuffle=True, num_workers=config.num_workers,
                                      collate_fn=Dataset.collate_fn, pin_memory=True)

        self._model = BunchDataParallel(model)
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._epoch_range = range(last_epoch + 1, config.num_epochs_to_finish + 1)

    def train(self):
        num_batches_in_epoch = len(self._dataloader)

        self._callback.on_train_begin(num_batches_in_epoch)

        for epoch in self._epoch_range:
            self._callback.on_epoch_begin(epoch, num_batches_in_epoch)

            for batch_index, item_tuple_batch in enumerate(self._dataloader):
                n_batch = batch_index + 1
                self._callback.on_batch_begin(epoch, num_batches_in_epoch, n_batch)

                item_batch = [Dataset.Item(*item_tuple) for item_tuple in item_tuple_batch]
                processed_image_batch = Bunch([it.processed_image for it in item_batch])
                processed_bboxes_batch = Bunch([it.processed_bboxes for it in item_batch])
                processed_masks_batch = Bunch([it.processed_masks for it in item_batch])
                classes_batch = Bunch([it.classes for it in item_batch])

                (anchor_objectness_losses, anchor_transformer_losses,
                 proposal_class_losses, proposal_transformer_losses,
                 mask_losses) = \
                    self._model.forward(processed_image_batch, processed_bboxes_batch, processed_masks_batch, classes_batch)

                anchor_objectness_loss = torch.stack(anchor_objectness_losses, dim=0).mean()
                anchor_transformer_loss = torch.stack(anchor_transformer_losses, dim=0).mean()
                proposal_class_loss = torch.stack(proposal_class_losses, dim=0).mean()
                proposal_transformer_loss = torch.stack(proposal_transformer_losses, dim=0).mean()
                mask_loss = torch.stack(mask_losses, dim=0).mean()

                loss = anchor_objectness_loss + anchor_transformer_loss + proposal_class_loss + proposal_transformer_loss + mask_loss
                assert not torch.isnan(loss).any(), 'Got `nan` loss. Please reduce the learning rate and try again.'

                if epoch == 1 and batch_index == 0:
                    self._callback.save_model_graph(image_shape=processed_image_batch[0].shape)

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                self._scheduler.warm_step()

                self._callback.on_batch_end(epoch, num_batches_in_epoch, n_batch,
                                            loss.item(),
                                            anchor_objectness_loss.item(), anchor_transformer_loss.item(),
                                            proposal_class_loss.item(), proposal_transformer_loss.item(),
                                            mask_loss.item())

                if self._callback.is_terminated():
                    break

            self._scheduler.step()

            if self._callback.is_terminated():
                break

            self._callback.on_epoch_end(epoch, num_batches_in_epoch)

        self._callback.on_train_end(num_batches_in_epoch)

        for datasets in [self.train_dataset.datasets, self.val_dataset.datasets, self.test_dataset.datasets]:
            for dataset in datasets:
                dataset: Dataset
                dataset.teardown_lmdb()

    @staticmethod
    def _validate_config_for_resuming(config: Config, checkpoint: Checkpoint):
        for group in checkpoint.optimizer.param_groups:
            # NOTE: Optimizer states were overwritten by resuming states, make sure that
            #       config states are consistent with resuming states to avoid unintended consequences
            assert config.learning_rate == group['initial_lr'], 'config.learning_rate is inconsistent with resuming one: {} vs {}'.format(config.learning_rate, group['initial_lr'])
            assert config.momentum == group['momentum'], 'config.momentum is inconsistent with resuming one: {} vs {}'.format(config.momentum, group['momentum'])
            assert config.weight_decay == group['weight_decay'], 'config.weight_decay is inconsistent with resuming one: {} vs {}'.format(config.weight_decay, group['weight_decay'])

    @staticmethod
    def _validate_config_for_finetuning(config: Config, checkpoint: Checkpoint):
        pass