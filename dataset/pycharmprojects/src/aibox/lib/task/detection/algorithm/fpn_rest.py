from dataclasses import dataclass
from typing import Union, Tuple, List

import torch
from torch import nn, Tensor
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from torchvision.ops.poolers import MultiScaleRoIAlign

from . import Algorithm
from ..backbone import Backbone
from ..head.roi import ROI as ROI
from ..head.rpn import RPN as RPN


class FPN(Algorithm):

    class Body(nn.Module):

        def __init__(self, backbone: Backbone, num_body_out: int):
            super().__init__()
            self.conv1 = backbone.component.conv1
            self.conv2 = backbone.component.conv2
            self.conv3 = backbone.component.conv3
            self.conv4 = backbone.component.conv4
            self.conv5 = backbone.component.conv5
            self.fpn = FeaturePyramidNetwork(in_channels_list=[backbone.component.num_conv2_out,
                                                               backbone.component.num_conv3_out,
                                                               backbone.component.num_conv4_out,
                                                               backbone.component.num_conv5_out],
                                             out_channels=num_body_out,
                                             extra_blocks=LastLevelMaxPool())

        def forward(self, image_batch: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
            c1_batch = self.conv1(image_batch)
            c2_batch = self.conv2(c1_batch)
            c3_batch = self.conv3(c2_batch)
            c4_batch = self.conv4(c3_batch)
            c5_batch = self.conv5(c4_batch)

            x_batch = {'c2': c2_batch, 'c3': c3_batch, 'c4': c4_batch, 'c5': c5_batch}
            x_out_batch = self.fpn(x_batch)

            p2_batch = x_out_batch['c2']
            p3_batch = x_out_batch['c3']
            p4_batch = x_out_batch['c4']
            p5_batch = x_out_batch['c5']
            p6_batch = x_out_batch['pool']
            return p2_batch, p3_batch, p4_batch, p5_batch, p6_batch








    def __init__(self, num_classes: int,
                 backbone: Backbone,
                 anchor_ratios: List[Tuple[int, int]], anchor_sizes: List[int],
                 train_rpn_pre_nms_top_n: int, train_rpn_post_nms_top_n: int,
                 eval_rpn_pre_nms_top_n: int, eval_rpn_post_nms_top_n: int,
                 num_anchor_samples_per_batch: int, num_proposal_samples_per_batch: int, num_detections_per_image: int,
                 anchor_smooth_l1_loss_beta: float, proposal_smooth_l1_loss_beta: float,
                 proposal_nms_threshold: float, detection_nms_threshold: float):
        super().__init__(num_classes,
                         backbone,
                         anchor_ratios, anchor_sizes,
                         train_rpn_pre_nms_top_n, train_rpn_post_nms_top_n,
                         eval_rpn_pre_nms_top_n, eval_rpn_post_nms_top_n,
                         num_anchor_samples_per_batch, num_proposal_samples_per_batch, num_detections_per_image,
                         anchor_smooth_l1_loss_beta, proposal_smooth_l1_loss_beta,
                         proposal_nms_threshold, detection_nms_threshold)

        self.body, num_body_out = self._build_body()
        self.rpn_head, num_rpn_extractor_out = self._build_rpn_head(num_extractor_in=num_body_out)
        self.roi_head = self._build_roi_head(num_extractor_in=num_body_out)

        self._multi_scale_roi_align = MultiScaleRoIAlign(featmap_names=['p2', 'p3', 'p4', 'p5'],
                                                         output_size=(7, 7),
                                                         sampling_ratio=2)

    def _build_body(self) -> Tuple[nn.Module, int]:
        num_body_out = 256
        body = self.Body(self.backbone, num_body_out)
        return body, num_body_out

    def _build_rpn_head(self, num_extractor_in: int) -> Tuple[RPN, int]:
        num_extractor_out = 256
        extractor = nn.Sequential(
            nn.Conv2d(in_channels=num_extractor_in, out_channels=num_extractor_out, kernel_size=3, padding=1),
            nn.ReLU()
        )
        head = RPN(extractor, num_extractor_out,
                   self.anchor_ratios, self.anchor_sizes,
                   self.train_rpn_pre_nms_top_n, self.train_rpn_post_nms_top_n,
                   self.eval_rpn_pre_nms_top_n, self.eval_rpn_post_nms_top_n,
                   self.num_anchor_samples_per_batch, self.anchor_smooth_l1_loss_beta,
                   self.proposal_nms_threshold)

        for m in head.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, val=0)

        return head, num_extractor_out

    def _build_roi_head(self, num_extractor_in: int) -> ROI:
        num_extractor_out = 1024
        extractor = nn.Sequential(
            nn.Linear(in_features=num_extractor_in * 7 * 7, out_features=num_extractor_out),
            nn.ReLU(),
            nn.Linear(in_features=num_extractor_out, out_features=num_extractor_out),
            nn.ReLU()
        )
        head = ROI(extractor, num_extractor_out,
                   self.num_classes,
                   self.num_proposal_samples_per_batch, self.num_detections_per_image,
                   self.proposal_smooth_l1_loss_beta,
                   self.detection_nms_threshold)

        for m in head.children():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
                nn.init.constant_(m.bias, val=0)

        return head

    def forward(
            self, padded_image_batch: Tensor,
            gt_bboxes_batch: List[Tensor] = None, gt_classes_batch: List[Tensor] = None
    ) -> Union[Tuple[Tensor, Tensor, Tensor, Tensor],
               Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor]]]:
        device = padded_image_batch.device
        batch_size, _, padded_image_height, padded_image_width = padded_image_batch.shape

        p2_batch, p3_batch, p4_batch, p5_batch, p6_batch = self.body(padded_image_batch)

        @dataclass
        class Level:
            name: str
            features_batch: Tensor
            anchor_scale: float
            anchor_bboxes_batch: List[Tensor]
            anchor_objectnesses_batch: Tensor
            anchor_transformers_batch: Tensor
            proposal_bboxes_batch: List[Tensor]
            proposal_probs_batch: List[Tensor]

        levels: List[Level] = []

        for name, features_batch, anchor_scale in zip(['p2', 'p3', 'p4', 'p5', 'p6'],
                                                      [p2_batch, p3_batch, p4_batch, p5_batch, p6_batch],
                                                      [.25, .5, 1, 2, 4]):
            _, _, features_height, features_width = features_batch.shape

            anchor_objectnesses_batch, anchor_transformers_batch = self.rpn_head.forward(features_batch)

            pyramidal_anchor_bboxes = \
                self.rpn_head.generate_anchors(
                    padded_image_width, padded_image_height,
                    num_x_anchors=features_width, num_y_anchors=features_height,
                    scale=anchor_scale
                ).to(device)
            anchor_bboxes_batch: List[Tensor] = [pyramidal_anchor_bboxes] * batch_size

            proposal_bboxes_batch, proposal_probs_batch = \
                self.rpn_head.generate_proposals_batch(
                    anchor_bboxes_batch,
                    anchor_objectnesses_batch, anchor_transformers_batch,
                    padded_image_width, padded_image_height
                )

            levels.append(
                Level(name,
                      features_batch,
                      anchor_scale,
                      anchor_bboxes_batch,
                      anchor_objectnesses_batch,
                      anchor_transformers_batch,
                      proposal_bboxes_batch,
                      proposal_probs_batch)
            )

        pyramidal_anchor_bboxes_batch = [torch.cat([level.anchor_bboxes_batch[b] for level in levels], dim=0) for b in range(batch_size)]
        pyramidal_anchor_objectnesses_batch = [torch.cat([level.anchor_objectnesses_batch[b] for level in levels], dim=0) for b in range(batch_size)]
        pyramidal_anchor_transformers_batch = [torch.cat([level.anchor_transformers_batch[b] for level in levels], dim=0) for b in range(batch_size)]

        pyramidal_proposal_bboxes_batch = [torch.cat([level.proposal_bboxes_batch[b] for level in levels], dim=0) for b in range(batch_size)]
        pyramidal_proposal_probs_batch = [torch.cat([level.proposal_probs_batch[b] for level in levels], dim=0) for b in range(batch_size)]

        if self.training:
            # manually generate a background bbox if no any ground-truth was provided
            for b in range(batch_size):
                gt_bboxes = gt_bboxes_batch[b]
                gt_classes = gt_classes_batch[b]
                if gt_bboxes.shape[0] == 0 and gt_classes.shape[0] == 0:
                    gt_bboxes_batch[b] = torch.tensor([
                        [padded_image_width // 4,
                         padded_image_height // 4,
                         padded_image_width // 4 + padded_image_width // 2,
                         padded_image_height // 4 + padded_image_height // 2]
                    ]).to(gt_bboxes)
                    gt_classes_batch[b] = torch.tensor([0]).to(gt_classes)

            anchor_objectness_loss_batch = torch.zeros((batch_size,)).to(device)
            anchor_transformer_loss_batch = torch.zeros((batch_size,)).to(device)

            for b in range(batch_size):
                gt_bboxes = gt_bboxes_batch[b]
                gt_classes = gt_classes_batch[b]
                pyramidal_anchor_bboxes = pyramidal_anchor_bboxes_batch[b]
                pyramidal_anchor_objectnesses = pyramidal_anchor_objectnesses_batch[b]
                pyramidal_anchor_transformers = pyramidal_anchor_transformers_batch[b]

                sampled_indices, sampled_gt_anchor_objectnesses, sampled_gt_anchor_transformers = \
                    self.rpn_head.sample(pyramidal_anchor_bboxes,
                                         gt_bboxes, gt_classes,
                                         padded_image_width, padded_image_height)

                if sampled_indices.shape[0] == 0:
                    continue

                sampled_anchor_objectnesses = pyramidal_anchor_objectnesses[sampled_indices]
                sampled_anchor_transformers = pyramidal_anchor_transformers[sampled_indices]

                anchor_objectness_loss, anchor_transformer_loss = \
                    self.rpn_head.loss(sampled_anchor_objectnesses, sampled_anchor_transformers,
                                       sampled_gt_anchor_objectnesses, sampled_gt_anchor_transformers)

                anchor_objectness_loss_batch[b] = anchor_objectness_loss
                anchor_transformer_loss_batch[b] = anchor_transformer_loss

            proposal_class_loss_batch = torch.zeros((batch_size,)).to(device)
            proposal_transformer_loss_batch = torch.zeros((batch_size,)).to(device)

            for b in range(batch_size):
                name_to_features_dict = {level.name: level.features_batch[b] for level in levels}
                gt_bboxes = gt_bboxes_batch[b]
                gt_classes = gt_classes_batch[b]
                pyramidal_proposal_bboxes = pyramidal_proposal_bboxes_batch[b]
                pyramidal_proposal_probs = pyramidal_proposal_probs_batch[b]

                _, sorted_indices = pyramidal_proposal_probs.sort(dim=0, descending=True)
                pyramidal_proposal_bboxes = pyramidal_proposal_bboxes[sorted_indices][:self.train_rpn_post_nms_top_n]
                pyramidal_proposal_probs = pyramidal_proposal_probs[sorted_indices][:self.train_rpn_post_nms_top_n]

                pyramidal_proposal_bboxes = torch.cat([pyramidal_proposal_bboxes, gt_bboxes], dim=0)

                sampled_indices, sampled_gt_proposal_classes, sampled_gt_proposal_transformers = \
                    self.roi_head.sample(pyramidal_proposal_bboxes,
                                         gt_bboxes, gt_classes)
                sampled_proposal_bboxes = pyramidal_proposal_bboxes[sampled_indices]

                if sampled_proposal_bboxes.shape[0] == 0:
                    continue

                pools = self._multi_scale_roi_align({k: v.unsqueeze(dim=0) for k, v in name_to_features_dict.items()},
                                                    boxes=[sampled_proposal_bboxes],
                                                    image_shapes=[(padded_image_height, padded_image_width)])

                proposal_classes, proposal_transformers = \
                    self.roi_head.forward(pools,
                                          pre_extract_transform=lambda x: x.view(x.shape[0], -1))

                proposal_class_loss, proposal_transformer_loss = \
                    self.roi_head.loss(proposal_classes, proposal_transformers,
                                       sampled_gt_proposal_classes,
                                       sampled_gt_proposal_transformers)

                proposal_class_loss_batch[b] = proposal_class_loss
                proposal_transformer_loss_batch[b] = proposal_transformer_loss

            return (anchor_objectness_loss_batch, anchor_transformer_loss_batch,
                    proposal_class_loss_batch, proposal_transformer_loss_batch)
        else:
            pools_batch = []
            for b in range(batch_size):
                name_to_features_dict = {level.name: level.features_batch[b] for level in levels}
                pyramidal_proposal_bboxes = pyramidal_proposal_bboxes_batch[b]
                pyramidal_proposal_probs = pyramidal_proposal_probs_batch[b]

                _, sorted_indices = pyramidal_proposal_probs.sort(dim=0, descending=True)
                pyramidal_proposal_bboxes = pyramidal_proposal_bboxes[sorted_indices][:self.eval_rpn_post_nms_top_n]
                pyramidal_proposal_probs = pyramidal_proposal_probs[sorted_indices][:self.eval_rpn_post_nms_top_n]

                pyramidal_proposal_bboxes_batch[b] = pyramidal_proposal_bboxes
                pyramidal_proposal_probs_batch[b] = pyramidal_proposal_probs

                pools = self._multi_scale_roi_align({k: v.unsqueeze(dim=0) for k, v in name_to_features_dict.items()},
                                                    boxes=[pyramidal_proposal_bboxes],
                                                    image_shapes=[(padded_image_height, padded_image_width)])
                pools_batch.append(pools)

            detection_bboxes_batch, detection_classes_batch, detection_probs_batch = [], [], []
            for pools in pools_batch:
                proposal_classes, proposal_transformers = \
                    self.roi_head.forward(pools,
                                          pre_extract_transform=lambda x: x.view(x.shape[0], -1))

                detection_bboxes, detection_classes, detection_probs = \
                    self.roi_head.generate_detections(pyramidal_proposal_bboxes,
                                                      proposal_classes, proposal_transformers,
                                                      padded_image_width, padded_image_height)

                detection_bboxes_batch.append(detection_bboxes)
                detection_classes_batch.append(detection_classes)
                detection_probs_batch.append(detection_probs)

            return (pyramidal_anchor_bboxes_batch, pyramidal_proposal_bboxes_batch, pyramidal_proposal_probs_batch,
                    detection_bboxes_batch, detection_classes_batch, detection_probs_batch)

    def remove_output_modules(self):
        del self.rpn_head.anchor_objectness
        del self.rpn_head.anchor_transformer
        del self.roi_head.proposal_class
        del self.roi_head.proposal_transformer