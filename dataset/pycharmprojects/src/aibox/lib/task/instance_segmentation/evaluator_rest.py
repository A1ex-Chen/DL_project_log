from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from itertools import chain
from typing import Dict, List, Union

import numpy as np
import torch
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.ops.boxes import remove_small_boxes
from tqdm import tqdm

from .coco_eval import CocoEvaluator
from .dataset import Dataset, ConcatDataset
from .model import Model
from .preprocessor import Preprocessor
from ...extension.data_parallel import BunchDataParallel, Bunch


class Evaluator:

    @dataclass
    class Prediction:
        sorted_all_image_ids: List[str]
        sorted_all_pred_bboxes: Tensor
        sorted_all_pred_classes: Tensor
        sorted_all_pred_probs: Tensor
        sorted_all_pred_probmasks: List[Tensor]
        image_id_to_pred_bboxes_dict: Dict[str, Tensor]
        image_id_to_pred_classes_dict: Dict[str, Tensor]
        image_id_to_pred_probs_dict: Dict[str, Tensor]
        image_id_to_pred_probmasks_dict: Dict[str, Tensor]
        image_id_to_process_dict_dict: Dict[str, Dict]
        image_id_to_gt_bboxes_dict: Dict[str, Tensor]
        image_id_to_gt_classes_dict: Dict[str, Tensor]
        image_id_to_gt_masks_dict: Dict[str, Tensor]
        image_id_to_difficulties_dict: Dict[str, Tensor]
        class_to_num_positives_dict: Dict[int, int]

    @dataclass
    class Evaluation:

        class Quality(Enum):
            LOOSEST = 'loosest'      # AP@0.05
            LOOSE = 'loose'          # AP@0.25
            STANDARD = 'standard'    # AP@0.50
            STRICT = 'strict'        # AP@0.75
            STRICTEST = 'strictest'  # AP@0.95

            def to_iou_threshold(self) -> float:
                if self == Evaluator.Evaluation.Quality.LOOSEST:
                    return 0.05
                elif self == Evaluator.Evaluation.Quality.LOOSE:
                    return 0.25
                elif self == Evaluator.Evaluation.Quality.STANDARD:
                    return 0.5
                elif self == Evaluator.Evaluation.Quality.STRICT:
                    return 0.75
                elif self == Evaluator.Evaluation.Quality.STRICTEST:
                    return 0.95
                else:
                    raise ValueError('Invalid quality')

        class Size(Enum):
            ALL = 'all'
            AREA_L = 'area_l'
            AREA_M = 'area_m'
            AREA_S = 'area_s'
            W_S = 'w_s'
            W_M = 'w_m'
            W_L = 'w_l'
            H_S = 'h_s'
            H_M = 'h_m'
            H_L = 'h_l'
            WH_SS = 'wh_ss'
            WH_SM = 'wh_sm'
            WH_SL = 'wh_sl'
            WH_MS = 'wh_ms'
            WH_MM = 'wh_mm'
            WH_ML = 'wh_ml'
            WH_LS = 'wh_ls'
            WH_LM = 'wh_lm'
            WH_LL = 'wh_ll'

        @dataclass
        class MetricResult:
            mean_value: float
            class_to_value_dict: Dict[int, float]

        quality: Quality
        size: Size
        class_to_inter_recall_array_dict: Dict[int, np.ndarray]
        class_to_inter_precision_array_dict: Dict[int, np.ndarray]
        class_to_recall_array_dict: Dict[int, np.ndarray]
        class_to_precision_array_dict: Dict[int, np.ndarray]
        class_to_f1_score_array_dict: Dict[int, np.ndarray]
        class_to_prob_array_dict: Dict[int, np.ndarray]
        metric_ap: MetricResult


    @torch.no_grad()



        class Size(Enum):
            ALL = 'all'
            AREA_L = 'area_l'
            AREA_M = 'area_m'
            AREA_S = 'area_s'
            W_S = 'w_s'
            W_M = 'w_m'
            W_L = 'w_l'
            H_S = 'h_s'
            H_M = 'h_m'
            H_L = 'h_l'
            WH_SS = 'wh_ss'
            WH_SM = 'wh_sm'
            WH_SL = 'wh_sl'
            WH_MS = 'wh_ms'
            WH_MM = 'wh_mm'
            WH_ML = 'wh_ml'
            WH_LS = 'wh_ls'
            WH_LM = 'wh_lm'
            WH_LL = 'wh_ll'

        @dataclass
        class MetricResult:
            mean_value: float
            class_to_value_dict: Dict[int, float]

        quality: Quality
        size: Size
        class_to_inter_recall_array_dict: Dict[int, np.ndarray]
        class_to_inter_precision_array_dict: Dict[int, np.ndarray]
        class_to_recall_array_dict: Dict[int, np.ndarray]
        class_to_precision_array_dict: Dict[int, np.ndarray]
        class_to_f1_score_array_dict: Dict[int, np.ndarray]
        class_to_prob_array_dict: Dict[int, np.ndarray]
        metric_ap: MetricResult

    def __init__(self, dataset: Union[Dataset, ConcatDataset], batch_size: int, num_workers: int):
        super().__init__()
        self._dataset = dataset
        self._dataloader = DataLoader(dataset, batch_size,
                                      shuffle=False,
                                      num_workers=num_workers,
                                      collate_fn=Dataset.collate_fn, pin_memory=True)
        self._quality = Evaluator.Evaluation.Quality.STANDARD
        self._size = Evaluator.Evaluation.Size.ALL

        if isinstance(self._dataset, Dataset):
            self._num_classes = self._dataset.num_classes()
        elif isinstance(self._dataset, ConcatDataset):
            self._num_classes = self._dataset.master.num_classes()
        else:
            raise TypeError

    @torch.no_grad()
    def predict(self, model: Model, needs_inv_process: bool) -> Prediction:
        model = BunchDataParallel(model)

        all_image_ids, all_pred_bboxes, all_pred_classes, all_pred_probs, all_pred_probmasks = [], [], [], [], []
        image_id_to_pred_bboxes_dict, image_id_to_pred_classes_dict, image_id_to_pred_probs_dict, image_id_to_pred_probmasks_dict, image_id_to_process_dict_dict = {}, {}, {}, {}, {}
        image_id_to_gt_bboxes_dict, image_id_to_gt_classes_dict, image_id_to_gt_masks_dict, image_id_to_difficulties_dict = {}, {}, {}, {}
        class_to_num_positives_dict = defaultdict(int)

        for _, item_batch in enumerate(tqdm(self._dataloader, mininterval=10)):
            processed_image_batch = Bunch([it.processed_image for it in item_batch])

            (detection_bboxes_batch, detection_classes_batch, detection_probs_batch, detection_probmasks_batch) = \
                model.eval().forward(processed_image_batch)

            for b, item in enumerate(item_batch):
                item: Dataset.Item
                image_id = item.image_id
                process_dict = item.process_dict

                detection_bboxes = detection_bboxes_batch[b].cpu()
                detection_classes = detection_classes_batch[b].cpu()
                detection_probs = detection_probs_batch[b].cpu()
                detection_probmasks = detection_probmasks_batch[b].cpu()

                if needs_inv_process:
                    detection_bboxes = Preprocessor.inv_process_bboxes(process_dict, detection_bboxes)
                    detection_probmasks = Preprocessor.inv_process_probmasks(process_dict, detection_probmasks)

                kept_indices = (detection_probs > 0.05).nonzero().flatten()
                detection_bboxes = detection_bboxes[kept_indices]
                detection_classes = detection_classes[kept_indices]
                detection_probs = detection_probs[kept_indices]
                detection_probmasks = detection_probmasks[kept_indices]

                kept_indices = remove_small_boxes(detection_bboxes, 1)
                pred_bboxes = detection_bboxes[kept_indices]
                pred_classes = detection_classes[kept_indices]
                pred_probs = detection_probs[kept_indices]
                pred_probmasks = detection_probmasks[kept_indices]

                all_image_ids.extend([image_id] * pred_bboxes.shape[0])
                all_pred_bboxes.append(pred_bboxes)
                all_pred_classes.append(pred_classes)
                all_pred_probs.append(pred_probs)
                all_pred_probmasks.append(pred_probmasks)

                gt_bboxes = item.bboxes
                gt_classes = item.classes
                gt_masks = item.masks
                difficulties = item.difficulties

                image_id_to_pred_bboxes_dict[image_id] = pred_bboxes
                image_id_to_pred_classes_dict[image_id] = pred_classes
                image_id_to_pred_probs_dict[image_id] = pred_probs
                image_id_to_pred_probmasks_dict[image_id] = pred_probmasks
                image_id_to_process_dict_dict[image_id] = process_dict

                image_id_to_gt_bboxes_dict[image_id] = gt_bboxes
                image_id_to_gt_classes_dict[image_id] = gt_classes
                image_id_to_gt_masks_dict[image_id] = gt_masks
                image_id_to_difficulties_dict[image_id] = difficulties

                for gt_class in gt_classes.unique().tolist():
                    class_mask = gt_classes == gt_class
                    num_positives = class_mask.sum().item()
                    num_positives -= (difficulties[class_mask] == 1).sum().item()
                    class_to_num_positives_dict[gt_class] += num_positives

        all_pred_bboxes = torch.cat(all_pred_bboxes, dim=0)
        all_pred_classes = torch.cat(all_pred_classes, dim=0)
        all_pred_probs = torch.cat(all_pred_probs, dim=0)
        all_pred_probmasks = list(chain(*all_pred_probmasks))

        sorted_indices = all_pred_probs.argsort(dim=-1, descending=True)
        sorted_all_image_ids = [all_image_ids[i.item()] for i in sorted_indices]
        sorted_all_pred_bboxes = all_pred_bboxes[sorted_indices]
        sorted_all_pred_classes = all_pred_classes[sorted_indices]
        sorted_all_pred_probs = all_pred_probs[sorted_indices]
        sorted_all_pred_probmasks = [all_pred_probmasks[i] for i in sorted_indices.tolist()]

        return Evaluator.Prediction(sorted_all_image_ids,
                                    sorted_all_pred_bboxes,
                                    sorted_all_pred_classes,
                                    sorted_all_pred_probs,
                                    sorted_all_pred_probmasks,
                                    image_id_to_pred_bboxes_dict,
                                    image_id_to_pred_classes_dict,
                                    image_id_to_pred_probs_dict,
                                    image_id_to_pred_probmasks_dict,
                                    image_id_to_process_dict_dict,
                                    image_id_to_gt_bboxes_dict,
                                    image_id_to_gt_classes_dict,
                                    image_id_to_gt_masks_dict,
                                    image_id_to_difficulties_dict,
                                    class_to_num_positives_dict)

    def evaluate(self, model: Model) -> Evaluation:
        # NOTE: Do later inverse process for reducing memory usage by probmasks of prediction.
        #       Instead, we inverse process `probmasks` in `evaluate_with_condition` one by one so that
        #       extremely large of original size can be handled well.
        prediction = self.predict(model, needs_inv_process=False)
        evaluation = self._evaluate_with_condition(prediction, self._quality, self._size, pred_needs_inv_process=True)
        return evaluation

    def _evaluate_with_condition(self, prediction: Prediction,
                                 quality: Evaluation.Quality, size: Evaluation.Size,
                                 pred_needs_inv_process: bool) -> Evaluation:
        assert quality in [Evaluator.Evaluation.Quality.STANDARD,
                           Evaluator.Evaluation.Quality.STRICT,
                           Evaluator.Evaluation.Quality.STRICTEST], \
            'Only `Quality.STANDARD`, `Quality.STRICT` and `Quality.STRICTEST` are supported now'
        assert size in [Evaluator.Evaluation.Size.ALL,
                        Evaluator.Evaluation.Size.AREA_L,
                        Evaluator.Evaluation.Size.AREA_M,
                        Evaluator.Evaluation.Size.AREA_S], \
            'Only `Size.ALL`, `Size.AREA_L`, `Size.AREA_M` and `Size.AREA_S` are supported now'

        # # TODO: change to use metric.average_precision
        n_threads = torch.get_num_threads()
        # FIXME remove this and make paste_masks_in_image run on the GPU
        torch.set_num_threads(1)

        def convert_to_coco_api(ds):
            coco_ds = COCO()
            # annotation IDs need to start at 1, not 0, see torchvision issue #1530
            ann_id = 1
            dataset = {'images': [], 'categories': [], 'annotations': []}
            categories = set()
            for img_idx in range(len(ds)):
                # find better way to get target
                # targets = ds.get_annotations(img_idx)
                item = ds[img_idx]
                image_id = item.image_id
                img_dict = {}
                img_dict['id'] = image_id
                img_dict['height'] = item.image.shape[1]
                img_dict['width'] = item.image.shape[2]
                dataset['images'].append(img_dict)
                bboxes = item.bboxes
                bboxes[:, 2:] -= bboxes[:, :2]  # convert to xywh
                bboxes = bboxes.tolist()
                labels = item.classes.tolist()
                areas = item.bboxes[:, 2] * item.bboxes[:, 3]
                areas = areas.tolist()
                iscrowd = item.difficulties.tolist()
                masks = item.masks
                # make masks Fortran contiguous for coco_mask
                masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)
                num_objs = len(bboxes)
                for i in range(num_objs):
                    ann = {}
                    ann['image_id'] = image_id
                    ann['bbox'] = bboxes[i]
                    ann['category_id'] = labels[i]
                    categories.add(labels[i])
                    ann['area'] = areas[i]
                    ann['iscrowd'] = iscrowd[i]
                    ann['id'] = ann_id
                    ann["segmentation"] = coco_mask.encode(masks[i].numpy().astype(np.uint8))
                    dataset['annotations'].append(ann)
                    ann_id += 1
            dataset['categories'] = [{'id': i} for i in sorted(categories)]
            coco_ds.dataset = dataset
            coco_ds.createIndex()
            return coco_ds

        coco = convert_to_coco_api(self._dataset)
        iou_types = ["bbox", "segm"]
        coco_evaluator = CocoEvaluator(coco, iou_types)

        for image_id in prediction.image_id_to_pred_bboxes_dict.keys():
            output = {
                'boxes': Preprocessor.inv_process_bboxes(prediction.image_id_to_process_dict_dict[image_id],
                                                         prediction.image_id_to_pred_bboxes_dict[image_id]) if pred_needs_inv_process else prediction.image_id_to_pred_bboxes_dict[image_id],
                'labels': prediction.image_id_to_pred_classes_dict[image_id],
                'masks': Preprocessor.inv_process_probmasks(prediction.image_id_to_process_dict_dict[image_id],
                                                            prediction.image_id_to_pred_probmasks_dict[image_id]) if pred_needs_inv_process else prediction.image_id_to_pred_probmasks_dict[image_id],
                'scores': prediction.image_id_to_pred_probs_dict[image_id]
            }
            res = {image_id: output}
            coco_evaluator.update(res)

        coco_evaluator.synchronize_between_processes()

        # accumulate predictions from all images
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        torch.set_num_threads(n_threads)

        class_to_inter_recall_array_dict = {c: np.linspace(0, 1, 101)
                                            for c in range(1, self._num_classes)}

        # coco_evaluator.coco_eval['segm'].eval['precision'] has shape (T, R, K, A, M) where
        #   T: 10 IoU thresholds [0.5, 0.55, ..., 0.95]
        #   R: 101 recall thresholds [0, 0.01, 0.02, ..., 1]
        #   K: number of classes without background
        #   A: 4 areas [all, small, medium, large]
        #   M: maximum of detections [1, 10, 100]

        if quality == Evaluator.Evaluation.Quality.STANDARD:
            iou_threshold_index = 0
        elif quality == Evaluator.Evaluation.Quality.STRICT:
            iou_threshold_index = 5
        elif quality == Evaluator.Evaluation.Quality.STRICTEST:
            iou_threshold_index = 9
        else:
            raise ValueError

        if size == Evaluator.Evaluation.Size.ALL:
            area_index = 0
        elif size == Evaluator.Evaluation.Size.AREA_S:
            area_index = 1
        elif size == Evaluator.Evaluation.Size.AREA_M:
            area_index = 2
        elif size == Evaluator.Evaluation.Size.AREA_L:
            area_index = 3
        else:
            raise ValueError

        max_detections_index = 2

        precisions = coco_evaluator.coco_eval['segm'].eval['precision']
        scores = coco_evaluator.coco_eval['segm'].eval['scores']  # shape (T, R, K, A, M) as mentioned above

        class_to_inter_precision_array_dict = {
            c: precisions[iou_threshold_index, :, c - 1, area_index, max_detections_index]
            for c in range(1, self._num_classes)
        }
        class_to_prob_array_dict = {
            c: scores[iou_threshold_index, :, c - 1, area_index, max_detections_index]
            for c in range(1, self._num_classes)
        }

        class_to_recall_array_dict = class_to_inter_recall_array_dict  # FIXME: temperately use interpolation one
        class_to_precision_array_dict = class_to_inter_precision_array_dict  # FIXME: temperately use interpolation one
        class_to_f1_score_array_dict = {
            c: 2 * class_to_recall_array_dict[c] * class_to_precision_array_dict[c] / (class_to_recall_array_dict[c] +
                                                                                       class_to_precision_array_dict[c]
                                                                                       + np.finfo(np.float32).eps)
            for c in range(1, self._num_classes)
        }

        mean_ap = coco_evaluator.coco_eval['segm'].stats[1]  # AP@0.5 (change to [0] to use AP@[.5:.95:.05]
        class_to_ap_dict = {c: precisions[iou_threshold_index, :, c - 1, area_index, max_detections_index].mean().item()
                            for c in range(1, self._num_classes)}

        evaluation = Evaluator.Evaluation(
            quality,
            size,
            class_to_inter_recall_array_dict,
            class_to_inter_precision_array_dict,
            class_to_recall_array_dict,
            class_to_precision_array_dict,
            class_to_f1_score_array_dict,
            class_to_prob_array_dict,
            metric_ap=Evaluator.Evaluation.MetricResult(mean_ap, class_to_ap_dict)
        )

        return evaluation