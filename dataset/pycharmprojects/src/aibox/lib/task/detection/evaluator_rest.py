from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.ops.boxes import remove_small_boxes
from tqdm import tqdm

from .dataset import Dataset, ConcatDataset
from .model import Model
from .preprocessor import Preprocessor
from ...extension.data_parallel import BunchDataParallel, Bunch
from ...metric.average_precision import AveragePrecision


class Evaluator:

    @dataclass
    class Prediction:
        sorted_all_image_ids: List[str]
        sorted_all_pred_bboxes: Tensor
        sorted_all_pred_classes: Tensor
        sorted_all_pred_probs: Tensor
        image_id_to_pred_bboxes_dict: Dict[str, Tensor]
        image_id_to_pred_classes_dict: Dict[str, Tensor]
        image_id_to_pred_probs_dict: Dict[str, Tensor]
        image_id_to_gt_bboxes_dict: Dict[str, Tensor]
        image_id_to_gt_classes_dict: Dict[str, Tensor]
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
        class_to_accuracy_array_dict: Dict[int, np.ndarray]
        class_to_f1_score_array_dict: Dict[int, np.ndarray]
        class_to_prob_array_dict: Dict[int, np.ndarray]
        metric_ap: MetricResult
        metric_top_f1_score: MetricResult
        metric_recall_at_top_f1_score: MetricResult
        metric_precision_at_top_f1_score: MetricResult
        metric_accuracy_at_top_f1_score: MetricResult
        coco_result: Optional[AveragePrecision.PyCOCOToolsResult]


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
        class_to_accuracy_array_dict: Dict[int, np.ndarray]
        class_to_f1_score_array_dict: Dict[int, np.ndarray]
        class_to_prob_array_dict: Dict[int, np.ndarray]
        metric_ap: MetricResult
        metric_top_f1_score: MetricResult
        metric_recall_at_top_f1_score: MetricResult
        metric_precision_at_top_f1_score: MetricResult
        metric_accuracy_at_top_f1_score: MetricResult
        coco_result: Optional[AveragePrecision.PyCOCOToolsResult]

    def __init__(self, dataset: Union[Dataset, ConcatDataset], batch_size: int, num_workers: int,
                 quality: Evaluation.Quality = Evaluation.Quality.STANDARD):
        super().__init__()
        self._dataset = dataset
        self._dataloader = DataLoader(dataset, batch_size,
                                      shuffle=False,
                                      num_workers=num_workers,
                                      collate_fn=Dataset.collate_fn, pin_memory=True)
        self._quality = quality
        self._size = Evaluator.Evaluation.Size.ALL

        if isinstance(self._dataset, Dataset):
            self._num_classes = self._dataset.num_classes()
        elif isinstance(self._dataset, ConcatDataset):
            self._num_classes = self._dataset.master.num_classes()
        else:
            raise TypeError

    @torch.no_grad()
    def predict(self, model: Model) -> Prediction:
        model = BunchDataParallel(model)

        all_image_ids, all_pred_bboxes, all_pred_classes, all_pred_probs = [], [], [], []
        image_id_to_pred_bboxes_dict, image_id_to_pred_classes_dict, image_id_to_pred_probs_dict = {}, {}, {}
        image_id_to_gt_bboxes_dict, image_id_to_gt_classes_dict, image_id_to_difficulties_dict = {}, {}, {}
        class_to_num_positives_dict = defaultdict(int)

        for _, item_batch in enumerate(tqdm(self._dataloader, mininterval=10)):
            processed_image_batch = Bunch([it.processed_image for it in item_batch])

            (_, _, _,
             detection_bboxes_batch, detection_classes_batch, detection_probs_batch) = \
                model.eval().forward(processed_image_batch)

            for b, item in enumerate(item_batch):
                item: Dataset.Item
                image_id = item.image_id
                process_dict = item.process_dict

                detection_bboxes = detection_bboxes_batch[b].cpu()
                detection_classes = detection_classes_batch[b].cpu()
                detection_probs = detection_probs_batch[b].cpu()

                detection_bboxes = Preprocessor.inv_process_bboxes(process_dict, detection_bboxes)

                kept_indices = (detection_probs > 0.05).nonzero().flatten()
                detection_bboxes = detection_bboxes[kept_indices]
                detection_classes = detection_classes[kept_indices]
                detection_probs = detection_probs[kept_indices]

                kept_indices = remove_small_boxes(detection_bboxes, 1)
                pred_bboxes = detection_bboxes[kept_indices]
                pred_classes = detection_classes[kept_indices]
                pred_probs = detection_probs[kept_indices]

                all_image_ids.extend([image_id] * detection_bboxes.shape[0])
                all_pred_bboxes.append(pred_bboxes)
                all_pred_classes.append(pred_classes)
                all_pred_probs.append(pred_probs)

                gt_bboxes = item.bboxes
                gt_classes = item.classes
                difficulties = item.difficulties

                image_id_to_pred_bboxes_dict[image_id] = pred_bboxes
                image_id_to_pred_classes_dict[image_id] = pred_classes
                image_id_to_pred_probs_dict[image_id] = pred_probs

                image_id_to_gt_bboxes_dict[image_id] = gt_bboxes
                image_id_to_gt_classes_dict[image_id] = gt_classes
                image_id_to_difficulties_dict[image_id] = difficulties

                for gt_class in gt_classes.unique().tolist():
                    class_mask = gt_classes == gt_class
                    num_positives = class_mask.sum().item()
                    num_positives -= (difficulties[class_mask] == 1).sum().item()
                    class_to_num_positives_dict[gt_class] += num_positives

        all_pred_bboxes = torch.cat(all_pred_bboxes, dim=0)
        all_pred_classes = torch.cat(all_pred_classes, dim=0)
        all_pred_probs = torch.cat(all_pred_probs, dim=0)

        sorted_indices = all_pred_probs.argsort(dim=-1, descending=True)
        sorted_all_image_ids = [all_image_ids[i.item()] for i in sorted_indices]
        sorted_all_pred_bboxes = all_pred_bboxes[sorted_indices]
        sorted_all_pred_classes = all_pred_classes[sorted_indices]
        sorted_all_pred_probs = all_pred_probs[sorted_indices]

        return Evaluator.Prediction(sorted_all_image_ids,
                                    sorted_all_pred_bboxes,
                                    sorted_all_pred_classes,
                                    sorted_all_pred_probs,
                                    image_id_to_pred_bboxes_dict,
                                    image_id_to_pred_classes_dict,
                                    image_id_to_pred_probs_dict,
                                    image_id_to_gt_bboxes_dict,
                                    image_id_to_gt_classes_dict,
                                    image_id_to_difficulties_dict,
                                    class_to_num_positives_dict)

    def evaluate(self, model: Model, returns_coco_result: bool = False) -> Evaluation:
        prediction = self.predict(model)
        evaluation = self._evaluate_with_condition(prediction, self._quality, self._size, returns_coco_result)
        return evaluation

    def _evaluate_with_condition(self, prediction: Prediction,
                                 quality: Evaluation.Quality, size: Evaluation.Size,
                                 returns_coco_result: bool) -> Evaluation:
        assert size == Evaluator.Evaluation.Size.ALL, 'Only `Size.ALL` is supported now'

        metric = AveragePrecision(
            {k: v.numpy() for k, v in prediction.image_id_to_pred_bboxes_dict.items()},
            {k: v.numpy() for k, v in prediction.image_id_to_pred_classes_dict.items()},
            {k: v.numpy() for k, v in prediction.image_id_to_pred_probs_dict.items()},
            {k: v.numpy() for k, v in prediction.image_id_to_gt_bboxes_dict.items()},
            {k: v.numpy() for k, v in prediction.image_id_to_gt_classes_dict.items()},
            {k: v.numpy() for k, v in prediction.image_id_to_difficulties_dict.items()},
            self._num_classes
        )
        mean_ap, class_to_result_dict = metric.evaluate(iou_threshold=quality.to_iou_threshold())

        class_to_ap_dict = {}
        class_to_inter_recall_array_dict = {}
        class_to_inter_precision_array_dict = {}
        class_to_recall_array_dict = {}
        class_to_precision_array_dict = {}
        class_to_accuracy_array_dict = {}
        class_to_f1_score_array_dict = {}
        class_to_prob_array_dict = {}
        class_to_top_f1_score_dict = {}
        class_to_recall_at_top_f1_score_dict = {}
        class_to_precision_at_top_f1_score_dict = {}
        class_to_accuracy_at_top_f1_score_dict = {}

        for c in range(1, self._num_classes):
            result = class_to_result_dict[c]
            ap = result.ap
            inter_recall_array = result.inter_recall_array
            inter_precision_array = result.inter_precision_array
            recall_array = result.recall_array
            precision_array = result.precision_array
            accuracy_array = result.accuracy_array
            f1_score_array = 2 * recall_array * precision_array / (recall_array + precision_array +
                                                                   np.finfo(np.float32).eps)
            prob_array = result.prob_array

            class_to_ap_dict[c] = ap
            class_to_inter_recall_array_dict[c] = inter_recall_array
            class_to_inter_precision_array_dict[c] = inter_precision_array
            class_to_recall_array_dict[c] = recall_array
            class_to_precision_array_dict[c] = precision_array
            class_to_accuracy_array_dict[c] = accuracy_array
            class_to_f1_score_array_dict[c] = f1_score_array
            class_to_prob_array_dict[c] = prob_array

            if f1_score_array.shape[0] > 0:
                top_f1_score_index = f1_score_array.argmax().item()
                class_to_top_f1_score_dict[c] = f1_score_array.max().item()
                class_to_recall_at_top_f1_score_dict[c] = recall_array[top_f1_score_index].item()
                class_to_precision_at_top_f1_score_dict[c] = precision_array[top_f1_score_index].item()
                class_to_accuracy_at_top_f1_score_dict[c] = accuracy_array[top_f1_score_index].item()
            else:
                class_to_top_f1_score_dict[c] = 0.
                class_to_recall_at_top_f1_score_dict[c] = 0.
                class_to_precision_at_top_f1_score_dict[c] = 0.
                class_to_accuracy_at_top_f1_score_dict[c] = 0.

        mean_top_f1_score = sum([v for _, v in class_to_top_f1_score_dict.items()]) / len(class_to_top_f1_score_dict)
        mean_recall_at_top_f1_score = sum([v for _, v in class_to_recall_at_top_f1_score_dict.items()]
                                          ) / len(class_to_recall_at_top_f1_score_dict)
        mean_precision_at_top_f1_score = sum([v for _, v in class_to_precision_at_top_f1_score_dict.items()]
                                             ) / len(class_to_precision_at_top_f1_score_dict)
        mean_accuracy_at_top_f1_score = sum([v for _, v in class_to_accuracy_at_top_f1_score_dict.items()]
                                            ) / len(class_to_accuracy_at_top_f1_score_dict)

        coco_result = metric.evaluate_by_pycocotools() if returns_coco_result else None

        evaluation = Evaluator.Evaluation(
            quality,
            size,
            class_to_inter_recall_array_dict,
            class_to_inter_precision_array_dict,
            class_to_recall_array_dict,
            class_to_precision_array_dict,
            class_to_accuracy_array_dict,
            class_to_f1_score_array_dict,
            class_to_prob_array_dict,
            metric_ap=Evaluator.Evaluation.MetricResult(mean_ap, class_to_ap_dict),
            metric_top_f1_score=Evaluator.Evaluation.MetricResult(mean_top_f1_score, class_to_top_f1_score_dict),
            metric_recall_at_top_f1_score=Evaluator.Evaluation.MetricResult(mean_recall_at_top_f1_score,
                                                                            class_to_recall_at_top_f1_score_dict),
            metric_precision_at_top_f1_score=Evaluator.Evaluation.MetricResult(mean_precision_at_top_f1_score,
                                                                               class_to_precision_at_top_f1_score_dict),
            metric_accuracy_at_top_f1_score=Evaluator.Evaluation.MetricResult(mean_accuracy_at_top_f1_score,
                                                                              class_to_accuracy_at_top_f1_score_dict),
            coco_result=coco_result
        )
        return evaluation