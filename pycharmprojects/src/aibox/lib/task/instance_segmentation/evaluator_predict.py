@torch.no_grad()
def predict(self, model: Model, needs_inv_process: bool) ->Prediction:
    model = BunchDataParallel(model)
    (all_image_ids, all_pred_bboxes, all_pred_classes, all_pred_probs,
        all_pred_probmasks) = [], [], [], [], []
    (image_id_to_pred_bboxes_dict, image_id_to_pred_classes_dict,
        image_id_to_pred_probs_dict, image_id_to_pred_probmasks_dict,
        image_id_to_process_dict_dict) = {}, {}, {}, {}, {}
    (image_id_to_gt_bboxes_dict, image_id_to_gt_classes_dict,
        image_id_to_gt_masks_dict, image_id_to_difficulties_dict) = {}, {}, {
        }, {}
    class_to_num_positives_dict = defaultdict(int)
    for _, item_batch in enumerate(tqdm(self._dataloader, mininterval=10)):
        processed_image_batch = Bunch([it.processed_image for it in item_batch]
            )
        (detection_bboxes_batch, detection_classes_batch,
            detection_probs_batch, detection_probmasks_batch) = model.eval(
            ).forward(processed_image_batch)
        for b, item in enumerate(item_batch):
            item: Dataset.Item
            image_id = item.image_id
            process_dict = item.process_dict
            detection_bboxes = detection_bboxes_batch[b].cpu()
            detection_classes = detection_classes_batch[b].cpu()
            detection_probs = detection_probs_batch[b].cpu()
            detection_probmasks = detection_probmasks_batch[b].cpu()
            if needs_inv_process:
                detection_bboxes = Preprocessor.inv_process_bboxes(process_dict
                    , detection_bboxes)
                detection_probmasks = Preprocessor.inv_process_probmasks(
                    process_dict, detection_probmasks)
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
    sorted_all_pred_probmasks = [all_pred_probmasks[i] for i in
        sorted_indices.tolist()]
    return Evaluator.Prediction(sorted_all_image_ids,
        sorted_all_pred_bboxes, sorted_all_pred_classes,
        sorted_all_pred_probs, sorted_all_pred_probmasks,
        image_id_to_pred_bboxes_dict, image_id_to_pred_classes_dict,
        image_id_to_pred_probs_dict, image_id_to_pred_probmasks_dict,
        image_id_to_process_dict_dict, image_id_to_gt_bboxes_dict,
        image_id_to_gt_classes_dict, image_id_to_gt_masks_dict,
        image_id_to_difficulties_dict, class_to_num_positives_dict)
