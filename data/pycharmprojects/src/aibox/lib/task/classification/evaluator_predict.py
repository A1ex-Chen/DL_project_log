@torch.no_grad()
def predict(self, model: Model) ->Prediction:
    model = BunchDataParallel(model)
    all_image_ids, all_pred_classes, all_pred_probs, all_gt_classes = [], [], [
        ], []
    for _, item_batch in enumerate(tqdm(self._dataloader, mininterval=10)):
        processed_image_batch = Bunch([it.processed_image for it in item_batch]
            )
        pred_prob_batch, pred_class_batch = model.eval().forward(
            processed_image_batch)
        for b, item in enumerate(item_batch):
            item: Dataset.Item
            image_id = item.image_id
            pred_class = pred_class_batch[b]
            pred_prob = pred_prob_batch[b]
            gt_class = item.cls
            all_image_ids.append(image_id)
            all_pred_classes.append(pred_class.cpu())
            all_pred_probs.append(pred_prob.cpu())
            all_gt_classes.append(gt_class)
    all_pred_classes = torch.stack(all_pred_classes, dim=0)
    all_pred_probs = torch.stack(all_pred_probs, dim=0)
    all_gt_classes = torch.stack(all_gt_classes, dim=0)
    sorted_indices = all_pred_probs.argsort(dim=-1, descending=True)
    sorted_all_image_ids = [all_image_ids[i.item()] for i in sorted_indices]
    sorted_all_pred_classes = all_pred_classes[sorted_indices]
    sorted_all_pred_probs = all_pred_probs[sorted_indices]
    sorted_all_gt_classes = all_gt_classes[sorted_indices]
    return Evaluator.Prediction(sorted_all_image_ids,
        sorted_all_pred_classes, sorted_all_pred_probs, sorted_all_gt_classes)
