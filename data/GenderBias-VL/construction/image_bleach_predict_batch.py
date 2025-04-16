def predict_batch(self, model, image, captions, box_threshold,
    text_threshold, device='cpu'):
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image, captions=captions)
    _logits = outputs['pred_logits'].cpu().sigmoid()
    _boxes = outputs['pred_boxes'].cpu()
    prediction_logits = _logits.clone()
    prediction_boxes = _boxes.clone()
    mask = prediction_logits.max(dim=2)[0] > box_threshold
    bboxes_batch = []
    predicts_batch = []
    phrases_batch = []
    tokenizer = model.tokenizer
    tokenized = tokenizer(captions[0])
    for i in range(prediction_logits.shape[0]):
        logits = prediction_logits[i][mask[i]]
        phrases = [get_phrases_from_posmap(logit > text_threshold,
            tokenized, tokenizer).replace('.', '') for logit in logits]
        boxes = prediction_boxes[i][mask[i]]
        phrases_batch.append(phrases)
        bboxes_batch.append(boxes)
        predicts_batch.append(logits.max(dim=1)[0])
    return bboxes_batch, predicts_batch, phrases_batch
