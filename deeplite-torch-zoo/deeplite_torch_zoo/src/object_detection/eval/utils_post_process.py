def post_process(preds):
    preds = preds.cpu().numpy()
    bboxes_batch = []
    for pred in preds:
        sample = post_process_sample(pred)
        sample = np.expand_dims(sample, axis=0)
        bboxes_batch.append(sample)
    bboxes_batch = np.concatenate(bboxes_batch, axis=0)
    return bboxes_batch
