@staticmethod
def collate_fn(batch):
    """Collates data samples into batches."""
    new_batch = {}
    keys = batch[0].keys()
    values = list(zip(*[list(b.values()) for b in batch]))
    for i, k in enumerate(keys):
        value = values[i]
        if k == 'img':
            value = torch.stack(value, 0)
        if k in ['masks', 'keypoints', 'bboxes', 'cls']:
            value = torch.cat(value, 0)
        new_batch[k] = value
    new_batch['batch_idx'] = list(new_batch['batch_idx'])
    for i in range(len(new_batch['batch_idx'])):
        new_batch['batch_idx'][i] += i
    new_batch['batch_idx'] = torch.cat(new_batch['batch_idx'], 0)
    targets = torch.cat([new_batch['batch_idx'].unsqueeze(-1), new_batch[
        'cls'], new_batch['bboxes']], axis=1)
    shapes = None
    if 'ratio_pad' in new_batch:
        shapes = [(ori_shape, ratio_pad) for ori_shape, ratio_pad in zip(
            new_batch['ori_shape'], new_batch['ratio_pad'])]
    return new_batch['img'], targets, new_batch['im_file'], shapes
