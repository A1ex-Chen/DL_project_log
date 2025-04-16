@torch.no_grad()
def _dequeue_and_enqueue(self, image_feat, text_feat, idxs=None):
    image_feats = concat_all_gather(image_feat)
    text_feats = concat_all_gather(text_feat)
    batch_size = image_feats.shape[0]
    ptr = int(self.queue_ptr)
    assert self.queue_size % batch_size == 0
    self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
    self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
    if idxs is not None:
        idxs = concat_all_gather(idxs)
        self.idx_queue[:, ptr:ptr + batch_size] = idxs.T
    ptr = (ptr + batch_size) % self.queue_size
    self.queue_ptr[0] = ptr
