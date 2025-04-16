def preprocess_batch(self, batch):
    """Preprocesses a batch of images for YOLOWorld training, adjusting formatting and dimensions as needed."""
    batch = super().preprocess_batch(batch)
    texts = list(itertools.chain(*batch['texts']))
    text_token = self.clip.tokenize(texts).to(batch['img'].device)
    txt_feats = self.text_model.encode_text(text_token).to(dtype=batch[
        'img'].dtype)
    txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
    batch['txt_feats'] = txt_feats.reshape(len(batch['texts']), -1,
        txt_feats.shape[-1])
    return batch
