def get_cross_attention_kwargs_without_grounded(self, hidden_size,
    repeat_batch, max_objs, device):
    """
        Prepare the cross-attention kwargs without information about the grounded input (boxes, mask, image embedding,
        phrases embedding) (All are zero tensor).
        """
    boxes = torch.zeros(max_objs, 4, device=device, dtype=self.text_encoder
        .dtype)
    masks = torch.zeros(max_objs, device=device, dtype=self.text_encoder.dtype)
    phrases_masks = torch.zeros(max_objs, device=device, dtype=self.
        text_encoder.dtype)
    image_masks = torch.zeros(max_objs, device=device, dtype=self.
        text_encoder.dtype)
    phrases_embeddings = torch.zeros(max_objs, hidden_size, device=device,
        dtype=self.text_encoder.dtype)
    image_embeddings = torch.zeros(max_objs, hidden_size, device=device,
        dtype=self.text_encoder.dtype)
    out = {'boxes': boxes.unsqueeze(0).repeat(repeat_batch, 1, 1), 'masks':
        masks.unsqueeze(0).repeat(repeat_batch, 1), 'phrases_masks':
        phrases_masks.unsqueeze(0).repeat(repeat_batch, 1), 'image_masks':
        image_masks.unsqueeze(0).repeat(repeat_batch, 1),
        'phrases_embeddings': phrases_embeddings.unsqueeze(0).repeat(
        repeat_batch, 1, 1), 'image_embeddings': image_embeddings.unsqueeze
        (0).repeat(repeat_batch, 1, 1)}
    return out
