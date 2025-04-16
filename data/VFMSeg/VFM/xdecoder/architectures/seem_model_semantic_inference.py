def semantic_inference(self, mask_cls, mask_pred):
    mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
    mask_pred = mask_pred.sigmoid()
    semseg = torch.einsum('qc,qhw->chw', mask_cls, mask_pred)
    return semseg
