def compute_objectness_and_cls_prob(self, cls_logits):
    assert cls_logits.shape[-1] == self.dataset_config.num_semcls + 1
    cls_prob = torch.nn.functional.softmax(cls_logits, dim=-1)
    objectness_prob = 1 - cls_prob[..., -1]
    return cls_prob[..., :-1], objectness_prob
