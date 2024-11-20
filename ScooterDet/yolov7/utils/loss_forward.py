@staticmethod
def forward(ctx, logits, targets, delta=1.0):
    classification_grads = torch.zeros(logits.shape).cuda()
    fg_labels = targets == 1
    fg_logits = logits[fg_labels]
    fg_num = len(fg_logits)
    threshold_logit = torch.min(fg_logits) - delta
    relevant_bg_labels = (targets == 0) & (logits >= threshold_logit)
    relevant_bg_logits = logits[relevant_bg_labels]
    relevant_bg_grad = torch.zeros(len(relevant_bg_logits)).cuda()
    rank = torch.zeros(fg_num).cuda()
    prec = torch.zeros(fg_num).cuda()
    fg_grad = torch.zeros(fg_num).cuda()
    max_prec = 0
    order = torch.argsort(fg_logits)
    for ii in order:
        fg_relations = fg_logits - fg_logits[ii]
        fg_relations = torch.clamp(fg_relations / (2 * delta) + 0.5, min=0,
            max=1)
        fg_relations[ii] = 0
        bg_relations = relevant_bg_logits - fg_logits[ii]
        bg_relations = torch.clamp(bg_relations / (2 * delta) + 0.5, min=0,
            max=1)
        rank_pos = 1 + torch.sum(fg_relations)
        FP_num = torch.sum(bg_relations)
        rank[ii] = rank_pos + FP_num
        current_prec = rank_pos / rank[ii]
        if max_prec <= current_prec:
            max_prec = current_prec
            relevant_bg_grad += bg_relations / rank[ii]
        else:
            relevant_bg_grad += bg_relations / rank[ii] * ((1 - max_prec) /
                (1 - current_prec))
        fg_grad[ii] = -(1 - max_prec)
        prec[ii] = max_prec
    classification_grads[fg_labels] = fg_grad
    classification_grads[relevant_bg_labels] = relevant_bg_grad
    classification_grads /= fg_num
    cls_loss = 1 - prec.mean()
    ctx.save_for_backward(classification_grads)
    return cls_loss
