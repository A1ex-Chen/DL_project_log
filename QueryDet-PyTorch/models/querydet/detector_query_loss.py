def query_loss(self, gt_small_obj, pred_small_obj, gammas, weights):
    pred_logits = [permute_to_N_HWA_K(x, 1).flatten() for x in pred_small_obj]
    gts = [x.flatten() for x in gt_small_obj]
    loss = sum([(sigmoid_focal_loss_jit(x, y, alpha=0.25, gamma=g,
        reduction='mean') * w) for x, y, g, w in zip(pred_logits, gts,
        gammas, weights)])
    return {'loss_query': loss}
