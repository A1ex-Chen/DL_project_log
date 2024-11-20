def compute_kd_loss(inputs, outputs, model, model_kd, proba_distribution_fn
    =None):
    if proba_distribution_fn is None:
        proba_distribution_fn = partial(F.softmax, dim=-1)
    with torch.no_grad():
        student_preds = proba_distribution_fn(outputs)
        input_kd = model_kd.normalize_input(inputs, model)
        outputs_teacher = model_kd.model(input_kd.detach())
        teacher_preds = proba_distribution_fn(outputs_teacher)
        kd_loss = F.kl_div(torch.log(student_preds), teacher_preds,
            reduction='batchmean')
    return kd_loss
