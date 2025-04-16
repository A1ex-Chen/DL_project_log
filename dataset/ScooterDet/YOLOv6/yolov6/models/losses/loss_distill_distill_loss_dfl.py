def distill_loss_dfl(self, logits_student, logits_teacher, temperature=20):
    logits_student = logits_student.view(-1, 17)
    logits_teacher = logits_teacher.view(-1, 17)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    log_pred_student = torch.log(pred_student)
    d_loss_dfl = F.kl_div(log_pred_student, pred_teacher, reduction='none'
        ).sum(1).mean()
    d_loss_dfl *= temperature ** 2
    return d_loss_dfl
