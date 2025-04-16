def distill_loss_cls(self, logits_student, logits_teacher, num_classes,
    temperature=20):
    logits_student = logits_student.view(-1, num_classes)
    logits_teacher = logits_teacher.view(-1, num_classes)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    log_pred_student = torch.log(pred_student)
    d_loss_cls = F.kl_div(log_pred_student, pred_teacher, reduction='sum')
    d_loss_cls *= temperature ** 2
    return d_loss_cls
