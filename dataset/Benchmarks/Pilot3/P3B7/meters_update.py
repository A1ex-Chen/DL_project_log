def update(self, logits, target):
    for task, logit in logits.items():
        pred = logit.argmax(dim=1, keepdim=True)
        correct = pred.eq(target[task].view_as(pred)).sum().item()
        self.correct[task] += correct
