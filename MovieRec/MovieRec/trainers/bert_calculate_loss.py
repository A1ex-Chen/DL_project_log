def calculate_loss(self, batch):
    seqs, labels = batch
    logits = self.model(seqs)
    logits = logits.view(-1, logits.size(-1))
    labels = labels.view(-1)
    loss = self.ce(logits, labels)
    return loss
