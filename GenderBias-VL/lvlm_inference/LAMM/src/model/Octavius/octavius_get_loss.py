def get_loss(self, logits, targets):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = targets[..., 1:].contiguous()
    shift_logits = shift_logits.view(-1, self.llama_model.config.vocab_size)
    shift_labels = shift_labels.view(-1)
    shift_labels = shift_labels.to(shift_logits.device)
    ar_loss_fct = nn.CrossEntropyLoss(reduction='none')
    loss = ar_loss_fct(shift_logits, shift_labels)
    return loss.mean()
