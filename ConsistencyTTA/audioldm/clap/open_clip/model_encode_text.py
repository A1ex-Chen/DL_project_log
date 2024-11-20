def encode_text(self, text, device):
    if self.text_branch_type == 'transformer':
        text = text.to(device=device, non_blocking=True)
        x = self.token_embedding(text)
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)
        x = self.text_branch(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        x = self.text_projection(x[torch.arange(x.shape[0]), text.argmax(
            dim=-1)])
    elif self.text_branch_type == 'bert':
        x = self.text_branch(input_ids=text['input_ids'].to(device=device,
            non_blocking=True), attention_mask=text['attention_mask'].to(
            device=device, non_blocking=True), token_type_ids=text[
            'token_type_ids'].to(device=device, non_blocking=True))[
            'pooler_output']
        x = self.text_projection(x)
    elif self.text_branch_type == 'roberta':
        x = self.text_branch(input_ids=text['input_ids'].to(device=device,
            non_blocking=True), attention_mask=text['attention_mask'].to(
            device=device, non_blocking=True))['pooler_output']
        x = self.text_projection(x)
    elif self.text_branch_type == 'bart':
        x = torch.mean(self.text_branch(input_ids=text['input_ids'].to(
            device=device, non_blocking=True), attention_mask=text[
            'attention_mask'].to(device=device, non_blocking=True))[
            'encoder_last_hidden_state'], axis=1)
        x = self.text_projection(x)
    else:
        logging.error(f'Model type {self.text_branch_type} not found')
        raise RuntimeError(f'Model type {self.text_branch_type} not found.')
    return x
