def embedding_batch(self, list_inputText, max_length=64, contextual=False):
    list_text = []
    list_attn_mask = []
    for text in list_inputText:
        ret = self.tokenizer.encode_plus(text=text, max_length=max_length,
            padding='max_length', truncation=True, add_special_tokens=True)
        list_text.append(ret['input_ids'])
        list_attn_mask.append(ret['attention_mask'])
    list_text = torch.as_tensor(list_text)
    list_attn_mask = torch.as_tensor(list_attn_mask)
    list_text = list_text.to(self.device)
    list_attn_mask = list_attn_mask.to(self.device)
    outputs = self.model(input_ids=list_text, attention_mask=list_attn_mask,
        output_hidden_states=True)
    hidden_states = outputs[2]
    last_layer = hidden_states[-1]
    unsqueeze_mask = torch.unsqueeze(list_attn_mask, dim=-1)
    masked_last_layer = last_layer.cpu() * unsqueeze_mask.cpu()
    if contextual:
        del outputs
        del hidden_states
        del last_layer
        del unsqueeze_mask
        torch.cuda.empty_cache()
        return masked_last_layer
    sum_mask = torch.sum(list_attn_mask.cpu(), dim=-1, keepdim=True)
    mean_last = torch.sum(masked_last_layer, dim=1) / sum_mask
    del outputs
    del hidden_states
    del last_layer
    del unsqueeze_mask
    torch.cuda.empty_cache()
    return mean_last
