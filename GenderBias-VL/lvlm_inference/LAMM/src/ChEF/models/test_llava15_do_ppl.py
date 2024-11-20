@torch.no_grad()
def do_ppl(self, batch_images, batch_prompt, batch_options, **kwargs):
    batch_input_ids = batch_tokenizer_image_token(batch_prompt, self.tokenizer
        ).to(self.device)
    batch_option_ids = batch_tokenizer_image_token(batch_options, self.
        tokenizer, add_special_tokens=False).to(self.device)
    (input_ids, position_ids, attention_mask, past_key_values,
        inputs_embeds, labels) = (self.model.
        prepare_inputs_labels_for_multimodal(batch_input_ids, None, None,
        None, batch_input_ids, batch_images))
    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
        position_ids=position_ids, past_key_values=past_key_values,
        inputs_embeds=inputs_embeds, labels=labels, use_cache=None,
        output_attentions=None, output_hidden_states=None, return_dict=None)
    logits = outputs.logits
    logits = logits[:, :-1].float()
    labels = labels[:, 1:]
    results = []
    for idx in range(labels.shape[0]):
        option_len = torch.sum(batch_option_ids[idx] != IGNORE_INDEX).item()
        non_zero_indices = torch.nonzero(labels[idx], as_tuple=False).squeeze()
        start_index = non_zero_indices.max() - option_len + 1
        end_index = start_index + option_len
        prob = F.softmax(logits[idx][start_index:end_index], dim=-1)
        rows = torch.arange(0, option_len)
        score = torch.log(prob[rows, batch_option_ids[idx][:option_len]]).mean(
            ).item()
        results.append(score)
    return results
