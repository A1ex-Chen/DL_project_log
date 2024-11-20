@torch.no_grad()
def do_ppl(self, batch_images, batch_prompt, batch_options, **kwargs):
    batch_images = torch.cat(batch_images, dim=0)
    batch_input_ids, attention_mask = batch_tokenizer_image_token(batch_prompt,
        self.tokenizer)
    batch_option_ids = batch_tokenizer_image_token(batch_options, self.
        tokenizer, add_special_tokens=False).to(self.device)[:, 1:]
    (input_ids, modality_indicators, attention_mask, past_key_values,
        inputs_embeds, labels) = (self.model.
        prepare_inputs_labels_for_multimodal(input_ids=batch_input_ids.to(
        self.device), attention_mask=attention_mask.to(self.device),
        past_key_values=None, labels=batch_input_ids.to(self.device),
        images=batch_images.to(dtype=self.dtype).to(self.device)))
    outputs = self.model.base_model(input_ids=input_ids,
        modality_indicators=modality_indicators, attention_mask=
        attention_mask, past_key_values=past_key_values, inputs_embeds=
        inputs_embeds, use_cache=None, output_attentions=None,
        output_hidden_states=None, return_dict=None)
    hidden_states = outputs[0]
    logits = self.model.lm_head(hidden_states)
    labels = labels[:, 1:]
    results = []
    for idx in range(labels.shape[0]):
        option_len = torch.sum(batch_option_ids[idx] != IGNORE_INDEX).item()
        non_zero_indices = torch.where(labels[idx] != -100)[0]
        start_index = non_zero_indices.max() - option_len + 1
        end_index = start_index + option_len
        prob = F.softmax(logits[idx][start_index:end_index], dim=-1)
        rows = torch.arange(0, option_len)
        score = torch.log(prob[rows, batch_option_ids[idx][:option_len]]).mean(
            ).item()
        results.append(score)
    return results
