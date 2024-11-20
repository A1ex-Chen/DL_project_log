def answer(self, conv, img_list, max_new_tokens=300, num_beams=1,
    min_length=1, top_p=0.9, repetition_penalty=1.0, length_penalty=1,
    temperature=1.0, max_length=2000):
    conv.append_message(conv.roles[1], None)
    embs = self.get_context_emb(conv, img_list)
    current_max_len = embs.shape[1] + max_new_tokens
    if current_max_len - max_length > 0:
        print(
            'Warning: The number of tokens in current conversation exceeds the max length. The model will not see the contexts outside the range.'
            )
    begin_idx = max(0, current_max_len - max_length)
    embs = embs[:, begin_idx:]
    outputs = self.model.llama_model.generate(inputs_embeds=embs,
        max_new_tokens=max_new_tokens, stopping_criteria=self.
        stopping_criteria, num_beams=num_beams, do_sample=False, min_length
        =min_length, top_p=top_p, repetition_penalty=repetition_penalty,
        length_penalty=length_penalty, temperature=temperature)
    output_token = outputs[0]
    if output_token[0] == 0:
        output_token = output_token[1:]
    if output_token[0] == 1:
        output_token = output_token[1:]
    output_text = self.model.llama_tokenizer.decode(output_token,
        add_special_tokens=False)
    output_text = output_text.split('###')[0]
    output_text = output_text.split('Assistant:')[-1].strip()
    conv.messages[-1][1] = output_text
    return output_text, output_token.cpu().numpy()
