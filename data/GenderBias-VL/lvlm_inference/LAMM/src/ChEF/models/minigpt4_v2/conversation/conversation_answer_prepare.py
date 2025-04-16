def answer_prepare(self, conv, img_list, max_new_tokens=300, num_beams=1,
    min_length=1, top_p=0.9, repetition_penalty=1.05, length_penalty=1,
    temperature=1.0, max_length=2000):
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    embs = self.model.get_context_emb(prompt, img_list)
    current_max_len = embs.shape[1] + max_new_tokens
    if current_max_len - max_length > 0:
        print(
            'Warning: The number of tokens in current conversation exceeds the max length. The model will not see the contexts outside the range.'
            )
    begin_idx = max(0, current_max_len - max_length)
    embs = embs[:, begin_idx:]
    generation_kwargs = dict(inputs_embeds=embs, max_new_tokens=
        max_new_tokens, stopping_criteria=self.stopping_criteria, num_beams
        =num_beams, do_sample=True, min_length=min_length, top_p=top_p,
        repetition_penalty=repetition_penalty, length_penalty=
        length_penalty, temperature=float(temperature))
    return generation_kwargs
