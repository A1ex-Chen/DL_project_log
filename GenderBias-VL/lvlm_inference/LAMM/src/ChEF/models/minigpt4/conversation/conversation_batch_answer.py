def batch_answer(self, image_list, question_list, chat_list, max_new_tokens
    =300, num_beams=5, min_length=1, top_p=0.9, repetition_penalty=1.0,
    length_penalty=-1.0, temperature=1.0, max_length=2000):
    embs_list = []
    for image, question, conv in zip(image_list, question_list, chat_list):
        img_list = []
        self.upload_img(image, conv, img_list)
        self.ask(question, conv)
        conv.append_message(conv.roles[1], None)
        embs = self.get_context_emb(conv, img_list)
        embs_list.append(embs)
    max_emb_token = max([x.shape[1] for x in embs_list])
    embs_list = torch.cat([F.pad(x, (0, 0, max_emb_token - x.shape[1], 0, 0,
        0), value=0) for x in embs_list], dim=0)
    assert max_emb_token + max_new_tokens < max_length
    outputs = self.model.llama_model.generate(inputs_embeds=embs_list,
        max_new_tokens=max_new_tokens, stopping_criteria=self.
        stopping_criteria, num_beams=num_beams, do_sample=False, min_length
        =min_length, top_p=top_p, repetition_penalty=repetition_penalty,
        length_penalty=length_penalty, temperature=temperature)
    batch_outputs = []
    for output_token in outputs:
        if output_token[0] == 0:
            output_token = output_token[1:]
        if output_token[0] == 1:
            output_token = output_token[1:]
        output_text = self.model.llama_tokenizer.decode(output_token,
            add_special_tokens=False)
        output_text = output_text.split('###')[0]
        output_text = output_text.split('Assistant:')[-1].strip()
        batch_outputs.append(output_text)
    return batch_outputs
