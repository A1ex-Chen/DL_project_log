def do_generate(self, input_image_list: list, input_prompt: str,
    max_new_tokens, **kwargs):
    img_embes_list = []
    for img_tensor in input_image_list:
        image_emb, _ = self.model.encode_img(img_tensor)
        img_embes_list.append(image_emb)
    embes, _ = self.chat.get_context_emb(input_prompt, img_embes_list)
    outputs = self.model.llama_model.generate(inputs_embeds=embes,
        max_new_tokens=max_new_tokens, stopping_criteria=self.chat.
        stopping_criteria, num_beams=5, do_sample=False, min_length=1,
        top_p=0.9, repetition_penalty=1.0, length_penalty=-1.0, temperature=1.0
        )
    output_token = outputs[0]
    if output_token[0] == 0:
        output_token = output_token[1:]
    if output_token[0] == 1:
        output_token = output_token[1:]
    output_text = self.model.llama_tokenizer.decode(output_token,
        add_special_tokens=False)
    output_text = output_text.split('###')[0]
    output_text = output_text.split('Assistant:')[-1].strip()
    return output_text
