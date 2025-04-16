def do_generate(self, image_list: torch.Tensor, prompt: str, max_new_tokens,
    **kwargs):
    vision_x = torch.stack([image_list], dim=0)
    vision_x = vision_x.to(self.model.device, dtype=self.dtype)
    lang_x = self.model.text_tokenizer([prompt], return_tensors='pt')
    generated_text = self.model.generate(vision_x=vision_x, lang_x=lang_x[
        'input_ids'].to(self.model.device), attention_mask=lang_x[
        'attention_mask'].to(self.model.device, dtype=self.dtype),
        max_new_tokens=max_new_tokens, num_beams=3, no_repeat_ngram_size=3)
    output = self.model.text_tokenizer.decode(generated_text[0])
    output = [x for x in output.split(' ') if not x.startswith('<')]
    out_label = output.index('GPT:')
    output = ' '.join(output[out_label + 1:])
    return output
