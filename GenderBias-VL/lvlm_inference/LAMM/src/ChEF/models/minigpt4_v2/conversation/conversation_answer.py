def answer(self, conv, img_list, **kargs):
    generation_dict = self.answer_prepare(conv, img_list, **kargs)
    output_token = self.model_generate(**generation_dict)[0]
    output_text = self.model.llama_tokenizer.decode(output_token,
        skip_special_tokens=True)
    output_text = output_text.split('###')[0]
    output_text = output_text.split('Assistant:')[-1].strip()
    conv.messages[-1][1] = output_text
    return output_text, output_token.cpu().numpy()
