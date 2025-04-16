def do_generate(self, image_list: list, prompt: str, max_new_tokens, **kwargs):
    outputs = self.model.generate({'prompt': [prompt], 'images': [
        image_list], 'top_p': 0.9, 'temperature': 1.0, 'max_tgt_len':
        max_new_tokens, 'modality_embeds': []})
    return outputs[0].split('\n###')[0]
