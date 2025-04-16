def do_generate(self, imgs, prompt, max_new_tokens, **kwargs):
    imgs = imgs.unsqueeze(0)
    output = self.model.generate({'image': imgs, 'prompt': prompt},
        max_length=max_new_tokens)[0]
    return output
