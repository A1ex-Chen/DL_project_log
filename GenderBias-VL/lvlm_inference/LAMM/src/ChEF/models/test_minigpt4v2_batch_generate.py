@torch.no_grad()
def batch_generate(self, batch_images, batch_prompt, max_new_tokens, **kwargs):
    outputs = []
    for idx, (image_list, prompt) in enumerate(zip(batch_images, batch_prompt)
        ):
        input_image_list = self.build_input_image(image_list)
        input_prompt = self.build_conversation(idx, input_image_list,
            prompt, generate=True, **kwargs)
        output = self.do_generate(input_image_list, input_prompt,
            max_new_tokens)
        outputs.append(output)
    return outputs
