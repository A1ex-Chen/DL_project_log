@torch.no_grad()
def ppl_inference(self, batch_images, batch_prompt, batch_options, **kwargs):
    """
            process a batch of images and questions, and then do_ppl
        """
    input_images, input_prompts = [], []
    for idx, (image_list, prompt) in enumerate(zip(batch_images, batch_prompt)
        ):
        input_prompt = self.build_conversation(idx, image_list, prompt,
            generate=False, **kwargs)
        input_image_list = self.build_input_image(image_list)
        input_prompts.append(input_prompt)
        input_images.append(input_image_list)
    return self.do_ppl(input_images, input_prompts, batch_options, **kwargs)
