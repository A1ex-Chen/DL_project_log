def _encode_image_with_pose(self, image, pose, device,
    num_images_per_prompt, do_classifier_free_guidance):
    img_prompt_embeds = self._encode_image(image, device,
        num_images_per_prompt, False)
    pose_prompt_embeds = self._encode_pose(pose, device,
        num_images_per_prompt, False)
    prompt_embeds = torch.cat([img_prompt_embeds, pose_prompt_embeds], dim=-1)
    prompt_embeds = self.cc_projection(prompt_embeds)
    if do_classifier_free_guidance:
        negative_prompt = torch.zeros_like(prompt_embeds)
        prompt_embeds = torch.cat([negative_prompt, prompt_embeds])
    return prompt_embeds
