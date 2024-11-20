def forward(self, inputs):
    task_types = []
    output_texts = []
    vision_embeds = []
    vision_masks = []
    vision_types = []
    if inputs['image'] is not None:
        image_inputs = inputs['image']
        task_types += image_inputs['task_type']
        output_texts += image_inputs['output_texts']
        image_paths = image_inputs['vision_paths']
        images = self.load_and_transform_image_data_clip(image_paths, self.
            device).to(self.llama_model.dtype)
        image_embeds = self.encode_image(images)
        vision_embeds.append(image_embeds)
        vision_mask = torch.ones(image_embeds.shape[0], image_embeds.shape[
            1], device=image_embeds.device, dtype=torch.long)
        vision_masks.append(vision_mask)
        vision_types.extend(['image'] * image_embeds.shape[0])
    if inputs['pcl'] is not None:
        pcl_inputs = inputs['pcl']
        task_types += pcl_inputs['task_type']
        output_texts += pcl_inputs['output_texts']
        pcl_embeds_ = self.encode_pcl(pcl_inputs)
        pcl_embeds = torch.zeros(pcl_embeds_.shape[0], self.
            num_vision_token, pcl_embeds_.shape[2], device=pcl_embeds_.
            device, dtype=pcl_embeds_.dtype)
        pcl_embeds[:, :self.args['num_query_rsp_3d']] = pcl_embeds_
        vision_embeds.append(pcl_embeds)
        vision_mask = torch.ones(pcl_embeds.shape[0], self.num_vision_token,
            device=pcl_embeds.device, dtype=torch.long)
        vision_mask[:, self.args['num_query_rsp_3d']:] = 0
        vision_masks.append(vision_mask)
        vision_types.extend(['pcl'] * pcl_embeds.shape[0])
    vision_embeds = torch.cat(vision_embeds, dim=0)
    vision_masks = torch.cat(vision_masks, dim=0)
    inputs_embeds, targets, attention_mask = self.prepare_prompt_embeds(
        vision_embeds, vision_masks, output_texts, vision_types, task_types)
    self.moe_set_gate(output_texts, inputs_embeds.device)
    outputs = self.llama_model(inputs_embeds=inputs_embeds, attention_mask=
        attention_mask, return_dict=True, labels=None)
    logits = outputs.logits
    gen_acc = self.get_acc(logits, targets)
    loss = self.get_loss(logits, targets)
    return loss, gen_acc
