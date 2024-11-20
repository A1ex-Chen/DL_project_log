def encode_pcl(self, inputs):
    task_type = inputs['task_type']
    mask = inputs['mask'].to(self.device)
    ref_object_feat_3d = inputs['vision_embeds_3d_ref'].to(self.device)
    ref_object_pos_3d = inputs['vision_pos_3d_ref'].to(self.device)
    vision_embeds_3d_scene_prop = inputs['vision_embeds_3d_scene_prop'].to(self
        .device)
    vision_pos_3d_scene_prop = inputs['vision_pos_3d_scene_prop'].to(self.
        device)
    for i, task_type_i in enumerate(task_type):
        if task_type_i == 'classification3d' or task_type_i == 'description3d':
            vision_embeds_3d_scene_prop[i] = torch.zeros((
                vision_embeds_3d_scene_prop.shape[1], ref_object_feat_3d[i]
                .shape[0]), dtype=torch.float16).to(self.device)
            vision_embeds_3d_scene_prop[i, 0] = ref_object_feat_3d[i]
            vision_pos_3d_scene_prop[i] = torch.zeros((
                vision_embeds_3d_scene_prop.shape[1], ref_object_pos_3d[i].
                shape[0]), dtype=torch.float16).to(self.device)
            vision_pos_3d_scene_prop[i, 0] = ref_object_pos_3d[i]
            mask[i, :] = 0
            mask[i, 0] = 1
    vision_embeds_3d = vision_embeds_3d_scene_prop
    vision_embeds_3d_pos = vision_pos_3d_scene_prop
    pos_embed_3d = self.pos_3d_proj(vision_embeds_3d_pos)
    vision_embeds_3d = vision_embeds_3d + pos_embed_3d
    mask = mask.unsqueeze(1).repeat(1, self.args['num_query_rsp_3d'], 1
        ).unsqueeze(1).repeat(1, self.args['num_heads_rsp_3d'], 1, 1)
    vision_embeds_3d = self.resampler_3d(vision_embeds_3d, mask)
    vision_embeds_3d = self.llama_proj_3d(vision_embeds_3d)
    return vision_embeds_3d
