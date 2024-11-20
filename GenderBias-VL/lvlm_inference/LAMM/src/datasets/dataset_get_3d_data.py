def get_3d_data(self, index):
    output_texts = self.data_3d['caption_list'][index]
    task_type = self.data_3d['task_type_list'][index]
    vision_embeds_3d_ref = self.data_3d['vision_embeds_3d_ref_list'][index]
    vision_embeds_3d_scene_prop = self.data_3d[
        'vision_embeds_3d_scene_prop_list'][index]
    vision_pos_3d_ref = self.data_3d['vision_pos_3d_ref_list'][index]
    vision_pos_3d_scene_prop = self.data_3d['vision_pos_3d_scene_prop_list'][
        index]
    scene_id = self.data_3d['scene_id_list'][index] if len(self.data_3d[
        'scene_id_list']) > 0 else None
    max_proposal_num = self.data_3d['max_proposal_num']
    vision_embeds_3d_scene_prop_padding = torch.zeros(max_proposal_num,
        vision_embeds_3d_scene_prop.shape[-1])
    vision_embeds_3d_scene_prop_padding[:vision_embeds_3d_scene_prop.shape[0]
        ] = vision_embeds_3d_scene_prop
    vision_pos_3d_scene_prop_padding = torch.zeros(max_proposal_num,
        vision_pos_3d_scene_prop.shape[-1])
    vision_pos_3d_scene_prop_padding[:vision_embeds_3d_scene_prop.shape[0]
        ] = vision_pos_3d_scene_prop
    mask = torch.zeros(max_proposal_num)
    mask[:vision_embeds_3d_scene_prop.shape[0]] = 1
    return dict(output_texts=output_texts, vision_type='pcl', task_type=
        task_type, vision_embeds_3d_ref=vision_embeds_3d_ref.reshape(-1),
        vision_embeds_3d_scene_prop=vision_embeds_3d_scene_prop_padding,
        vision_pos_3d_ref=vision_pos_3d_ref.reshape(-1),
        vision_pos_3d_scene_prop=vision_pos_3d_scene_prop_padding, mask=
        mask, scene_id=scene_id)
