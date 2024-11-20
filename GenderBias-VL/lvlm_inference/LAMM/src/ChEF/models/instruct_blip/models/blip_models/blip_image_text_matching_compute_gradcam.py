def compute_gradcam(model, visual_input, text_input, tokenized_text,
    block_num=6):
    model.text_encoder.base_model.base_model.encoder.layer[block_num
        ].crossattention.self.save_attention = True
    output = model({'image': visual_input, 'text_input': text_input},
        match_head='itm')
    loss = output[:, 1].sum()
    model.zero_grad()
    loss.backward()
    with torch.no_grad():
        mask = tokenized_text.attention_mask.view(tokenized_text.
            attention_mask.size(0), 1, -1, 1, 1)
        token_length = tokenized_text.attention_mask.sum(dim=-1) - 2
        token_length = token_length.cpu()
        grads = model.text_encoder.base_model.base_model.encoder.layer[
            block_num].crossattention.self.get_attn_gradients()
        cams = model.text_encoder.base_model.base_model.encoder.layer[block_num
            ].crossattention.self.get_attention_map()
        cams = cams[:, :, :, 1:].reshape(visual_input.size(0), 12, -1, 24, 24
            ) * mask
        grads = grads[:, :, :, 1:].clamp(0).reshape(visual_input.size(0), 
            12, -1, 24, 24) * mask
        gradcams = cams * grads
        gradcam_list = []
        for ind in range(visual_input.size(0)):
            token_length_ = token_length[ind]
            gradcam = gradcams[ind].mean(0).cpu().detach()
            gradcam = torch.cat((gradcam[0:1, :], gradcam[1:token_length_ +
                1, :].sum(dim=0, keepdim=True) / token_length_, gradcam[1:, :])
                )
            gradcam_list.append(gradcam)
    return gradcam_list, output
