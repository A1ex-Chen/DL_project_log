@torch.no_grad()
def do_generate_3d(self, modality_inputs, question_list, top_p=0.9,
    temperature=1.0):
    modality_inputs.update({'top_p': top_p, 'temperature': temperature,
        'max_tgt_len': self.max_tgt_len, 'modality_embeds': [], 'prompt':
        question_list})
    response = self.model.generate(modality_inputs)
    conv = conv_templates[self.conv_mode]
    ans_list = []
    for res in response:
        ans_list.append(res.split(conv.sep2 if conv.sep2 is not None else
            conv.sep)[0])
    return ans_list
