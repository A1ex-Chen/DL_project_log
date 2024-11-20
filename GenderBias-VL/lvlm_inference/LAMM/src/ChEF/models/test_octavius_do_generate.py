@torch.no_grad()
def do_generate(self, modality_inputs, question_list, top_p=0.9,
    temperature=1.0):
    response = self.model.generate({'prompt': question_list, 'image_paths':
        modality_inputs, 'top_p': top_p, 'temperature': temperature,
        'max_tgt_len': self.max_tgt_len, 'modality_embeds': []})
    conv = conv_templates[self.conv_mode]
    ans_list = []
    for res in response:
        ans_list.append(res.split(conv.sep2 if conv.sep2 is not None else
            conv.sep)[0])
    return ans_list
