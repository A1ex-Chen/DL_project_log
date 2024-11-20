@torch.no_grad()
def do_generate_3d(self, modality_inputs, question_list, max_new_tokens=128):
    modality_inputs.update({'top_p': 0.9, 'temperature': 1.0, 'max_tgt_len':
        max_new_tokens, 'modality_embeds': [], 'prompt': question_list})
    outputs = self.model.generate(modality_inputs)
    return [output.split('\n###')[0] for output in outputs]
