@torch.no_grad()
def batch_generate(self, modality_inputs, question_list, sys_msg=None,
    dataset_name=None, **kwargs):
    prompts = self.generate_conversation_text(question_list, history=[],
        sys_msg=sys_msg)
    if dataset_name == 'ScienceQA':
        outputs = self.do_generate_vqa(modality_inputs, prompts)
    else:
        outputs = self.do_generate(modality_inputs, prompts)
    return outputs
