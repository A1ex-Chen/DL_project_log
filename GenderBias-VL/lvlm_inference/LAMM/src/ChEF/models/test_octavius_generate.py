@torch.no_grad()
def generate(self, modality_input, question, sys_msg=None, dataset_name=
    None, task_name=None, **kwargs):
    prompts = self.generate_conversation_text([question], history=[],
        sys_msg=sys_msg)
    if task_name.endswith('octavius3d'):
        outputs = self.do_generate_3d(modality_input, prompts)
    elif dataset_name == 'ScienceQA':
        outputs = self.do_generate_vqa([modality_input], prompts)
    else:
        outputs = self.do_generate([modality_input], prompts)
    return outputs[0]
