@torch.no_grad()
def batch_generate_3d(self, modality_inputs, question_list, sys_msg=None,
    **kwargs):
    prompts = self.generate_conversation_text(question_list, history=[],
        sys_msg=sys_msg)
    outputs = self.do_generate_3d(modality_inputs, prompts)
    return outputs
