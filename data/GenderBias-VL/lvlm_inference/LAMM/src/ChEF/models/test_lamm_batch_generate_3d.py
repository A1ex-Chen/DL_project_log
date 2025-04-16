@torch.no_grad()
def batch_generate_3d(self, modality_inputs, question_list, max_new_tokens=
    128, sys_msg=None, **kwargs):
    prompts = self.generate_conversation_text(question_list, sys_msg=sys_msg)
    outputs = self.do_generate_3d(modality_inputs, prompts, max_new_tokens)
    return outputs
