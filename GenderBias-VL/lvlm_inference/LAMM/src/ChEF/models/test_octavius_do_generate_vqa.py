@torch.no_grad()
def do_generate_vqa(self, modality_inputs, question_list, top_p=0.9,
    temperature=1.0):
    conv = conv_templates[self.conv_mode]
    reasoning_list = self.do_generate(modality_inputs, question_list)
    option_prompt = []
    for prompt_1, response_1 in zip(question_list, reasoning_list):
        option_prompt.append(prompt_1 + response_1 +
            f""" {conv.sep2 if conv.sep2 is not None else conv.sep}
ANSWER:""")
    final_answer_list = self.do_generate(modality_inputs, option_prompt)
    all_answer_list = []
    for reasoning, option in zip(reasoning_list, final_answer_list):
        all_answer_list.append(reasoning + '\n The answer is ' + option)
    return all_answer_list
