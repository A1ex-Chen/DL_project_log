def _build_prompt(self, prompts, tgt_subjects, prompt_strength=1.0,
    prompt_reps=20):
    rv = []
    for prompt, tgt_subject in zip(prompts, tgt_subjects):
        prompt = f'a {tgt_subject} {prompt.strip()}'
        rv.append(', '.join([prompt] * int(prompt_strength * prompt_reps)))
    return rv
