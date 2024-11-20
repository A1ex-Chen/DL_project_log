def build_inputs(self, tokenizer, query: str, history: List[Tuple[str, str]
    ]=[], meta_instruction=''):
    prompt = ''
    if meta_instruction:
        prompt += (
            f'<s>[UNUSED_TOKEN_146]system\n{meta_instruction}[UNUSED_TOKEN_145]\n'
            )
    else:
        prompt += '<s>'
    for record in history:
        prompt += f"""[UNUSED_TOKEN_146]user
{record[0]}[UNUSED_TOKEN_145]
[UNUSED_TOKEN_146]assistant
{record[1]}[UNUSED_TOKEN_145]
"""
    prompt += (
        f'[UNUSED_TOKEN_146]user\n{query}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n'
        )
    return tokenizer([prompt], return_tensors='pt')
