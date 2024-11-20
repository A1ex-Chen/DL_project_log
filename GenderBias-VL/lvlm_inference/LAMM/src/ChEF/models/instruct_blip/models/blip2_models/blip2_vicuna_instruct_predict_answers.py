def predict_answers(self, samples, num_beams=5, inference_method='generate',
    max_len=10, min_len=1, num_ans_candidates=128, answer_list=None, prompt
    ='', length_penalty=0, **kwargs):
    if isinstance(samples['text_input'], str):
        samples['text_input'] = [samples['text_input']]
    if prompt:
        if prompt.count('{}') == 2:
            if 'ocr_tokens' in samples:
                text_input = [prompt.format(', '.join(samples['ocr_tokens']
                    [i][:30]), samples['text_input'][i]) for i in range(len
                    (samples['text_input']))]
            elif 'choices' in samples:
                text_input = []
                for i in range(len(samples['text_input'])):
                    this_choices = [f'({string.ascii_lowercase[j]}) {ch}' for
                        j, ch in enumerate(samples['choices'][i])]
                    this_choices = ' '.join(this_choices)
                    text_input.append(prompt.format(samples['text_input'][i
                        ], this_choices))
        else:
            text_input = [prompt.format(question) for question in samples[
                'text_input']]
    else:
        text_input = samples['text_input']
    samples['prompt'] = text_input
    output_text = self.generate(samples, num_beams=num_beams, max_length=
        max_len, min_length=min_len, length_penalty=length_penalty)
    if 'apply_lemmatizer' in samples.keys() and samples['apply_lemmatizer']:
        output_text = self._lemmatize(output_text)
    return output_text
