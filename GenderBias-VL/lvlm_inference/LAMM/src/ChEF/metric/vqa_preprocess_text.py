def preprocess_text(self, answer):
    output_text = answer
    output_text = output_text.split('###')[0]
    output_text = output_text.split('Assistant:')[-1].strip()
    output_text = output_text.strip()
    pattern = re.compile('([A-Z]\\.)')
    res = pattern.findall(output_text)
    if len(res) > 0:
        return '(' + res[0][:-1] + ')'
    pattern = re.compile('\\([A-Z]')
    res = pattern.findall(output_text)
    if len(res) > 0:
        return res[0] + ')'
    return output_text
