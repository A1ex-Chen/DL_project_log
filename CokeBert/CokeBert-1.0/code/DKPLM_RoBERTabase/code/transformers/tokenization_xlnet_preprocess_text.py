def preprocess_text(self, inputs):
    if self.remove_space:
        outputs = ' '.join(inputs.strip().split())
    else:
        outputs = inputs
    outputs = outputs.replace('``', '"').replace("''", '"')
    if six.PY2 and isinstance(outputs, str):
        outputs = outputs.decode('utf-8')
    if not self.keep_accents:
        outputs = unicodedata.normalize('NFKD', outputs)
        outputs = ''.join([c for c in outputs if not unicodedata.combining(c)])
    if self.do_lower_case:
        outputs = outputs.lower()
    return outputs
