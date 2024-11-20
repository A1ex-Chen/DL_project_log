def fetch_answer(self, answer, choices):
    if not self.content_only:
        answer = self.preprocess_text(answer)
        copt = self.infer_option(answer)
        if copt:
            return copt, 1, 0
    if answer in choices:
        return self.choices[choices.index(answer)], 0, 1
    return self.infer_text(answer, choices), 0, 1
