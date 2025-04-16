def __init__(self, *args, eval_examples=None, post_process_function=None,
    **kwargs):
    super().__init__(*args, **kwargs)
    self.eval_examples = eval_examples
    self.post_process_function = post_process_function
