def post_processor(self):
    return processors.TemplateProcessing(single=['$A', '</s>'], pair=['$A',
        '</s>', '$B', '</s>'], special_tokens=[('</s>', self.
        original_tokenizer.convert_tokens_to_ids('</s>'))])
