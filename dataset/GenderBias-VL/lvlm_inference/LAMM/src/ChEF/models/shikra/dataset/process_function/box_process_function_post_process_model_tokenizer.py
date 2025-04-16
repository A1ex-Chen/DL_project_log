def post_process_model_tokenizer(self, model, preprocessor, model_args,
    training_args):
    tokenizer = preprocessor['text']
    additional_special_tokens = [self.box_begin, self.box_sep, self.box_end,
        self.point_begin, self.point_sep, self.point_end]
    for i in range(self.num_bins):
        additional_special_tokens.append(f'<bin_{i}>')
    smart_tokenizer_and_embedding_resize({'additional_special_tokens':
        additional_special_tokens}, tokenizer, model)
    return model, preprocessor
