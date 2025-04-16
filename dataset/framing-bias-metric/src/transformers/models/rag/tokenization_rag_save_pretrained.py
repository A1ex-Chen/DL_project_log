def save_pretrained(self, save_directory):
    if os.path.isfile(save_directory):
        raise ValueError('Provided path ({}) should be a directory, not a file'
            .format(save_directory))
    os.makedirs(save_directory, exist_ok=True)
    question_encoder_path = os.path.join(save_directory,
        'question_encoder_tokenizer')
    generator_path = os.path.join(save_directory, 'generator_tokenizer')
    self.question_encoder.save_pretrained(question_encoder_path)
    self.generator.save_pretrained(generator_path)
