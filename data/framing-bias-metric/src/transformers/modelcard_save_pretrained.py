def save_pretrained(self, save_directory_or_file):
    """Save a model card object to the directory or file `save_directory_or_file`."""
    if os.path.isdir(save_directory_or_file):
        output_model_card_file = os.path.join(save_directory_or_file,
            MODEL_CARD_NAME)
    else:
        output_model_card_file = save_directory_or_file
    self.to_json_file(output_model_card_file)
    logger.info('Model card saved in {}'.format(output_model_card_file))
