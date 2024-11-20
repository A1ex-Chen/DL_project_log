def get_model(model_path):
    return spm.SentencePieceProcessor(model_file=model_path)
