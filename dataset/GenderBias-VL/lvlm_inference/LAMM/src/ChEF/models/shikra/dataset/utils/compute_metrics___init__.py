def __init__(self, preprocessor: Dict[str, Any]):
    self.preprocessor = preprocessor
    self.tokenizer = self.preprocessor['text']
