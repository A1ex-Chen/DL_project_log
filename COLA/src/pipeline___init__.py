def __init__(self, model_path=None, device=None):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    generator = transformers.pipeline('text-generation', model=
        AutoModelForCausalLM.from_pretrained(model_path), tokenizer=
        tokenizer, framework='pt', device=device)
    self.tokenizer = tokenizer
    self.generator = generator
