def __init__(self, device) ->None:
    self.device = device
    self.model, self.vis_processors, _ = load_model_and_preprocess(name=
        'blip2_vicuna_instruct', model_type='vicuna7b', is_eval=True,
        device=self.device)
    self.tokenizer = self.model.llm_tokenizer
    self.model.max_txt_len = 512
