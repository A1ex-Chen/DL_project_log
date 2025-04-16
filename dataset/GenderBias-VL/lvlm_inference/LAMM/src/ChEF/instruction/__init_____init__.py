def __init__(self, prompt, answer_template, incontext_cfg=None, dataset=None):
    self.prompt = prompt
    self.CoT_prompt = "Let's think step by step."
    self.answer_template = answer_template
    self.incontext_cfg = incontext_cfg
    if incontext_cfg:
        self.ice_num = incontext_cfg.get('ice_num', 1)
        if self.ice_num == 0:
            self.incontext_cfg = None
            return
        self.retriever = build_retriever(dataset, dataset, **incontext_cfg)
        self.retriever.seed = incontext_cfg['random_seed']
        self.ice_idx_list = self.retriever.retrieve()
