def __init__(self, model, vis_processor, device='cuda:0', stopping_criteria
    =None):
    self.device = device
    self.model = model
    self.vis_processor = vis_processor
    if stopping_criteria is not None:
        self.stopping_criteria = stopping_criteria
    else:
        stop_words_ids = [torch.tensor([2]).to(self.device)]
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(
            stops=stop_words_ids)])
