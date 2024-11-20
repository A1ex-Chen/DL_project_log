def __init__(self, model, vis_processor, device='cuda:0'):
    self.device = device
    self.model = model
    self.vis_processor = vis_processor
    self.stop_words_ids = [torch.tensor([835]).to(self.device), torch.
        tensor([2277, 29937]).to(self.device)]
    self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(
        stops=self.stop_words_ids)])
