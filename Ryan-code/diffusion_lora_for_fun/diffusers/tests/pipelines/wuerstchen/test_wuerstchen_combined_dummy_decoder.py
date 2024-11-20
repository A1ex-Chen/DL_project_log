@property
def dummy_decoder(self):
    torch.manual_seed(0)
    model_kwargs = {'c_cond': self.text_embedder_hidden_size, 'c_hidden': [
        320], 'nhead': [-1], 'blocks': [4], 'level_config': ['CT'],
        'clip_embd': self.text_embedder_hidden_size, 'inject_effnet': [False]}
    model = WuerstchenDiffNeXt(**model_kwargs)
    return model.eval()
