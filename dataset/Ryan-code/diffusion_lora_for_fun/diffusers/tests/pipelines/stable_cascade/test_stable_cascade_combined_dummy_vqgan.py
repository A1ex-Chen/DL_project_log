@property
def dummy_vqgan(self):
    torch.manual_seed(0)
    model_kwargs = {'bottleneck_blocks': 1, 'num_vq_embeddings': 2}
    model = PaellaVQModel(**model_kwargs)
    return model.eval()
