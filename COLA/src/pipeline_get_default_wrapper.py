@staticmethod
def get_default_wrapper(cls=0):
    return AllenSRLWrapper(allennlp_models.pretrained.load_predictor(
        'structured-prediction-srl-bert', cuda_device=cls))
