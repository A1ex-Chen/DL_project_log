def __init__(self, use_pca=False, use_activation=False, audio_load_worker=8):
    self.__get_model(use_pca=use_pca, use_activation=use_activation)
    self.audio_load_worker = audio_load_worker
