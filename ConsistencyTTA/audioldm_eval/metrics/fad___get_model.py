def __get_model(self, use_pca=False, use_activation=False):
    """
        Params:
        -- x   : Either
            (i)  a string which is the directory of a set of audio files, or
            (ii) a np.ndarray of shape (num_samples, sample_length)
        """
    self.model = torch.hub.load('harritaylor/torchvggish', 'vggish')
    if not use_pca:
        self.model.postprocess = False
    if not use_activation:
        self.model.embeddings = nn.Sequential(*list(self.model.embeddings.
            children())[:-1])
    self.model.eval()
