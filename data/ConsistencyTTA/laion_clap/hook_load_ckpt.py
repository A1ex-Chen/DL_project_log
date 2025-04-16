def load_ckpt(self, ckpt=None, model_id=-1, verbose=False):
    """Load the pretrained checkpoint of CLAP model

        Parameters
        ----------
        ckpt: str
            if ckpt is specified, the model will load this ckpt, otherwise the model will download the ckpt from zenodo. 
 
            For fusion model, it will download the 630k+audioset fusion model (id=3). For non-fusion model, it will download the 630k+audioset model (id=1).
        model_id:
            if model_id is specified, you can download our best ckpt, as:
                id = 0 --> 630k non-fusion ckpt 

                id = 1 --> 630k+audioset non-fusion ckpt 

                id = 2 --> 630k fusion ckpt 

                id = 3 --> 630k+audioset fusion ckpt 

            Note that if your model is specied as non-fusion model but you download a fusion model ckpt, you will face an error.
        """
    download_link = 'https://huggingface.co/lukewys/laion_clap/resolve/main/'
    download_names = ['630k-best.pt', '630k-audioset-best.pt',
        '630k-fusion-best.pt', '630k-audioset-fusion-best.pt']
    if ckpt is not None:
        print(f'Load the specified checkpoint {ckpt} from users.')
    else:
        print(f'Load our best checkpoint in the paper.')
        if model_id == -1:
            model_id = 3 if self.enable_fusion else 1
        package_dir = os.path.dirname(os.path.realpath(__file__))
        weight_file_name = download_names[model_id]
        ckpt = os.path.join(package_dir, weight_file_name)
        if os.path.exists(ckpt):
            print(f'The checkpoint is already downloaded')
        else:
            print('Downloading laion_clap weight files...')
            ckpt = wget.download(download_link + weight_file_name, os.path.
                dirname(ckpt))
            print('Download completed!')
    print('Loading LAION-CLAP Checkpoint...')
    ckpt = load_state_dict(ckpt, skip_params=True)
    self.model.load_state_dict(ckpt)
    if verbose:
        param_names = [n for n, p in self.model.named_parameters()]
        for n in param_names:
            print(n, '\t', 'Loaded' if n in ckpt else 'Unloaded')
