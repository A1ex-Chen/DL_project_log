def __init__(self, enable_fusion=False, device=None, amodel='HTSAT-tiny',
    tmodel='roberta') ->None:
    """Initialize CLAP Model

        Parameters
        ----------
        enable_fusion: bool
            if true, it will create the fusion clap model, otherwise non-fusion clap model (default: false) 
        device: str
            if None, it will automatically detect the device (gpu or cpu)
        amodel: str
            audio encoder architecture, default: HTSAT-tiny
        tmodel: str
            text encoder architecture, default: roberta
        """
    super(CLAP_Module, self).__init__()
    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    precision = 'fp32'
    if enable_fusion:
        fusion_type = 'aff_2d'
        model, model_cfg = create_model(amodel, tmodel, precision=precision,
            device=device, enable_fusion=enable_fusion, fusion_type=fusion_type
            )
    else:
        model, model_cfg = create_model(amodel, tmodel, precision=precision,
            device=device, enable_fusion=enable_fusion)
    self.enable_fusion = enable_fusion
    self.model = model
    self.model_cfg = model_cfg
    self.tokenize = RobertaTokenizer.from_pretrained('roberta-base')
