def __init__(self, model, optimizer, device=None, input_type='img', vis_dir
    =None, threshold=0.3, eval_sample=False, loss_corr=False, loss_corr_bw=
    False, loss_recon=True, vae_beta=0.0001):
    self.model = model
    self.optimizer = optimizer
    self.device = device
    self.input_type = input_type
    self.vis_dir = vis_dir
    self.threshold = threshold
    self.eval_sample = eval_sample
    self.loss_corr = loss_corr
    self.loss_recon = loss_recon
    self.loss_corr_bw = loss_corr_bw
    self.vae_beta = vae_beta
    self.eval_iou = (self.model.decoder is not None and self.model.
        vector_field is not None)
    if vis_dir is not None and not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
