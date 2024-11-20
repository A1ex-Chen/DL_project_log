def load_from_pretrain(self, pretrained_checkpoint_path):
    checkpoint = torch.load(pretrained_checkpoint_path)
    self.base.load_state_dict(checkpoint['model'])
