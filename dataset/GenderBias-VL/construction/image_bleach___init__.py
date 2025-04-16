def __init__(self, args):
    self.args = args
    os.makedirs(os.path.join(exp_dir, self.args.sub_exp), exist_ok=True)
    self.sub_exp_dir = os.path.join(exp_dir, self.args.sub_exp)
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.setup_seed()
    self.build_dataset()
    self.build_model()
