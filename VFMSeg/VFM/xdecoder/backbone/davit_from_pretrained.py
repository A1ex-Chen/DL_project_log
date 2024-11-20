def from_pretrained(self, pretrained='', pretrained_layers=[], verbose=True):
    if os.path.isfile(pretrained):
        print(f'=> loading pretrained model {pretrained}')
        pretrained_dict = torch.load(pretrained, map_location='cpu')
        self.from_state_dict(pretrained_dict, pretrained_layers, verbose)
