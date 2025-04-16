def init_parameters(self):
    if self.attnpool is not None:
        std = self.attnpool.c_proj.in_features ** -0.5
        nn.init.normal_(self.attnpool.q_proj.weight, std=std)
        nn.init.normal_(self.attnpool.k_proj.weight, std=std)
        nn.init.normal_(self.attnpool.v_proj.weight, std=std)
        nn.init.normal_(self.attnpool.c_proj.weight, std=std)
    for resnet_block in [self.layer1, self.layer2, self.layer3, self.layer4]:
        for name, param in resnet_block.named_parameters():
            if name.endswith('bn3.weight'):
                nn.init.zeros_(param)
