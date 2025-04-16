def forward(self, pc, text, image=None):
    pc_embed, pc_feat = self.encode_pc(pc)
    if image is not None:
        return {'text_embed': text, 'pc_embed': pc_embed, 'image_embed':
            image, 'pc_feat': pc_feat, 'logit_scale': self.logit_scale.exp()}
    else:
        return {'text_embed': text, 'pc_embed': pc_embed, 'logit_scale':
            self.logit_scale.exp()}
