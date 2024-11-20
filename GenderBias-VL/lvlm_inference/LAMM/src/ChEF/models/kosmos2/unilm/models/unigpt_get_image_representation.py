def get_image_representation(self, img_src_tokens):
    img_src_tokens = img_src_tokens.to(next(self.img_model.parameters()).device
        )
    img_output = self.img_model(img_src_tokens)
    src_len = img_output.size(0)
    img_output = img_output.transpose(0, 1)
    img_output = img_output.reshape(-1, img_output.size(-1))
    if self.img_connector is not None:
        img_output = self.img_connector(img_output, src_len=src_len)
    return img_output
