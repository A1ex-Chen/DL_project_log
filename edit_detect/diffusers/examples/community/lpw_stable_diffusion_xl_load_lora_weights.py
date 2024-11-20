def load_lora_weights(self, pretrained_model_name_or_path_or_dict: Union[
    str, Dict[str, torch.Tensor]], **kwargs):
    state_dict, network_alphas = self.lora_state_dict(
        pretrained_model_name_or_path_or_dict, unet_config=self.unet.config,
        **kwargs)
    self.load_lora_into_unet(state_dict, network_alphas=network_alphas,
        unet=self.unet)
    text_encoder_state_dict = {k: v for k, v in state_dict.items() if 
        'text_encoder.' in k}
    if len(text_encoder_state_dict) > 0:
        self.load_lora_into_text_encoder(text_encoder_state_dict,
            network_alphas=network_alphas, text_encoder=self.text_encoder,
            prefix='text_encoder', lora_scale=self.lora_scale)
    text_encoder_2_state_dict = {k: v for k, v in state_dict.items() if 
        'text_encoder_2.' in k}
    if len(text_encoder_2_state_dict) > 0:
        self.load_lora_into_text_encoder(text_encoder_2_state_dict,
            network_alphas=network_alphas, text_encoder=self.text_encoder_2,
            prefix='text_encoder_2', lora_scale=self.lora_scale)
