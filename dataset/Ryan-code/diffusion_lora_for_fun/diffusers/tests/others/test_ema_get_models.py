def get_models(self, decay=0.9999):
    unet = UNet2DConditionModel.from_pretrained(self.model_id, subfolder='unet'
        )
    unet = unet.to(torch_device)
    ema_unet = EMAModel(unet.parameters(), decay=decay, model_cls=
        UNet2DConditionModel, model_config=unet.config)
    return unet, ema_unet
